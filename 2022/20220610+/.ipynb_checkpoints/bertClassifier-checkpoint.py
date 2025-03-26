import os, errno
import json
import yaml
import argparse
import sys
import traceback
import time
import pandas as pd
import torch
import torch.utils.data as data_utils
import transformers
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# default constants

# The name of the bert model to load
VBERT_MODEL = "vBERT-2020-Base"

# vBERT-Base's hidden layer size is 768
# vBERT-Large's hidden layer size is 1024
CLASSIFICATION_LAYER_WIDTH = 768 if "Base" in VBERT_MODEL else 1024

# Maximum string length
# Anything longer will be truncated
# Maximum length can not exceed 512, though generally can be shorter
MAX_STRING_LEN = 128


# ## Class for transforming text for input into BERT
# 
# This class tokenizes the text and converts the tokens into tensors of token ids with an attention mask, along with the string's class label

class Prepare(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data[1][index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.data[0][index]

        target = torch.tensor(label, dtype=torch.long)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': label
        }

    def __len__(self):
        return self.len

# ## vBERT Class for classification
# 
# To use vBERT for classification, you have to add a classification layer on top of the language model. 
# vBERT encodes the data.  The classification layer receives the encoding from vBERT and claculates class probabilities.

class BertClassifier(torch.nn.Module):
    # Create a classification network that uses vBERT to encode the input and pass to a classification layer
    #
    # Input configurable parameters:
    # root: full path name of the model directory, where the .pt, vocab and .model.json/bert_config.json file are found
    # name: name of the model (for reporting/diagnostic)
    # input_size: width of classification layer (e.g. 768)
    # output_size: number of output classes (task dependent)
    # max_length: maximum number of characters in input string for trainig/inference
    # classes: array of output_size with class names

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.config = config
        self.model_dir = config.get("root", ".")
        self.model_name = config.get("name", VBERT_MODEL)
        self.classification_layer_width = config.get("input_size", CLASSIFICATION_LAYER_WIDTH)
        self.classes = config.get("classes", ["0", "1"])
        self.num_classes = config.get("output_size", len(self.classes))
        self.max_string_length = config.get("max_length", MAX_STRING_LEN)
        self.bert_config_file = self.model_dir + "/bert_config.json"
        self.model_file_path = self.model_dir + "/model.pt"
        self.device = config.get("device", "cpu")
        self.metrics = [ "accuracy", "precision", "recall", "F-score" ]

        # Load the vBERT vocabulary into the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)

    def load_bert_for_training(self):
        print('Loading vBERT model: ' + self.model_name)
        self.l1 = BertModel.from_pretrained(pretrained_model_name_or_path=self.config.bert_model_dir)
        print("Adding {}x{} classification layer".format(self.classification_layer_width, self.num_classes))
        self.classifier = torch.nn.Linear(self.classification_layer_width, self.num_classes)

    def load_bert_for_inference(self):
        print('Loading vBERT config')
        self.l1 = BertModel(BertConfig.from_pretrained(self.bert_config_file))
        print("Adding {}x{} classification layer".format(self.classification_layer_width, self.num_classes))
        self.classifier = torch.nn.Linear(self.classification_layer_width, self.num_classes)

    # Encode the input with vBERT, read output from the last layer of vBERT, and pass to the classification layer
    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        output = self.classifier(pooler)
        return output

    def do_inference(self, data_loader, test_output=None):
        model = self.eval()
        predictions = []
        real_values = []
        with torch.no_grad():
            for data in data_loader:
                input_ids = data['ids'].to(self.device)
                attention_mask = data['mask'].to(self.device)
                targets = data['targets'].to(self.device)
                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds)
                real_values.extend(targets)
                if test_output != None:
                    test_output.extend(outputs)
        predictions = torch.stack(predictions).cpu()
        real_values = torch.stack(real_values).cpu()

        test_accu = 100 * accuracy_score(real_values, predictions)
        test_precision, test_recall, test_fscore, ignore = precision_recall_fscore_support(real_values, predictions, average='macro')
        test_precision *= 100
        test_recall *= 100
        test_fscore *= 100
        metrics = [ test_accu, test_precision, test_recall, test_fscore ]
        return predictions, real_values, metrics

    def classify_text(self, txt: str, conf=None):
        dataset = pd.DataFrame.from_dict({ 'row': [ 0, txt ]}, orient='index')
        prepared_set = Prepare(dataset, self.tokenizer, self.max_string_length)
        params = {'batch_size': 1,
                    'shuffle': False,
                    'num_workers': 0
                }
        loader = DataLoader(prepared_set, **params)
        outputs = []
        predictions, real_values, metrics = self.do_inference(loader, outputs)
        classes = self.classes
        result = { 'classes': [ ], 'text': txt, 'top_class': classes[int(predictions[0])], 'top_class_index': int(predictions[0]) }
        xi = 0
        for x in classes:
            result['classes'].append({ 'class_name': x, 'confidence': float(outputs[0][xi]) })
            xi += 1
        return result

class InstaMLConfig():
    def __init__(self, config_json_file):
        with open(config_json_file, "r") as cjf:
            self.jconf = json.load(cjf)
        pth, fname = os.path.split(config_json_file)
        if len(pth) < 1:
            pth = "."
        self.name = self.jconf.get('name', '')
        self.model_dir = pth

    def load_instaML_model_file(self):
        model_info_file_name = self.model_dir + "/.model.json"
        with open(model_info_file_name, "r") as mif:
            self.model_info = json.load(mif)
        return self.model_info

    def save_instaML_model_file(self):
        model_info_file_name = self.model_dir + "/.model.json"
        with open(model_info_file_name, "w") as mif:
            json.dump(self.model_info, mif, indent=4)
