import pandas as pd
import os
import torch
import transformers
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    # end
# end

def compute_metrics(pred):
    labels = pred.label_ids.reshape(-1)
    # pred = np.argmax(pred, axis=1)
    preds = pred.predictions.argmax(-1).reshape(-1)

    # print('labels: {}'.format(labels))
    # print('pred: {}'.format(preds))
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
# end

def read_passages(path_data, test_split=0.8):
    df_train = pd.read_csv(path_data)
    anchors_all = df_train['anchor'].to_list()
    documents_all = df_train['log'].to_list()
    labels_str_all = df_train['label'].to_list()
    
    samples_all = [(anchor, document) for anchor, document in zip(anchors_all,documents_all)]
    
    labels_index = sorted(list(set(labels_str_all)))
    index_labels = {l:idx for idx, l in enumerate(labels_index)}
    labels_all = [ index_labels[label_str] for label_str in labels_str_all]
    
    ids_sample = range(len(anchors_all))
    ids_train, ids_others, labels_train, labels_others = train_test_split(ids_sample, labels_all, stratify=labels_all, test_size=test_split)
    ids_eval, ids_test, labels_eval, labels_test = train_test_split(ids_others, labels_others, stratify=labels_others, test_size=0.5)
    
    samples_train = [samples_all[id_train] for id_train in ids_train]
    samples_eval = [samples_all[id_eval] for id_eval in ids_eval]
    samples_test = [samples_all[id_test] for id_test in ids_test]

    return samples_train, samples_eval, samples_test, labels_train, labels_eval, labels_test, labels_index
# end


model_name = "distilbert-base-uncased"
model_dir = 'model_esxdeploy_m1_82'
max_length = 512
dir_data = 'datasource'
filename = 'nimbus_trainingdataset_pair_20221207.csv'


tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

path_file = os.path.join(dir_data, filename)

train_samples, valid_samples, test_samples, train_labels, valid_labels, test_labels, target_names = read_passages(path_file)


train_encodings = tokenizer.batch_encode_plus(train_samples, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
valid_encodings = tokenizer.batch_encode_plus(valid_samples, truncation=True, padding=True, max_length=max_length, return_tensors='pt')


train_dataset = SimpleDataset(train_encodings, train_labels)
valid_dataset = SimpleDataset(valid_encodings, valid_labels)


if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
    print('load model from local')
    model_info = model_dir
else:
    print('load model from official')
    model_info = model_name
    
model = DistilBertForSequenceClassification.from_pretrained(model_info, num_labels=len(target_names))



training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=1,               # log & save weights each logging_steps
    evaluation_strategy="epoch",     # evaluate each `logging_steps`
    learning_rate=2e-5,
    save_strategy='epoch',
    save_total_limit=20,
    metric_for_best_model='f1'
)


trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

print('[debug] jinyuj: running on device {}'.format(torch.cuda.get_device_name(0)))
trainer.train()



