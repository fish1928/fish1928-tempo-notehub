import json
import os
import sys
from datetime import datetime
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 512

FILENAME_TRAIN = 'train.csv'
FILENAME_TEST = 'test.csv'

DIR_OUTPUT = 'results'

def get_ts():
    return datetime.utcnow().replace(microsecond=0).isoformat()
# end


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

def read_passages(path_data, test_size):
    path_label = os.path.join('/'.join(path_data.split('/')[:-1]), 'labels.json')    #hot fix label

    
    df = pd.read_csv(path_data)

    documents = df['processed'].to_list()
    labels_str = df['target'].to_list()

    samples = documents

    
    with open(path_label, 'r') as file:
        labels_list = sorted(json.load(file))
    # end
    
    labels_all = {l: idx for idx, l in enumerate(labels_list)}
    
    labels = [labels_all[label_str] for label_str in labels_str]

    if test_size > 0:
        return train_test_split(samples, labels, test_size=test_size), labels_list
    else:
        return (samples, samples, labels, labels), labels_list
    # end
# end


def compute_metrics(pred):
    labels = pred.label_ids.reshape(-1)
    preds = pred.predictions.argmax(-1).reshape(-1)

    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')
    recall = recall_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, zero_division=1, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
# end

def main_train_and_evaluate(round, folder_data, eval_size):
    print('[{}] start main_train_and_evaluate with {} {} {}'.format(get_ts(), round, folder_data, eval_size))

    model_name = MODEL_NAME
    max_length = MAX_LENGTH
    output_dir = DIR_OUTPUT

    filename_train = FILENAME_TRAIN
    filename_test = FILENAME_TEST

    path_train_relative = os.path.join(folder_data, filename_train)
    (train_samples, valid_samples, train_labels, valid_labels), target_names = read_passages(path_train_relative, eval_size)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
    train_encodings = tokenizer.batch_encode_plus(train_samples, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    valid_encodings = tokenizer.batch_encode_plus(valid_samples, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    train_dataset = SimpleDataset(train_encodings, train_labels)
    valid_dataset = SimpleDataset(valid_encodings, valid_labels)

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=20,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=2,  # batch size for evaluation
        warmup_steps=0,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,
        # load the best model when finished training (default metric is loss)    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=1,  # log & save weights each logging_steps
        evaluation_strategy="epoch",  # evaluate each `logging_steps`
        learning_rate=2e-5,
        save_strategy='epoch',
        save_total_limit=7,
        metric_for_best_model='f1'
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)]
    )

    print('[{}] start training...'.format(get_ts()))
    trainer.train()

    info_state_model = trainer.evaluate()
    print('[{}] finish training.'.format(get_ts()))

    ################## start to do eval ##################

    path_test_relative = os.path.join(folder_data, filename_test)
    (samples_test, _, indexs_label_test, _), target_names = read_passages(path_test_relative, 0)
    labels_test = [target_names[index_label_test] for index_label_test in indexs_label_test]

    list_conf_output = []
    list_label_output = []

    for sample_test, label_origin in zip(samples_test, labels_test):
        input_tokenized = tokenizer.encode_plus(sample_test, padding=True, truncation=True, max_length=max_length,
                                                return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model(**input_tokenized, output_hidden_states=True, output_attentions=True)
        # end

        probas_evaluate = torch.nn.functional.softmax(out.logits, dim=-1)
        answer_evaluate = int(probas_evaluate.argmax().cpu())

        label_evaluate = target_names[answer_evaluate]

        list_conf_output.append(probas_evaluate.cpu().numpy().tolist()[0][answer_evaluate])
        list_label_output.append(label_evaluate)
    # end

    print('[{}] finish testing.'.format(get_ts()))

    average_conf = sum(list_conf_output) / len(list_conf_output)
    rate_highconf = len([conf_output for conf_output in list_conf_output if conf_output > 0.9]) / len(list_conf_output)
    rate_correct = len([True for label_output, label_test in zip(list_label_output, labels_test) if label_output == label_test]) / len(labels_test)

    info_final = {
        'state_model': info_state_model,
        'average_conf': average_conf,
        'rate_highconf': rate_highconf,
        'rate_correct': rate_correct,
        'samples': len(samples_test)
    }

    filename_output = f'output-{eval_size}-{round}.json'
    path_file_output = os.path.join(folder_data, filename_output)
    with open(path_file_output, 'w+') as file:
        file.write(json.dumps(info_final))
    # end

    print('[{}] main_train_and_evaluate finished.'.format(get_ts()))
# end

if __name__ == '__main__':
    filename_config = sys.argv[1]
    with open(filename_config, 'r') as file:
        list_config_run = json.load(file)
    # end

    for config_run in list_config_run:
        if config_run['round_current'] >= config_run['round_max']:
            continue
        else:
            folder_data = config_run['folder_data']
            eval_size = config_run['eval_size']
            round_current = config_run['round_current']
            round_max = config_run['round_max']
            if round_current < round_max:
                main_train_and_evaluate(round_current, folder_data, eval_size)
            # end

            round_current += 1
            config_run['round_current'] = round_current
            break
        # end
    # end

    with open(filename_config, 'w+') as file:
        file.write(json.dumps(list_config_run))
    # end

    print('[{}] all finished.'.format(get_ts()))
# end