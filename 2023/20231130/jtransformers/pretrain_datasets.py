import os
import spacy
import random
import json
import torch

from utils import parse_csv_file_to_json


def Multi30k(language_pair=None):
    corpus_lines_train = []

    for lan in language_pair:
        with open('text/train.{}'.format(lan), 'r') as file:
            corpus_lines_train.append(file.read().splitlines())
        # end
    # end

    corpus_train = [(b,) for a, b in zip(*corpus_lines_train)]

    corpus_lines_eval = []

    for lan in language_pair:
        with open('text/val.{}'.format(lan), 'r') as file:
            corpus_lines_eval.append(file.read().splitlines())
        # end
    # end

    corpus_eval = [(b,) for a, b in zip(*corpus_lines_eval)]

    return corpus_train, corpus_eval, None


# end


def Quora(split=0.05):
    filename_quora = 'quora_duplicate_questions.tsv'

    contents_quora = parse_csv_file_to_json(filename_quora)
    list_corpus_quora = []
    for c in contents_quora:
        label = int(c['is_duplicate'])
        score = 1.0 if label else 0.5
        corpus_quora = (c['question1'], c['question2'], score)
        list_corpus_quora.append(corpus_quora)
    # end

    indexs_all = list(range(len(list_corpus_quora)))
    random.shuffle(indexs_all)

    index_split = int(split * len(list_corpus_quora))

    indexs_eval = indexs_all[:index_split]
    indexs_train = indexs_all[index_split:]

    list_corpus_eval = [list_corpus_quora[i_e] for i_e in indexs_eval]
    list_corpus_train = [list_corpus_quora[i_t] for i_t in indexs_train]

    return list_corpus_train, list_corpus_eval, None


# end

def BookCorpus2000(split=0.1):
    filename = 'bookcorpus_2000.json'

    with open(filename, 'r') as file:
        list_corpus = json.load(file)
    # end

    indexs_all = list(range(len(list_corpus)))
    random.shuffle(indexs_all)

    index_split = int(split * len(list_corpus))

    indexs_eval = indexs_all[:index_split]
    indexs_train = indexs_all[index_split:]

    list_corpus_eval = [list_corpus[i_e] for i_e in indexs_eval]
    list_corpus_train = [list_corpus[i_t] for i_t in indexs_train]

    return list_corpus_train, list_corpus_eval, None


# end


def BookCorpus2000_SC(split=0.1, labels=5):
    filename = 'bookcorpus_2000.json'

    with open(filename, 'r') as file:
        lines = json.load(file)
    # end

    list_corpus = [(line, random.randint(0, labels - 1)) for line in lines]

    indexs_all = list(range(len(list_corpus)))
    random.shuffle(indexs_all)

    index_split = int(split * len(list_corpus))

    indexs_eval = indexs_all[:index_split]
    indexs_train = indexs_all[index_split:]

    list_corpus_eval = [list_corpus[i_e] for i_e in indexs_eval]
    list_corpus_train = [list_corpus[i_t] for i_t in indexs_train]

    return list_corpus_train, list_corpus_eval, None
# end


def BookCorpus2000_Sim(split=0.1):
    filename = 'bookcorpus_2000.json'

    with open(filename, 'r') as file:
        lines = json.load(file)
    # end

    list_corpus = [(line, line, random.randint(0, 1)) for line in lines]

    indexs_all = list(range(len(list_corpus)))
    random.shuffle(indexs_all)

    index_split = int(split * len(list_corpus))

    indexs_eval = indexs_all[:index_split]
    indexs_train = indexs_all[index_split:]

    list_corpus_eval = [list_corpus[i_e] for i_e in indexs_eval]
    list_corpus_train = [list_corpus[i_t] for i_t in indexs_train]

    return list_corpus_train, list_corpus_eval, None
# end


def BookCorpus(split=0.0001, used=-1):
    import datasets

    list_corpus = datasets.load_dataset('bookcorpus')['train']['text'][:used]  # 70,000,000, 70 Million

    indexs_all = list(range(len(list_corpus)))
    random.shuffle(indexs_all)

    index_split = int(split * len(list_corpus))

    indexs_eval = indexs_all[:index_split]
    indexs_train = indexs_all[index_split:]

    list_corpus_eval = [list_corpus[i_e] for i_e in indexs_eval]
    list_corpus_train = [list_corpus[i_t] for i_t in indexs_train]

    return list_corpus_train, list_corpus_eval, None


# end


def load_vocab(filename_vocab):
    vocab_tgt = torch.load(filename_vocab)
    return vocab_tgt
# end

def load_spacy():
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_en
# end