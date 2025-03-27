import torch
import os
from os.path import exists
import torch.nn as nn
# from torch.nn.functional import log_softmax, pad, one_hot
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torch.utils.data import DataLoader
import random
import json
import csv
from pathlib import Path
import shutil
import re
import threading
from accelerate import Accelerator

### utils.py ###

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    def __iadd__(self, other):
        for k, v in self.items():
            if k in other and other[k]:
                self[k] += other[k]
            # end
        # end

        return self
    # end
# end


# Takes the file paths as arguments
def parse_csv_file_to_json(path_file_csv):
    # create a dictionary
    elements = []

    # Open a csv reader called DictReader
    with open(path_file_csv, encoding='utf-8') as file_csv:
    #with open(path_file_csv) as file_csv:
        reader_csv = csv.DictReader(file_csv, delimiter="\t")

        # Convert each row into a dictionary
        # and add it to data
        for dict_head_value in reader_csv:
            element = {}

            for head, value in dict_head_value.items():
                if value and (value[0] in ["[", "{"]):
                    element[head] = value
                else:
                    element[head] = value

            elements.append(element)
        # end
    # end

    return elements
# end

### utils.py ###



### core.py ###

"Produce N identical layers."
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
# end


class MultiHeadedAttention(nn.Module):

    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    # end


    "Compute 'Scaled Dot Product Attention'"
    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # print('jinyuj: scores: {}, mask: {}'.format(scores.shape, mask.shape))
            scores = scores.masked_fill(mask == 0, -1e9)
        # end
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # end
        return torch.matmul(p_attn, value), p_attn
    # end


    "Implements Figure 2"
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
    # end
# end class


"""
A residual connection followed by a layer norm.
Note for code simplicity the norm is first as opposed to last.
"""
class ResidualLayer(nn.Module):

    def __init__(self, size, dropout=0.1, eps=1e-6):
        super(ResidualLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(size, eps)
        self.dropout = nn.Dropout(p=dropout)
    # end

    "Apply residual connection to any sublayer with the same size."
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    # end
# end class


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    # end
# end


class SimpleIDEmbeddings(nn.Module):
    def __init__(self, size_vocab, dim_hidden, id_pad):
        super(SimpleIDEmbeddings, self).__init__()
        self.lut = nn.Embedding(size_vocab, dim_hidden, padding_idx=id_pad)
        self.dim_hidden = dim_hidden

    def forward(self, x):
        result = self.lut(x)
        return result * math.sqrt(self.dim_hidden)
    # end

    def get_shape(self):
        return (self.lut.num_embeddings, self.lut.embedding_dim)
    # end
# end


"Implement the PE function."
class PositionalEncoding(nn.Module):

    def __init__(self, dim_positional, max_len=512):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        self.dim_positional = dim_positional
        pe = torch.zeros(max_len, dim_positional)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_positional, 2) * -(math.log(10000.0) / dim_positional)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to('cuda')
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
    # end
# end


class SimpleEmbedder(nn.Module):    # no segment embedder as we do not need that
    def __init__(self, size_vocab=None, dim_hidden=128, dropout=0.1, id_pad=0):
        super(SimpleEmbedder, self).__init__()
        self.size_vocab = size_vocab
        self.dim_hidden = dim_hidden
        self.id_pad = id_pad

        self.embedder = nn.Sequential(
            SimpleIDEmbeddings(size_vocab, dim_hidden, id_pad),
            PositionalEncoding(dim_hidden),
            nn.Dropout(p=dropout)
        )
    # end

    def forward(self, ids_input):   # (batch, seqs_with_padding)
        return self.embedder(ids_input)
    # end

    def get_vocab_size(self):
        return self.size_vocab
    # end
# end

### core.py ###



class SimpleEncoderLayer(nn.Module):

    def __init__(self, dim_hidden, dim_feedforward, n_head, dropout=0.1):
        super(SimpleEncoderLayer, self).__init__()

        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.dim_feedforward = dim_feedforward

        self.layer_attention = MultiHeadedAttention(n_head, dim_hidden)
        self.layer_feedforward = PositionwiseFeedForward(dim_hidden, dim_feedforward, dropout)
        self.layers_residual = clones(ResidualLayer(dim_hidden, dropout), 2)
    # end

    def forward(self, embeddings, masks, *args):
        embeddings = self.layers_residual[0](embeddings, lambda embeddings: self.layer_attention(embeddings, embeddings, embeddings, masks))
        return self.layers_residual[1](embeddings, self.layer_feedforward)
    # end
# end



class SimpleDecoderLayer(nn.Module):

    def __init__(self, dim_hidden, dim_feedforward, n_head, dropout=0.1):
        super(SimpleDecoderLayer, self).__init__()

        self.n_head = n_head
        self.dim_hidden = dim_hidden
        self.dim_feedforward = dim_feedforward

        self.layer_attention_decoder = MultiHeadedAttention(n_head, dim_hidden)
        self.layer_attention_encoder = MultiHeadedAttention(n_head, dim_hidden)
        self.layer_feedforward = PositionwiseFeedForward(dim_hidden, dim_feedforward, dropout)
        self.layers_residual = clones(ResidualLayer(dim_hidden, dropout), 3)

    def forward(self, embeddings, masks_encoder, output_encoder, masks_decoder, *args):
        embeddings = self.layers_residual[0](embeddings, lambda embeddings: self.layer_attention_decoder(embeddings, embeddings, embeddings, masks_decoder))
        embeddings = self.layers_residual[1](embeddings, lambda embeddings: self.layer_attention_encoder(embeddings, output_encoder, output_encoder, masks_encoder))
        return self.layers_residual[2](embeddings, self.layer_feedforward)
    # end
# end


class SimpleTransformerStack(nn.Module):

    def __init__(self, obj_layer, n_layers):
        super(SimpleTransformerStack, self).__init__()
        self.layers = clones(obj_layer, n_layers)

        self.norm = torch.nn.LayerNorm(obj_layer.dim_hidden)
    # end

    def forward(self, embedding_encoder=None, masks_encoder=None, output_encoder=None, embedding_decoder=None, masks_decoder=None ,noncache=False, **kwargs):  # input -> (batch, len_seq, vocab)

        if output_encoder is not None and embedding_decoder is not None and masks_decoder is not None:
            embeddings = embedding_decoder
        else:
            embeddings = embedding_encoder
        # end

        for layer in self.layers:
            embeddings = layer(embeddings, masks_encoder, output_encoder, masks_decoder)
        # end

        output = self.norm(embeddings)
        return output
    # end

# end


class SimpleEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, embedder_encoder, embedder_decoder, pooling=False):
        super(SimpleEncoderDecoder, self).__init__()

        self.pooling = pooling
        
        self.embedder_encoder = embedder_encoder
        self.encoder = encoder

        self.embedder_decoder = embedder_decoder
        self.decoder = decoder

    # end

    def forward(self, ids_encoder=None, masks_encoder=None, ids_decoder=None, masks_decoder=None, **kwargs):
        output_encoder = None
        output_encoder_pooled = None
        output_decoder = None
        
        output_encoder = self.embed_and_encode(ids_encoder=ids_encoder, masks_encoder=masks_encoder)
        output = output_encoder
        
        if self.pooling:
            output_encoder_refilled = output_encoder.masked_fill(masks_encoder.transpose(-1,-2)==False, 0)
            output_encoder_pooled = torch.mean(output_encoder_refilled, dim=-2)
            
            output_encoder_pooled_expanded = output_encoder_pooled.unsqueeze(-2).expand(output_encoder.shape)
            output = output_encoder_pooled_expanded
        # end
        
        if self.embedder_decoder and self.decoder:
            output_decoder = self.embed_and_decode(ids_decoder=ids_decoder, masks_encoder=masks_encoder, output_encoder=output, masks_decoder=masks_decoder)
        # end if
        
        return {'output_encoder': output_encoder, 'output_encoder_pooled': output_encoder_pooled, 'output_decoder': output_decoder}
    # end
    
    def embed_and_encode(self, ids_encoder=None, masks_encoder=None, **kwargs):
        
        embedding_encoder = self.embedder_encoder(ids_encoder)
        output_encoder = self.encoder(
            embedding_encoder=embedding_encoder,
            masks_encoder=masks_encoder,
        )
        
        return output_encoder
    # end

    
    def embed_and_decode(self, ids_decoder=None, masks_encoder=None, output_encoder=None, masks_decoder=None, **kwargs):
        
        embedding_decoder = self.embedder_decoder(ids_decoder)
        output_decoder = self.decoder(
            masks_encoder=masks_encoder,
            output_encoder=output_encoder,    #(len_seq, dim_hidden) -> (1, dim_hidden)
            embedding_decoder=embedding_decoder,
            masks_decoder=masks_decoder,
        )

        return output_decoder
    # end
    

    def get_vocab_size(self, name_embedder):
        embedder = getattr(self, f'embedder_{name_embedder}')
        return embedder.get_vocab_size()
    # end

# end

class LinearAndNorm(nn.Module):
    def __init__(self, dim_in = None, dim_out = None, dropout=0.1, eps_norm=1e-12):
        super(LinearAndNorm, self).__init__()

        self.linear = torch.nn.Linear(dim_in, dim_out)
        self.norm = torch.nn.LayerNorm(dim_out, eps_norm)
        self.dropout = torch.nn.Dropout(p=dropout)
    # end

    def forward(self, seqs_in):
        return self.dropout(self.norm(self.linear(seqs_in).relu()))
    # end
# end




class Batch:

    def __init__(self, **kwargs):
        self.kwargs = {}
        for k, v in kwargs.items():
            if v is not None and type(v) is not bool:
                self.kwargs[k] = v.cuda()
            # end
        # end
        
    # end

    def __call__(self):
        return self.kwargs
    # end
# end



class Collator_Base:

    def __init__(self, tokenizer, size_seq_max, need_masked=0.3):
        self.tokenizer = tokenizer
        self.size_seq_max = size_seq_max
        self.need_masked = need_masked

        index_special_token_2_id = {k: v for k, v in zip(tokenizer.all_special_tokens, tokenizer.all_special_ids)}

        self.id_pad = index_special_token_2_id['[PAD]']
        self.id_mask = index_special_token_2_id['[MASK]']
        self.id_cls = index_special_token_2_id['[CLS]']
        self.id_sep = index_special_token_2_id['[SEP]']
        self.id_unk = index_special_token_2_id['[UNK]']
        
        self.regex_special_token = re.compile(r'\[(PAD|MASK|CLS|SEP|EOL|UNK)\]')
        
        self.index_randtoken_start = 999
        self.index_randtoken_end = 30521
    # end

    def _preprocess(self, line):
        line = re.sub(self.regex_special_token, r'<\1>', line)
        line = re.sub(r'''('|"|`){2}''', '', line)
        line = re.sub(r'\.{2,3}', '', line)
        line = re.sub(r' {2,}', ' ', line)
        line = line.lstrip().rstrip()
        return line
    # end
    
    def _get_random_tokens(self):
        return random.randint(self.index_randtoken_start, self.index_randtoken_end)
    # end

    
    def pad_sequences(self, sequences, size_seq_max, need_diagonal=False,
                      need_masked=0):  # need_diagonal and need_masked cannot both set, one for bert seq one for s2s seq
        
        sequences = copy.deepcopy(sequences)
        
        id_pad = self.id_pad
        id_mask = self.id_mask

        sequences_masked_padded = []
        labels_padded = []

        for sequence in sequences:

            len_seq = len(sequence)
            label = copy.deepcopy(sequence)

            if need_masked:
                indexs_masked = list(range(1, len_seq - 1))  # 0 = cls, -1 = sep
                random.shuffle(indexs_masked)
                anchor_mask_all = round(need_masked * (len_seq - 2)) or 1
                anchor_mask_replace = int(anchor_mask_all / 2)

                if anchor_mask_replace:  # not 0
                    indexs_replaced = indexs_masked[:anchor_mask_replace]
                    for index_replaced in indexs_replaced:
                        sequence[index_replaced] = self._get_random_tokens()
                    # end
                # end

                indexs_masked = indexs_masked[anchor_mask_replace:anchor_mask_all]
            # end


            count_pad = size_seq_max - len_seq
            
            label = torch.LongTensor(label)
            label_padded = torch.cat((label, torch.LongTensor([id_pad] * count_pad)))
            labels_padded.append(label_padded)

            if need_masked:

                sequence_masked = torch.LongTensor(sequence)
                sequence_masked.index_fill_(0, torch.LongTensor(indexs_masked), id_mask)
                sequence_masked_padded = torch.cat((sequence_masked, torch.LongTensor([id_pad] * count_pad)))

                sequences_masked_padded.append(sequence_masked_padded)
            # end
        #   # end for

        inputs = torch.stack(labels_padded)  # (batch, size_seq_max)
        if need_masked:
            inputs_masked_padded = torch.stack(sequences_masked_padded)
        # end

        masks_segment = (inputs != self.id_pad).unsqueeze(-2)  # (nbatch, 1, seq)
        masks_attention = self.make_std_mask(inputs, self.id_pad) if need_diagonal else masks_segment

        if need_masked:
            masks_masked = (inputs_masked_padded != id_mask).unsqueeze(-2)
            masks_attention = masks_attention & masks_masked
            return inputs_masked_padded, masks_attention, masks_segment, inputs  # (inputs, masks_attention, masks_segment, labels)
        else:
            return inputs, masks_attention, masks_segment, None
        # end
    # end


    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & self.subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask
    # end
# end



class Collator_BERT_Encoded_254(Collator_Base):

    def __call__(self, list_tokenized_merged):
        # for tokenized in list_tokenized_merged:
            # if type(tokenized) == tuple:
                # return None        
        if type(list_tokenized_merged[0]) == tuple:
            return None
        # end
        
        len_tokenized_accumulated = 2  # add cls and sep
        list_tokenized_merged = [tokenized_merged[:self.size_seq_max - len_tokenized_accumulated] for tokenized_merged in list_tokenized_merged]

        # Process III. Add begin and stop special token, same as jinyuj_transformers_quora.ipynb
        tokens_input_encoder = []
        tokens_input_decoder = []
        tokens_label_decoder = []

        for tokenized_merged in list_tokenized_merged:
            if type(tokenized_merged) != tuple:
                tokens_input_encoder.append([self.id_cls] + tokenized_merged + [self.id_sep])
                tokens_input_decoder.append([self.id_cls] + tokenized_merged)
                tokens_label_decoder.append(tokenized_merged + [self.id_sep])

        # for tokenized_merged in list_tokenized_merged:
            # tokens_input_encoder.append([self.id_cls] + tokenized_merged + [self.id_sep])
            # tokens_input_decoder.append([self.id_cls] + tokenized_merged)
            # tokens_label_decoder.append(tokenized_merged + [self.id_sep])
        # end

        inputs_encoder, masks_encoder, segments_encoder, labels_encoder = self.pad_sequences(tokens_input_encoder,
                                                                                             self.size_seq_max,
                                                                                             need_masked=self.need_masked)
        inputs_decoder, masks_decoder, segments_decoder, _ = self.pad_sequences(tokens_input_decoder, self.size_seq_max,
                                                                                need_diagonal=True)
        labels_decoder, masks_label, segments_label, _ = self.pad_sequences(tokens_label_decoder, self.size_seq_max)

        return Batch(
            ids_encoder=inputs_encoder,  # contains [mask]s
            masks_encoder=masks_encoder,
            labels_encoder=labels_encoder,  # doesn't contain [mask]
            segments_encoder=segments_encoder,
            ids_decoder=inputs_decoder,
            masks_decoder=masks_decoder,
            labels_decoder=labels_decoder,
            segments_label=segments_label
        )
    # end
# end



class SimpleEncodedDataset(torch.utils.data.Dataset):

    # info_file_rows = {'path_file': 1,000,000,...}
    def __init__(self, folder_dataset_base, info_file_rows, split=0.001):
        self.folder_dataset_base = folder_dataset_base
        self.list_tokenized_eval = []
        self.dict_filename_loaded = {filename: False for filename, num_rows in info_file_rows.items()}
        self.list_corpus_idx_filename_train = []

        for filename, num_lines in info_file_rows.items():
            idxs_eval = list(range(num_lines))
            random.shuffle(idxs_eval)
            idxs_eval = idxs_eval[:round(len(idxs_eval) * split)]

            for idx_eval in idxs_eval:
                self.list_tokenized_eval.append((idx_eval, filename))
            # end

            set_idxs_eval = set(idxs_eval)
            for idx_train in range(num_lines):
                if idx_train in set_idxs_eval:
                    continue
                # end

                self.list_corpus_idx_filename_train.append((idx_train, filename))
            # end
        # end

        self.is_train = True
        self.rows_cached = []
        self.filename_cached = None
        self.idx_restored = -1
        self.idx_current = -1
    # end

    def restore(self, idx_restored=-1):
        self.idx_restored = idx_restored
    # end


    def __getitem__(self, idx):  # should not have problem now
        # if eval, use all cached eval tokenized
        if not self.is_train:
            return self.list_tokenized_eval[idx]
        # end
        
        if idx < self.idx_restored:
            return (None, None) # same to eval for collator to skip
        # end

        self.idx_current = idx

        # if train
        idxs_in_file, filename_current = self.list_corpus_idx_filename_train[idx]

        # if file not fully used
        if filename_current != self.filename_cached:

            # load new file
            print('switch from {} to {}'.format(self.filename_cached, filename_current))
            path_file = os.path.join(self.folder_dataset_base, filename_current)
            with open(path_file, 'r') as file:  # update rows_cached
                self.rows_cached = file.read().splitlines()
            # end

            self.filename_cached = filename_current

            if not self.dict_filename_loaded[filename_current]:
                for id_list_eval, tokenized_eval in enumerate(self.list_tokenized_eval):
                    if type(tokenized_eval) is tuple:
                        if tokenized_eval[1] == filename_current:
                            self.list_tokenized_eval[id_list_eval] = self._fransfer_one_line_to_tokenized(self.rows_cached[tokenized_eval[0]])
                        # end
                    # end
                # end
                self.dict_filename_loaded[filename_current] = True
            # end
        # end

        return self._fransfer_one_line_to_tokenized(self.rows_cached[idxs_in_file])
    # end

    def __len__(self):
        if self.is_train:
            return len(self.list_corpus_idx_filename_train)
        else:
            return len(self.list_tokenized_eval)
        # end
    # end

    def _fransfer_one_line_to_tokenized(self, str_line):
        tokenized = [int(t) for t in str_line.split(', ') if t]
        return tokenized
    # end

    def train(self):
        self.is_train = True
    # end

    def eval(self):
        self.is_train = False
        self.idx_restored = -1
    # end
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



def BookCorpus(split=0.0001, used=-1):
    import datasets
    
    list_corpus = datasets.load_dataset('bookcorpus')['train']['text'][:used]   # 70,000,000, 70 Million
    
    indexs_all = list(range(len(list_corpus)))
    random.shuffle(indexs_all)
    
    index_split = int(split * len(list_corpus))
    
    indexs_eval = indexs_all[:index_split]
    indexs_train = indexs_all[index_split:]
    
    list_corpus_eval = [list_corpus[i_e] for i_e in indexs_eval]
    list_corpus_train = [list_corpus[i_t] for i_t in indexs_train]
    
    return list_corpus_train, list_corpus_eval, None
# end


class SaverAndLoader:
    def __init__(self, path_checkpoints='./checkpoints'):
        self.dict_name_item = {}
        self.path_checkpoints = path_checkpoints
        self.metadata = None
    # end
    
    def add_item(self, item, name=None):
        if not name:
            name = item.__class__.__name__
        # end
        
        self.dict_name_item[name] = item
        return self
    # end
    
    
    def update_checkpoint(self, name_checkpoint, name_checkpoint_previous=None, metadata=None):  # epoch_n
        if not self.dict_name_item:
            print(f'[ALERT] no item added, skip saving checkpoint.')
            return
        # end
        
        if name_checkpoint_previous:
            result = self._delete_checkpoint_folder(name_checkpoint_previous)
            if result:
                print(f'[INFO] {name_checkpoint_previous} is cleared.')
            else:
                print(f'[ALERT] {name_checkpoint_previous} fail to be cleared.')
            # end
        # end
        
        folder_checkpoint = self._create_checkpoint_folder(name_checkpoint)
        for name_item, item in self.dict_name_item.items():
            path_checkpoint_item = os.path.join(folder_checkpoint, f'{name_item}.pt')
            torch.save(item.state_dict(), path_checkpoint_item)
            
            size_file_saved_MB = os.path.getsize(path_checkpoint_item) / 1024 / 1024
            print(f'[INFO] {name_item} is saved, {size_file_saved_MB} MB')
        # end
        
        if metadata:
            path_file_metadata = os.path.join(folder_checkpoint, 'metadata.json')
            with open(path_file_metadata,'w+') as file:
                file.write(json.dumps(metadata, indent=4))
            # end
            print(f'[INFO] metadata updated at {path_file_metadata}, : {metadata}')
            self.metadata = metadata
        # end
        
        print(f'[INFO] {name_checkpoint} is saved')
    # end

    
    def load_item_state(self, name_checkpoint, instance_item, name_item=None):
        if not name_item:
            name_item = instance_item.__class__.__name__
        # end
        
        path_checkpoint_item = os.path.join(self.path_checkpoints, name_checkpoint, f'{name_item}.pt')
        if not os.path.exists(path_checkpoint_item):
            print(f'[ERROR] {path_checkpoint_item} not exists')
            return None
        # end
        if issubclass(instance_item.__class__, torch.nn.Module):
            instance_item.load_state_dict(torch.load(path_checkpoint_item), strict=False)
        else:
            instance_item.load_state_dict(torch.load(path_checkpoint_item))
        # end
        
        print(f'[INFO] {name_item} loaded for {name_checkpoint}.')
        return instance_item
    # end
    
    def load_metadata(self, name_checkpoint):
        path_folder_checkpoint = os.path.join(self.path_checkpoints, name_checkpoint)
        path_metadata = os.path.join(path_folder_checkpoint, 'metadata.json')
        
        if os.path.exists(path_metadata):
            with open(path_metadata, 'r') as file:
                self.metadata = json.load(file)
            # end
            print(f'[INFO] {path_metadata} loaded: {self.metadata}')
        else:
            print(f'[WARN] no metadata found.')
        # end
    # end
    
    
    def list_items(self):
        return list(self.dict_name_item.keys())
    # end
    
    def _create_checkpoint_folder(self, name_checkpoint):
        path_folder_target = os.path.join(self.path_checkpoints, name_checkpoint)
        Path(path_folder_target).mkdir(parents=True, exist_ok=True)
        return path_folder_target
    # end
    
    def _delete_checkpoint_folder(self, name_checkpoint_previous):
        path_folder_target = os.path.join(self.path_checkpoints, name_checkpoint_previous)
        if os.path.exists(path_folder_target):
            shutil.rmtree(path_folder_target, ignore_errors=True)
        # end
        return (not os.path.exists(path_folder_target))
    # end
# end



class SimpleEncoderHead_MLM(nn.Module):

    @classmethod
    def get_info_accuracy_template(cls):
        return Dotdict({
            'corrects_segmented': 0,
            'corrects_masked': 0,
            'num_segmented': 0,
            'num_masked': 0 
        })
    # end
    
    def __init__(self, model, size_vocab, dim_hidden=128, dropout=0.1):
        super(SimpleEncoderHead_MLM, self).__init__()
        
        self.ffn = LinearAndNorm(dim_in=dim_hidden, dim_out=dim_hidden, dropout=dropout)
        self.extractor = torch.nn.Linear(dim_hidden, size_vocab, bias=False)
        self.extractor.weight = nn.Parameter(model.embedder_encoder.embedder[0].lut.weight)

        self.func_loss = torch.nn.CrossEntropyLoss().cuda()
    # end


    def forward(self, output_encoder=None, labels_encoder=None, segments_encoder=None, masks_encoder=None, **kwargs):   # labels_input -> (batch, seq, labels)
        output_ffn = self.ffn(output_encoder)
        output_mlm = self.extractor(output_ffn) # output_mlm = prediction_logits
        
        return {'output': output_mlm, 'labels_encoder': labels_encoder, 'segments_encoder': segments_encoder, 'masks_encoder': masks_encoder}


    
    def compute_loss(self, output=None, labels_encoder=None, segments_encoder=None, masks_encoder=None):
        
        output_mlm = output
        labels_mlm = labels_encoder
        
        info_acc = SimpleEncoderHead_MLM.get_info_accuracy_template()
        
        segments_encoder_2d = segments_encoder.transpose(-1,-2)[:,:,0]
        hidden_mlm_segmented = output_mlm.masked_select(segments_encoder_2d.unsqueeze(-1)).reshape(-1, output_mlm.shape[-1]) # should be (segmented_all_batchs, size_vocab)
        
        loss_segments = self.func_loss(hidden_mlm_segmented, labels_mlm.masked_select(segments_encoder_2d))
        info_acc.corrects_segmented = torch.sum(hidden_mlm_segmented.argmax(-1) == labels_mlm.masked_select(segments_encoder_2d)).cpu().item()
        info_acc.num_segmented = hidden_mlm_segmented.shape[0]
        
        masks_masked = torch.logical_xor(masks_encoder, segments_encoder) & segments_encoder # True is masked
        masks_masked_perbatch = masks_masked[:,0,:]
        hidden_mlm_masked = output_mlm.masked_select(masks_masked_perbatch.unsqueeze(-1)).reshape(-1, output_mlm.shape[-1])

        if hidden_mlm_masked.shape[0] != 0:
            loss_masked = self.func_loss(hidden_mlm_masked, labels_mlm.masked_select(masks_masked_perbatch))       
            info_acc.corrects_masked = torch.sum(hidden_mlm_masked.argmax(-1) == labels_mlm.masked_select(masks_masked_perbatch)).cpu().item()
            info_acc.num_masked = hidden_mlm_masked.shape[0]
        else:
            loss_masked = 0
            info_acc.corrects_masked = 0
            info_acc.num_masked = 1
        # end
        
        loss_mlm = loss_segments + loss_masked * 3
        
        return loss_mlm, info_acc
    # end
# end


class SimpleDecoderHead_S2S(nn.Module):

    @classmethod
    def get_info_accuracy_template(cls):
        return Dotdict({
            'corrects_segmented': 0,
            'num_segmented': 0 
        })
    # end


    def __init__(self, model, size_vocab, dim_hidden=128, dropout=0.1):
        super(SimpleDecoderHead_S2S, self).__init__()
        
        self.ffn = LinearAndNorm(dim_in=dim_hidden, dim_out=dim_hidden, dropout=dropout)
        self.extractor = torch.nn.Linear(dim_hidden, size_vocab, bias=False)
        self.extractor.weight = nn.Parameter(model.embedder_decoder.embedder[0].lut.weight)

        self.func_loss = torch.nn.CrossEntropyLoss().cuda()
    # end


    def forward(self, output_decoder=None, labels_decoder=None, segments_label=None, **kwargs):   # labels_input -> (batch, seq, labels)
        
        output_ffn = self.ffn(output_decoder)
        output_s2s = self.extractor(output_ffn)   # output_mlm = prediction_logits
        
        return {'output': output_s2s, 'labels_decoder': labels_decoder, 'segments_label': segments_label}
    # end


    def compute_loss(self, output=None, labels_decoder=None, segments_label=None):
        output_s2s = output
        labels_s2s = labels_decoder
        
        info_acc = SimpleDecoderHead_S2S.get_info_accuracy_template()
        
        segments_label_2d = segments_label.transpose(-1,-2)[:,:,0]
        hidden_s2s_segmented = output_s2s.masked_select(segments_label_2d.unsqueeze(-1)).reshape(-1, output_s2s.shape[-1])

        loss_segments = self.func_loss(hidden_s2s_segmented, labels_s2s.masked_select(segments_label_2d))
        info_acc.corrects_segmented = torch.sum(hidden_s2s_segmented.argmax(-1) == labels_s2s.masked_select(segments_label_2d)).cpu().item()
        info_acc.num_segmented = hidden_s2s_segmented.shape[0]
        
        return loss_segments * 4, info_acc
    # end

# end


class Trainer(nn.Module):
    def __init__(self, model):
        super(Trainer, self).__init__()
        self.index_name_head = set()
        self.model = model
    # end

    def register(self, head):
        name_head = head.__class__.__name__
        setattr(self, name_head, head)
        self.index_name_head.add(name_head)
        return self
    # end

    def forward(self, **kwargs):
        output_model = self.model(**kwargs)
        dict_head_output = {}
        
        for name in self.index_name_head:
            head = getattr(self, name)
            dict_head_output[name] = head.forward(**{**output_model, **kwargs})
        # end
        
        return dict_head_output
    # end

    def get_head(self, name_klass):
        if type(name_klass) is type:
            name_klass = klass.__name__
        # end
        
        return getattr(self, name_klass)
    # end
# end


class Builder:
    
    @classmethod
    def build_model_with_mlm_v2(cls, size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
        embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
        encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

        model = SimpleEncoderDecoder(encoderstack, None, embedder_encoder, None)
        head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)

        trainer = Trainer(model).register(head_mlm)

        return trainer
    # end
    
    @classmethod
    def build_model_with_s2s_v2(cls, size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
        embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
        encoderstack = SimpleTransformerStack(sample_encoder, n_layer)
        
        embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
        decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

        model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
        head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)
        
        manager = HeadManager().register(head_s2s)
        trainer = Trainer(model=model, manager=manager)

        return trainer
    # end
    
    @classmethod
    def build_model_with_2heads(cls, size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
        embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
        encoderstack = SimpleTransformerStack(sample_encoder, n_layer)
        
        embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
        decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

        model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
        head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)
        head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)

        trainer = Trainer(model).register(head_mlm).register(head_s2s)
        return trainer
    # end
    
    @classmethod
    def load_model_with_2heads(cls, size_vocab, dim_hidden, dim_feedforward, n_head, n_layer, loader, name_checkpoint):
        embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
        encoderstack = SimpleTransformerStack(sample_encoder, n_layer)
        
        embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
        sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
        decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

        model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
        head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)
        head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)
        
        loader.add_item(model)
        loader.add_item(head_s2s)
        loader.add_item(head_mlm)
        
        trainer = Trainer(model).register(head_mlm).register(head_s2s)
        
        for p in trainer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # end
        # end
        
        loader.load_item_state(name_checkpoint, model)
        loader.load_item_state(name_checkpoint, head_s2s)
        loader.load_item_state(name_checkpoint, head_mlm)
        loader.load_metadata(name_checkpoint)

        return trainer
    # end

# end

import re
import json
import transformers
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import to_map_style_dataset
from transformers import AutoTokenizer


GPUS = [0]
# GPUS = [0,1]
# torch.cuda.set_device(GPUS[0])

accelerator = Accelerator()
# source
seq_max = 256
batch_size = 12


# model & head
n_head = 12
dim_per_head = 64
dim_hidden = n_head * dim_per_head # 768
dim_feedforward = dim_hidden * 4 # 3072

n_layer = 6

# optimizer
lr_base_optimizer = 1e-4
betas_optimizer = (0.9, 0.999)
eps_optimizer = 1e-9

# scheduler
warmup = 200

# epochs
steps_per_save = 80000
epochs = 3
epoch_last = 2


# saver
name_epoch_last = f'epoch_{epoch_last}'
folder_base = 'checkpoints_0'


# dataset
# folder_dataset = 'bookcorpus_merged_254_10k'
folder_dataset = 'wikipassage_merged_254_1m'
filenames_dataset = sorted([f for f in os.listdir(folder_dataset) if f[0] != '.'], key=lambda name: int(name.split('.')[0]))
# list_size_per_file = [1000000, 1000000, 1000000, 1000000, 237940]
list_size_per_file = [
    1000000,1000000,1000000,1000000,1000000,
    1000000,1000000,1000000,1000000,1000000,
    1000000,1000000,1000000,1000000,1000000,
    1000000,1000000,1000000,398130
]
info_filename_rows = {k:v for k,v in zip(filenames_dataset, list_size_per_file)}

source = SimpleEncodedDataset(folder_dataset, info_filename_rows)
# end

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
collator = Collator_BERT_Encoded_254(tokenizer, seq_max)

dataloader_train = DataLoader(source, batch_size*len(GPUS), shuffle=False, collate_fn=collator)
dataloader_eval = DataLoader(source, batch_size*len(GPUS), shuffle=False, collate_fn=collator)

loader = SaverAndLoader(folder_base)
trainer = Builder.load_model_with_2heads(tokenizer.vocab_size, dim_hidden, dim_feedforward, n_head, n_layer, loader, name_epoch_last)

trainer = trainer
# trainer = torch.nn.DataParallel(trainer, device_ids=GPUS)

optimizer = torch.optim.AdamW(trainer.parameters(), lr=1e-4, betas=betas_optimizer, eps=1e-08, weight_decay=0.01)
lr_scheduler = transformers.get_scheduler(
    name="cosine_with_restarts", optimizer=optimizer, num_warmup_steps=100000, num_training_steps=len(dataloader_train) * 5
)

# trainer, optimizer, dataloader_train, dataloader_eval = accelerator.prepare(trainer, optimizer, dataloader_train, dataloader_eval)

loader.load_item_state(name_epoch_last, optimizer)
loader.load_item_state(name_epoch_last, lr_scheduler)

loader.add_item(optimizer)
loader.add_item(lr_scheduler)

if loader.metadata:
    source.restore(**loader.metadata)


trainer, optimizer, dataloader_train, dataloader_eval = accelerator.prepare(trainer, optimizer, dataloader_train, dataloader_eval)

def train_a_batch(batch, trainer, optimizer=None, scheduler=None):
    if batch is None:
        return None, None
    # end
    
    dict_head_output = trainer.forward(**batch())
    
    loss_mlm, info_acc_mlm = trainer.module.get_head('SimpleEncoderHead_MLM').compute_loss(**dict_head_output['SimpleEncoderHead_MLM'])
    loss_s2s, info_acc_s2s = trainer.module.get_head('SimpleDecoderHead_S2S').compute_loss(**dict_head_output['SimpleDecoderHead_S2S'])
    
    # crossentropy loss
    loss_all = loss_mlm + loss_s2s
    loss_all_value = loss_all.item()
    
    
    # loss_all.backward()
    accelerator.backward(loss_all)
    
    if optimizer:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # end
    
    if scheduler:
        scheduler.step()
    # end
    
    return loss_all_value, Dotdict({'mlm': info_acc_mlm, 's2s': info_acc_s2s})
# end


def evaluate_a_batch(batch, trainer, *args, **kwargs):
    if batch is None:
        return None, None
    # end
    
    with torch.no_grad():
        dict_head_output = trainer.forward(**batch())
    # end
    
    loss_mlm, info_acc_mlm = trainer.module.get_head('SimpleEncoderHead_MLM').compute_loss(**dict_head_output['SimpleEncoderHead_MLM'])
    loss_s2s, info_acc_s2s = trainer.module.get_head('SimpleDecoderHead_S2S').compute_loss(**dict_head_output['SimpleDecoderHead_S2S'])
    

    # crossentropy loss
    loss_all = loss_s2s + loss_mlm
    loss_all_value = loss_all.item()
    
    return loss_all_value, Dotdict({'mlm': info_acc_mlm, 's2s': info_acc_s2s})
# end


from datetime import datetime
from tqdm import tqdm
import time

name_checkpoint_current = None
name_checkpoint_last = None


# print('main pid:', os.getpid())
for e in range(epoch_last+1, epoch_last+1+epochs):
    name_checkpoint_current = f'epoch_{e}'
    
    info_acc_heads_train = Dotdict({
        'mlm': SimpleEncoderHead_MLM.get_info_accuracy_template(),
        's2s': SimpleDecoderHead_S2S.get_info_accuracy_template()
    })

    # train phase
    trainer.train()
    source.train()
    
    losss_per_e = []
    for i, batch in enumerate(tqdm(dataloader_train)):
        loss_current, info_acc_heads_batch = train_a_batch(batch, trainer, optimizer, None)
        if loss_current is None:
            continue
        # end
        info_acc_heads_train += info_acc_heads_batch
        losss_per_e.append(loss_current)
        lr_scheduler.step()
        if accelerator.is_main_process and i >= steps_per_save and i % steps_per_save == 0:
            print(f'[INFO] start to save model at epoch: epoch_{epoch_last}, step: {i}, idx: {source.idx_current}')
            loader.update_checkpoint(f'epoch_{epoch_last}', name_checkpoint_last, metadata={'idx_restored': source.idx_current})
            name_checkpoint_last = f'epoch_{epoch_last}'
            print('[{}] Epoch: {} training ends. Status: Average loss: {}, Average MLM accuracy: {}, Average S2S accuracy: {}'.format(
                datetime.utcnow(), e, sum(losss_per_e) / len(losss_per_e),
                info_acc_heads_train.mlm.corrects_masked / info_acc_heads_train.mlm.num_masked,
                info_acc_heads_train.s2s.corrects_segmented / info_acc_heads_train.s2s.num_segmented,
            ))
        # end
    # end

    loss_average_per_e = sum(losss_per_e) / len(losss_per_e)
    # print('[{}] Epoch: {} training ends. Status: Average loss: {}, Average MLM accuracy: {}'.format(
    print('[{}] Epoch: {} training ends. Status: Average loss: {}, Average MLM accuracy: {}, Average S2S accuracy: {}'.format(
        datetime.utcnow(), e, loss_average_per_e,
        info_acc_heads_train.mlm.corrects_masked / info_acc_heads_train.mlm.num_masked,
        info_acc_heads_train.s2s.corrects_segmented / info_acc_heads_train.s2s.num_segmented,
    ))
    
    # lr_scheduler.step() # schedule per 2 epoch
    
    # eval phase
    info_acc_heads_eval = Dotdict({
        'mlm': SimpleEncoderHead_MLM.get_info_accuracy_template(),
        's2s': SimpleDecoderHead_S2S.get_info_accuracy_template()
    })
    
    trainer.eval()
    source.eval()
    losss_per_e = []
    for i, batch in enumerate(tqdm(dataloader_eval)):
        loss_current, info_acc_heads_batch = evaluate_a_batch(batch, trainer)
        if loss_current is None:
            continue
        # end
        info_acc_heads_eval += info_acc_heads_batch
        losss_per_e.append(loss_current)
    # end
    
    loss_average_per_e = sum(losss_per_e) / len(losss_per_e)
    print('[{}] Epoch: {} Evalutation ends. Status: Average loss: {}, Average MLM accuracy: {}, Average S2S accuracy: {}'.format(        
        datetime.utcnow(), e, loss_average_per_e,
        info_acc_heads_eval.mlm.corrects_masked / info_acc_heads_eval.mlm.num_masked,
        info_acc_heads_eval.s2s.corrects_segmented / info_acc_heads_eval.s2s.num_segmented,
    ))
    
    name_checkpoint_current = f'epoch_{e}'
    if accelerator.is_main_process:
        loader.update_checkpoint(name_checkpoint_current, name_checkpoint_last)
    name_checkpoint_last = name_checkpoint_current
# end
