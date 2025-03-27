import torch
import random
import re

from utils import Batch


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
    # end

    def _preprocess(self, line):
        line = re.sub(self.regex_special_token, r'<\1>', line)
        line = re.sub(r'''('|"|`){2}''', '', line)
        line = re.sub(r'\.{2,3}', '', line)
        line = re.sub(r' {2,}', ' ', line)
        line = line.lstrip().rstrip()
        return line
    # end

    # return masks_attention?, return masks_segment?
    def pad_sequences(self, sequences, size_seq_max, need_diagonal=False,
                      need_masked=0):  # need_diagonal and need_masked cannot both set, one for bert seq one for s2s seq
        id_pad = self.id_pad
        id_mask = self.id_mask

        sequences_padded = []
        sequences_masked_padded = []

        for sequence in sequences:
            len_seq = len(sequence)

            count_pad = size_seq_max - len_seq

            sequence = torch.LongTensor(sequence)
            sequence_padded = torch.cat((sequence, torch.LongTensor([id_pad] * count_pad)))
            sequences_padded.append(sequence_padded)

            if need_masked:
                index_masked = list(range(1, len_seq - 1))
                random.shuffle(index_masked)
                anchor_mask = int(need_masked * (len_seq - 2)) or 1
                index_masked = torch.LongTensor(index_masked[:anchor_mask])
                # index_masked = torch.LongTensor(index_masked[:int(need_masked * (len_seq-2))])

                sequence_masked = sequence.detach().clone()
                sequence_masked.index_fill_(0, index_masked, id_mask)
                sequence_masked_padded = torch.cat((sequence_masked, torch.LongTensor([id_pad] * count_pad)))

                sequences_masked_padded.append(sequence_masked_padded)
            # end
        #   # end for

        inputs = torch.stack(sequences_padded)  # (batch, size_seq_max)
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


class Collator_SC(Collator_Base):



    def __call__(self, list_corpus_source):

        tokens_input_encoder = []
        tokens_input_decoder = []
        tokens_label_decoder = []
        labels_similarity = []
        labels_sc = []

        for corpus_source in list_corpus_source:  # (line0, line1, sim), output of zip remove single case
            if len(corpus_source) == 3:  # (line0, line1, sim)
                corpus_line = [corpus_source[0], corpus_source[1]]
                labels_similarity.append(corpus_source[2])
            elif len(corpus_source) == 2:  # (line, label_sc)
                corpus_line = [corpus_source[0]]
                labels_sc.append(corpus_source[1])
            else:
                corpus_line = [corpus_source[0]]
            # end

            for line in corpus_line:
                tokens = self.tokenizer.encode(self._preprocess(line), add_special_tokens=False)

                # TODO: check edge
                if len(tokens) > self.size_seq_max - 2:
                    tokens = tokens[:self.size_seq_max - 2]
                # end

                tokens_input_encoder.append([self.id_cls] + tokens + [self.id_sep])
                tokens_input_decoder.append([self.id_cls] + tokens)
                tokens_label_decoder.append(tokens + [self.id_sep])
            # end

        # end

        inputs_encoder, masks_encoder, segments_encoder, labels_encoder = self.pad_sequences(tokens_input_encoder,
                                                                                             self.size_seq_max,
                                                                                             need_masked=self.need_masked)
        inputs_decoder, masks_decoder, segments_decoder, _ = self.pad_sequences(tokens_input_decoder, self.size_seq_max,
                                                                                need_diagonal=True)
        labels_decoder, masks_label, segments_label, _ = self.pad_sequences(tokens_label_decoder, self.size_seq_max)
        # labels_similarity = torch.Tensor(labels_similarity).unsqueeze(0).transpose(0,1)
        labels_similarity = torch.Tensor(labels_similarity)
        labels_sc = torch.LongTensor(labels_sc)

        return Batch(
            ids_encoder=inputs_encoder,  # contains [mask]s
            masks_encoder=masks_encoder,
            labels_encoder=labels_encoder,  # doesn't contain [mask]
            segments_encoder=segments_encoder,
            ids_decoder=inputs_decoder,
            masks_decoder=masks_decoder,
            labels_decoder=labels_decoder,
            segments_label=segments_label,
            labels_similarity=labels_similarity,
            labels_sc=labels_sc
        )

    # end
# end


class Collator_BERT:

    def __call__(self, list_sequence_batch):
        list_sequence_batch = [self._preprocess(sequence) for sequence in list_sequence_batch]  # remove special tokens

        list_sequence_tokenized = self.tokenizer.batch_encode_plus(list_sequence_batch, add_special_tokens=False)[
            'input_ids']

        # Process I.
        list_list_tokenized = []

        # batch initialized condition
        list_tokenized_cache = []
        len_tokenized_accumulated = 2  # add cls and sep

        while list_sequence_tokenized:
            tokenized_poped = list_sequence_tokenized.pop(0)
            len_tokenized_current = len(tokenized_poped)

            if len_tokenized_accumulated + len_tokenized_current > self.size_seq_max:
                if list_tokenized_cache:
                    list_list_tokenized.append(list_tokenized_cache)

                    # clear
                    list_tokenized_cache = []
                    len_tokenized_accumulated = 2
                # end
            # end

            list_tokenized_cache.append(tokenized_poped)
            len_tokenized_accumulated += len_tokenized_current
        # end

        list_list_tokenized.append(list_tokenized_cache)

        # Process II. Merge list_tokenized
        list_tokenized_merged = []

        for list_tokenized in list_list_tokenized:
            # tokenized_merged = [token for tokenized_padded in [tokenized + [self.id_eol] for tokenized in list_tokenized] for token in tokenized_padded]
            tokenized_merged = [token for tokenized in list_tokenized for token in tokenized][:self.size_seq_max - 2]
            list_tokenized_merged.append(tokenized_merged)
        # end

        # Process III. Add begin and stop special token, same as jinyuj_transformers_quora.ipynb
        tokens_input_encoder = []
        tokens_input_decoder = []
        tokens_label_decoder = []

        for tokenized_merged in list_tokenized_merged:
            tokens_input_encoder.append([self.id_cls] + tokenized_merged + [self.id_sep])
            tokens_input_decoder.append([self.id_cls] + tokenized_merged)
            tokens_label_decoder.append(tokenized_merged + [self.id_sep])
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


class Collator_BERT_MIXED:

    def __call__(self, list_sequence_batch):
        list_sequence_batch = [self._preprocess(sequence) for sequence in list_sequence_batch]  # remove special tokens

        list_sequence_tokenized = self.tokenizer.batch_encode_plus(list_sequence_batch, add_special_tokens=False)[
            'input_ids']

        # Process I.
        list_list_tokenized = []

        # batch initialized condition
        list_tokenized_cache = []
        len_tokenized_accumulated = 2  # add cls and sep

        while list_sequence_tokenized:
            tokenized_poped = list_sequence_tokenized.pop(0)
            len_tokenized_current = len(tokenized_poped)

            if len_tokenized_accumulated + len_tokenized_current > self.size_seq_max:
                if list_tokenized_cache:
                    list_list_tokenized.append(list_tokenized_cache)

                    # clear
                    list_tokenized_cache = []
                    len_tokenized_accumulated = 2
                # end
            # end

            list_tokenized_cache.append(tokenized_poped)
            len_tokenized_accumulated += len_tokenized_current
        # end

        list_list_tokenized.append(list_tokenized_cache)

        # Process II. Merge list_tokenized
        list_tokenized_merged = []

        for list_tokenized in list_list_tokenized:
            # tokenized_merged = [token for tokenized_padded in [tokenized + [self.id_eol] for tokenized in list_tokenized] for token in tokenized_padded]
            tokenized_merged = [token for tokenized in list_tokenized for token in tokenized][:self.size_seq_max - 2]
            list_tokenized_merged.append(tokenized_merged)
        # end


        # Process III. Mix origin sentence
        len_mixed = len(list_tokenized_merged)
        indexs_origin = list(range(len(list_sequence_tokenized)))
        random.shuffle(indexs_origin)
        indexs_mixed = indexs_origin[:len_mixed]
        list_tokenized_merged += [list_sequence_tokenized[index_mixed] for index_mixed in indexs_mixed]


        # Process IV. Add begin and stop special token, same as jinyuj_transformers_quora.ipynb
        tokens_input_encoder = []
        tokens_input_decoder = []
        tokens_label_decoder = []

        for tokenized_merged in list_tokenized_merged:
            tokens_input_encoder.append([self.id_cls] + tokenized_merged + [self.id_sep])
            tokens_input_decoder.append([self.id_cls] + tokenized_merged)
            tokens_label_decoder.append(tokenized_merged + [self.id_sep])
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


class Collator_BERT_MIXED(Collator_Base):

    def __call__(self, list_sequence_batch):
        list_sequence_batch = [self._preprocess(sequence) for sequence in list_sequence_batch]  # remove special tokens

        list_sequence_tokenized = self.tokenizer.batch_encode_plus(list_sequence_batch, add_special_tokens=False)[
            'input_ids']

        # Process I.
        list_list_tokenized = []

        # batch initialized condition
        list_tokenized_cache = []
        len_tokenized_accumulated = 2  # add cls and sep

        # while list_sequence_tokenized:
        for tokenized_top in list_sequence_tokenized:
            len_tokenized_current = len(tokenized_top)

            if len_tokenized_accumulated + len_tokenized_current > self.size_seq_max:
                if list_tokenized_cache:
                    list_list_tokenized.append(list_tokenized_cache)

                    # clear
                    list_tokenized_cache = []
                    len_tokenized_accumulated = 2
                # end
            # end

            list_tokenized_cache.append(tokenized_top)
            len_tokenized_accumulated += len_tokenized_current
        # end

        list_list_tokenized.append(list_tokenized_cache)

        # Process II. Merge list_tokenized
        list_tokenized_merged = []

        for list_tokenized in list_list_tokenized:
            # tokenized_merged = [token for tokenized_padded in [tokenized + [self.id_eol] for tokenized in list_tokenized] for token in tokenized_padded]
            tokenized_merged = [token for tokenized in list_tokenized for token in tokenized][:self.size_seq_max - 2]
            list_tokenized_merged.append(tokenized_merged)
        # end

        # Process III. Mix origin sentence if merged
        len_mixed = len(list_tokenized_merged)
        if len_mixed >= 1 and len(list_tokenized_merged[0]) > len(list_sequence_tokenized[0]):
            indexs_origin = list(range(len(list_sequence_tokenized)))
            random.shuffle(indexs_origin)
            indexs_mixed = indexs_origin[:len_mixed]
            list_tokenized_merged += [list_sequence_tokenized[index_mixed] for index_mixed in indexs_mixed]
        # end

        # Process IV. Add begin and stop special token, same as jinyuj_transformers_quora.ipynb
        tokens_input_encoder = []
        tokens_input_decoder = []
        tokens_label_decoder = []

        for tokenized_merged in list_tokenized_merged:
            tokens_input_encoder.append([self.id_cls] + tokenized_merged + [self.id_sep])
            tokens_input_decoder.append([self.id_cls] + tokenized_merged)
            tokens_label_decoder.append(tokenized_merged + [self.id_sep])
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