import torch
from torch import nn

from utils import Dotdict
from core import LinearAndNorm


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

        self.func_loss = torch.nn.CrossEntropyLoss()

        self.keys_cache = ['output', 'labels_s2s', 'segments_decoder']
        self.cache = Dotdict({
            'output': None,
            'labels_s2s': None,
            'segments_decoder': None
        })

    # end

    def forward(self, model, labels_decoder=None, segments_label=None, nocache=False,
                **kwargs):  # labels_input -> (batch, seq, labels)
        output_decoder = model.decoder.cache.output
        output_ffn = self.ffn(output_decoder)
        output_s2s = self.extractor(output_ffn)  # output_mlm = prediction_logits

        if not nocache:
            self.cache.segments_label = segments_label
            self.cache.labels_s2s = labels_decoder
            self.cache.output = output_s2s
        # end

        return output_s2s

    # end

    def get_loss(self):
        labels_s2s = self.cache.labels_s2s
        output_s2s = self.cache.output
        info_acc = SimpleDecoderHead_S2S.get_info_accuracy_template()

        segments_label = self.cache.segments_label
        segments_label_2d = segments_label.transpose(-1, -2)[:, :, 0]
        hidden_s2s_segmented = output_s2s.masked_select(segments_label_2d.unsqueeze(-1)).reshape(-1,
                                                                                                 output_s2s.shape[-1])

        loss_segments = self.func_loss(hidden_s2s_segmented, labels_s2s.masked_select(segments_label_2d))
        info_acc.corrects_segmented = torch.sum(
            hidden_s2s_segmented.argmax(-1) == labels_s2s.masked_select(segments_label_2d)).cpu().item()
        info_acc.num_segmented = hidden_s2s_segmented.shape[0]

        return loss_segments * 4, info_acc

    # end

    def evaluate(self):
        pass

    # end

    def clear_cache(self):
        for key_cache in self.keys_cache:
            self.cache[key_cache] = None
        # end
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

        self.keys_cache = ['labels_mlm', 'masks_encoder', 'segments_encoder', 'output']
        self.cache = Dotdict({
            'labels_mlm': None,
            'masks_encoder': None,
            'segments_encoder': None,
            'output': None
        })

        self.func_loss = torch.nn.CrossEntropyLoss()

    # end

    def forward(self, model, labels_encoder=None, segments_encoder=None, masks_encoder=None, nocache=False,
                **kwargs):  # labels_input -> (batch, seq, labels)
        output_encoder = model.encoder.cache.output
        output_ffn = self.ffn(output_encoder)
        output_mlm = self.extractor(output_ffn)  # output_mlm = prediction_logits

        if not nocache:
            self.cache.labels_mlm = labels_encoder
            self.cache.masks_encoder = masks_encoder
            self.cache.segments_encoder = segments_encoder
            self.cache.output = output_mlm
        # end

        return output_mlm

    # end

    def get_loss(self):

        labels_mlm = self.cache.labels_mlm
        masks_encoder = self.cache.masks_encoder
        segments_encoder = self.cache.segments_encoder
        output_mlm = self.cache.output

        info_acc = SimpleEncoderHead_MLM.get_info_accuracy_template()

        segments_encoder_2d = segments_encoder.transpose(-1, -2)[:, :, 0]
        hidden_mlm_segmented = output_mlm.masked_select(segments_encoder_2d.unsqueeze(-1)).reshape(-1, output_mlm.shape[
            -1])  # should be (segmented_all_batchs, size_vocab)

        loss_segments = self.func_loss(hidden_mlm_segmented, labels_mlm.masked_select(segments_encoder_2d))
        info_acc.corrects_segmented = torch.sum(
            hidden_mlm_segmented.argmax(-1) == labels_mlm.masked_select(segments_encoder_2d)).cpu().item()
        info_acc.num_segmented = hidden_mlm_segmented.shape[0]

        masks_masked = torch.logical_xor(masks_encoder, segments_encoder) & segments_encoder  # True is masked
        masks_masked_perbatch = masks_masked[:, 0, :]
        hidden_mlm_masked = output_mlm.masked_select(masks_masked_perbatch.unsqueeze(-1)).reshape(-1,
                                                                                                  output_mlm.shape[-1])

        if hidden_mlm_masked.shape[0] != 0:
            loss_masked = self.func_loss(hidden_mlm_masked, labels_mlm.masked_select(masks_masked_perbatch))
            info_acc.corrects_masked = torch.sum(
                hidden_mlm_masked.argmax(-1) == labels_mlm.masked_select(masks_masked_perbatch)).cpu().item()
            info_acc.num_masked = hidden_mlm_masked.shape[0]
        else:
            loss_masked = 0
            info_acc.corrects_masked = 0
            info_acc.num_masked = 1
        # end

        loss_mlm = loss_segments + loss_masked * 3

        return loss_mlm, info_acc

    # end

    def clear_cache(self):
        for key_cache in self.keys_cache:
            self.cache[key_cache] = None
        # end
    # end


# end


class SimpleEncoderHead_AveragePooling_SC(nn.Module):  # SC-> SequenceClassification

    @classmethod
    def get_info_accuracy_template(cls):
        return Dotdict({
            'corrects': 0,
            'num': 0
        })

    # end

    def __init__(self, num_labels, dim_hidden=128, dropout=0.1):
        super(SimpleEncoderHead_AveragePooling_SC, self).__init__()

        self.ffn = LinearAndNorm(dim_in=dim_hidden, dim_out=dim_hidden, dropout=dropout)
        self.classifier = torch.nn.Linear(dim_hidden, num_labels, bias=False)

        self.keys_cache = ['labels_sc', 'output']
        self.cache = Dotdict({
            'labels_sc': None,
            'output': None
        })

        self.func_loss = torch.nn.CrossEntropyLoss()

    # end

    def forward(self, model, labels_sc=None, nocache=False, **kwargs):  # labels_input -> (batch, seq, labels)
        output_encoder_pooled = model.cache.output_encoder_pooled
        output_ffn = self.ffn(output_encoder_pooled)
        output_sc = self.classifier(output_ffn)  # output_sc = prediction_logits

        if not nocache:
            self.cache.labels_sc = labels_sc
            self.cache.output = output_sc
        # end

        return output_sc

    # end

    def get_loss(self):

        labels_sc = self.cache.labels_sc
        output_sc = self.cache.output

        info_acc = SimpleEncoderHead_AveragePooling_SC.get_info_accuracy_template()

        loss_sc = self.func_loss(output_sc, labels_sc)
        info_acc.corrects = torch.sum(output_sc.argmax(-1) == labels_sc).cpu().item()
        info_acc.num = output_sc.shape[0]

        return loss_sc, info_acc

    # end

    def clear_cache(self):
        for key_cache in self.keys_cache:
            self.cache[key_cache] = None
        # end
    # end
# end


class SimpleEncoderHead_Similarity(nn.Module):

    @classmethod
    def get_info_accuracy_template(cls):
        return Dotdict({
            'meansquares': []
        })

    # end

    def __init__(self):
        super(SimpleEncoderHead_Similarity, self).__init__()

        self.func_loss = torch.nn.MSELoss()
        self.cos_score_transformation = torch.nn.Identity()
        self.keys_cache = ['labels_sim', 'output']
        self.cache = Dotdict({
            'labels_sim': None,
            'output': None
        })

    # end

    def forward(self, model, labels_similarity=None, nocache=False,
                **kwargs):  # labels_sim (batch/2, 1)   for every two sentences, we have a label

        output_encoder_pooled = model.cache.output_encoder_pooled
        size_batch, dim_hidden = output_encoder_pooled.shape

        if size_batch % 2 != 0:
            raise Exception('sim calculation is not prepared as size_batch % 2 != 0')
        # end

        # pooling (batch, pair, dim_hidden)
        output_pooling = output_encoder_pooled.squeeze(1).view(-1, 2,
                                                               dim_hidden)  # might cls + sep, but abandon now (as it's not easy to get sep for every batch, different location)
        output_pooling_x1 = output_pooling[:, 0, :]
        output_pooling_x2 = output_pooling[:, 1, :]
        sims = self.cos_score_transformation(
            torch.cosine_similarity(output_pooling_x1, output_pooling_x2))  # -> (batch, scores)

        if not nocache:
            self.cache.output = sims
            self.cache.labels_sim = labels_similarity
        # end

        return sims

    # end

    def get_loss(self):
        sims = self.cache.output
        labels_sim = self.cache.labels_sim
        info_acc = SimpleEncoderHead_Similarity.get_info_accuracy_template()

        loss_sim = self.func_loss(sims, labels_sim)
        info_acc.meansquares.append((torch.mean((sims - labels_sim) ** 2)).cpu().item())
        return loss_sim * 64, info_acc

    # end

    def clear_cache(self):
        for key_cache in self.keys_cache:
            self.cache[key_cache] = None
        # end

    # end

    def evaluate(self):
        pass
    # end
# end