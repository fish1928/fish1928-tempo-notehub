import torch
from torch import nn

from utils import Dotdict
from core import MultiHeadedAttention, PositionwiseFeedForward, clones, ResidualLayer

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
        embeddings = self.layers_residual[0](embeddings,
                                             lambda embeddings: self.layer_attention(embeddings, embeddings, embeddings,
                                                                                     masks))
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
        embeddings = self.layers_residual[0](embeddings,
                                             lambda embeddings: self.layer_attention_decoder(embeddings, embeddings,
                                                                                             embeddings, masks_decoder))
        embeddings = self.layers_residual[1](embeddings,
                                             lambda embeddings: self.layer_attention_encoder(embeddings, output_encoder,
                                                                                             output_encoder,
                                                                                             masks_encoder))
        return self.layers_residual[2](embeddings, self.layer_feedforward)
    # end


# end


class SimpleTransformerStack(nn.Module):

    def __init__(self, obj_layer, n_layers):
        super(SimpleTransformerStack, self).__init__()
        self.layers = clones(obj_layer, n_layers)

        self.norm = torch.nn.LayerNorm(obj_layer.dim_hidden)
        self.keys_cache = ['output']
        self.cache = Dotdict({
            'output': None
        })

    # end

    def forward(self, embedding_encoder=None, masks_encoder=None, output_encoder=None, embedding_decoder=None,
                masks_decoder=None, noncache=False, **kwargs):  # input -> (batch, len_seq, vocab)

        if output_encoder is not None and embedding_decoder is not None and masks_decoder is not None:
            embeddings = embedding_decoder
        else:
            embeddings = embedding_encoder
        # end

        for layer in self.layers:
            embeddings = layer(embeddings, masks_encoder, output_encoder, masks_decoder)
        # end

        output = self.norm(embeddings)

        if not noncache:
            self.cache.output = output
        # end

        return output

    # end

    # def get_vocab_size(self):
    #     return self.embedder.embedder_token.shape[-1]
    # # end

    def clear_cache(self):
        for key_cache in self.keys_cache:
            self.cache[key_cache] = None
        # end
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

        self.keys_cache = ['output_encoder_pooled']
        self.cache = Dotdict({
            'output_encoder_pooled': None
        })

    # end

    def forward(self, ids_encoder=None, masks_encoder=None, ids_decoder=None, masks_decoder=None, nocache=False,
                **kwargs):

        output_encoder = self.embed_and_encode(ids_encoder=ids_encoder, masks_encoder=masks_encoder, nocache=nocache)
        output = output_encoder

        if self.pooling:
            output_encoder_refilled = output_encoder.masked_fill(masks_encoder.transpose(-1, -2) == False, 0)
            output_encoder_pooled = torch.mean(output_encoder_refilled, dim=-2)
            self.cache.output_encoder_pooled = output_encoder_pooled

            output_encoder_pooled_expanded = output_encoder_pooled.unsqueeze(-2).expand(output_encoder.shape)
            output = output_encoder_pooled_expanded
        # end

        if self.embedder_decoder and self.decoder:
            output_decoder = self.embed_and_decode(ids_decoder=ids_decoder, masks_encoder=masks_encoder,
                                                   output_encoder=output, masks_decoder=masks_decoder, nocache=nocache)
            output = output_decoder
        # end if

        return output

    # end

    def embed_and_encode(self, ids_encoder=None, masks_encoder=None, nocache=False, **kwargs):
        self.encoder.clear_cache()

        embedding_encoder = self.embedder_encoder(ids_encoder)
        output_encoder = self.encoder(
            embedding_encoder=embedding_encoder,
            masks_encoder=masks_encoder,
            nocache=nocache
        )

        return output_encoder

    # end

    def embed_and_decode(self, ids_decoder=None, masks_encoder=None, output_encoder=None, masks_decoder=None,
                         nocache=False, **kwargs):
        self.decoder.clear_cache()

        embedding_decoder = self.embedder_decoder(ids_decoder)
        output_decoder = self.decoder(
            masks_encoder=masks_encoder,
            output_encoder=output_encoder,  # (len_seq, dim_hidden) -> (1, dim_hidden)
            embedding_decoder=embedding_decoder,
            masks_decoder=masks_decoder,
            nocache=nocache
        )

        return output_decoder

    # end

    def clear_cache(self):
        self.encoder.clear_cache()

        for key_cache in self.keys_cache:
            self.cache[key_cache] = None
        # end

        if self.decoder:
            self.decoder.clear_cache()
        # end

    # end

    def get_vocab_size(self, name_embedder):
        embedder = getattr(self, f'embedder_{name_embedder}')
        return embedder.get_vocab_size()
    # end

# end