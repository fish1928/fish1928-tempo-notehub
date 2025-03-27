import copy
import math
import torch
from torch import nn

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


### core.py ###