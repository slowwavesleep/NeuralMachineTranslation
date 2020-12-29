from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Union

from src.nn.layers import LstmEncoder, LstmEncoderPacked, LstmDecoder, LstmDecoderPacked, get_pad_mask, SpatialDropout


def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 mask: Union[None, Tensor] = None) -> Tensor:

    similarity = query.bmm(key.transpose(1, 2))

    # scale similarity matrix by square root of number of dimensions
    scale = query.size(-1) ** 0.5

    if mask is not None:
        similarity = similarity.masked_fill(mask, float('-inf'))

    softmax = F.softmax(similarity / scale, dim=-1)

    return softmax.bmm(value)


class BaselineModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):

        super(BaselineModel, self).__init__()

        self.encoder = LstmEncoder(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   hidden_size=hidden_size,
                                   lstm_layers=lstm_layers,
                                   layer_dropout=layer_dropout,
                                   spatial_dropout=spatial_dropout,
                                   bidirectional=bidirectional,
                                   padding_index=padding_index)

        self.decoder = LstmDecoder(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   hidden_size=hidden_size,
                                   lstm_layers=lstm_layers,
                                   spatial_dropout=spatial_dropout,
                                   padding_index=padding_index)

    def forward(self, encoder_seq, decoder_seq):
        encoder_seq, memory = self.encoder(encoder_seq)
        output = self.decoder(decoder_seq, memory)

        return output


class LstmAttentionModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int,
                 layer_dropout: float,
                 spatial_dropout: float,
                 bidirectional: bool,
                 padding_index: int):
        super(LstmAttentionModel, self).__init__()

        self.directions = 2 if bidirectional else 1

        self.encoder = LstmEncoderPacked(vocab_size=vocab_size,
                                         emb_dim=emb_dim,
                                         hidden_size=hidden_size,
                                         lstm_layers=lstm_layers,
                                         layer_dropout=layer_dropout,
                                         spatial_dropout=spatial_dropout,
                                         bidirectional=bidirectional,
                                         padding_index=padding_index)

        self.decoder = LstmDecoderPacked(vocab_size=vocab_size,
                                         emb_dim=emb_dim,
                                         hidden_size=hidden_size * self.directions,
                                         lstm_layers=lstm_layers,
                                         spatial_dropout=spatial_dropout,
                                         padding_index=padding_index,
                                         head=False)

        self.key_projection = nn.Linear(hidden_size * self.directions, hidden_size * self.directions)
        self.value_projection = nn.Linear(hidden_size * self.directions, hidden_size * self.directions)
        self.query_projection = nn.Linear(hidden_size * self.directions, hidden_size * self.directions)

        self.fc = nn.Linear(hidden_size * self.directions,
                            vocab_size)

        self.layer_norm = nn.LayerNorm(hidden_size * self.directions)
        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

    def forward(self, encoder_seq, decoder_seq):

        mask = get_pad_mask(encoder_seq, decoder_seq)

        encoder_seq, memory = self.encoder(encoder_seq)

        decoder_seq = self.decoder(decoder_seq, memory)

        query = self.query_projection(decoder_seq)
        key = self.key_projection(encoder_seq)
        value = self.value_projection(encoder_seq)

        attention = scaled_dot_product_attention(query, key, value, mask)

        output = torch.tanh(decoder_seq + attention)

        output = self.layer_norm(output)

        output = self.spatial_dropout(output)

        return self.fc(output)

