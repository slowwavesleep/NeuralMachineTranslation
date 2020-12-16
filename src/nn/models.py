from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F


def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_k: int,
                 dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class SpatialDropout(torch.nn.Dropout2d):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class LstmEncoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):
        super(LstmEncoder, self).__init__()

        if lstm_layers < 2 and layer_dropout != 0:
            model_dropout = 0

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=layer_dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, encoder_seq):
        """
        :param encoder_seq: tokenized source sentences
        :return: output and memory of LSTM
        """

        encoder_seq = self.embedding(encoder_seq)

        encoder_seq = self.spatial_dropout(encoder_seq)

        output, memory = self.lstm(encoder_seq)

        return output, memory


class LstmDecoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0,
                 head: bool = True):

        super(LstmDecoder, self).__init__()

        self.head = head

        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True)

        self.spatial_dropout = SpatialDropout(p=spatial_dropout)

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=padding_index)

        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=vocab_size)

    def forward(self, decoder_seq, memory):
        decoder_seq = self.embedding(decoder_seq)

        decoder_seq = self.spatial_dropout(decoder_seq)

        output, _ = self.lstm(decoder_seq, memory)

        if self.head:
            output = self.fc(output)

        return output


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

        self.encoder = LstmEncoder(vocab_size,
                                   emb_dim,
                                   hidden_size,
                                   lstm_layers,
                                   layer_dropout,
                                   spatial_dropout,
                                   bidirectional,
                                   padding_index)

        self.decoder = LstmDecoder(vocab_size,
                                   emb_dim,
                                   hidden_size,
                                   lstm_layers,
                                   spatial_dropout,
                                   padding_index)

    def forward(self, encoder_seq, decoder_seq):
        encoder_seq, memory = self.encoder(encoder_seq)
        output = self.decoder(decoder_seq, memory)

        return output


class LstmAttention(nn.Module):

    def __init__(self,
                 vocab_size,
                 emb_dim,
                 hidden_size,
                 lstm_layers,
                 layer_dropout,
                 spatial_dropout,
                 bidirectional,
                 padding_index):

        super(LstmAttention, self).__init__()

        self.encoder = LstmEncoder(vocab_size=vocab_size,
                                   emb_dim=emb_dim,
                                   hidden_size=hidden_size,
                                   lstm_layers=lstm_layers,
                                   layer_dropout=layer_dropout,
                                   spatial_dropout=spatial_dropout,
                                   bidirectional=bidirectional,
                                   padding_index=padding_index)

        self.decoder = LstmDecoder(vocab_size,
                                   emb_dim,
                                   hidden_size,
                                   lstm_layers,
                                   spatial_dropout,
                                   padding_index,
                                   head=False)

        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.query_projection = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size,
                            vocab_size)

    def forward(self, encoder_seq, decoder_seq):

        encoder_seq, memory = self.encoder(encoder_seq)

        decoder_seq = self.decoder(decoder_seq, memory)

        query = self.query_projection(decoder_seq)
        key = self.key_projection(encoder_seq)
        value = self.value_projection(encoder_seq)

        attention = scaled_dot_product_attention(query, key, value)

        output = decoder_seq + attention

        return self.fc(output)



