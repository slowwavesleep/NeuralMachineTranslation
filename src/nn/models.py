from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


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
        encoder_seq = self.embedding(encoder_seq)

        encoder_seq = self.spatial_dropout(encoder_seq)

        output, memory = self.lstm(encoder_seq)

        return output, memory


class LstmEncoderPacked(LstmEncoder):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 layer_dropout: float = 0.,
                 spatial_dropout: float = 0.,
                 bidirectional: bool = False,
                 padding_index: int = 0):
        super().__init__(vocab_size,
                         emb_dim,
                         hidden_size,
                         lstm_layers,
                         layer_dropout,
                         spatial_dropout,
                         bidirectional,
                         padding_index)

    def forward(self, encoder_seq):
        encoder_lens = encoder_seq.size(-1) - (encoder_seq == 0).sum(-1)

        encoder_seq = self.embedding(encoder_seq)

        encoder_seq = self.spatial_dropout(encoder_seq)

        encoder_seq = pack_padded_sequence(input=encoder_seq,
                                           lengths=encoder_lens,
                                           batch_first=True,
                                           enforce_sorted=False)

        encoder_seq, memory = self.lstm(encoder_seq)

        encoder_seq = pad_packed_sequence(sequence=encoder_seq,
                                          batch_first=True)[0]

        return encoder_seq, memory


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


class LstmDecoderPacked(LstmDecoder):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_size: int,
                 lstm_layers: int = 1,
                 spatial_dropout: float = 0.,
                 padding_index: int = 0,
                 head: bool = True):
        super().__init__(vocab_size,
                         emb_dim,
                         hidden_size,
                         lstm_layers,
                         spatial_dropout,
                         padding_index,
                         head)

    def forward(self, decoder_seq, memory):
        decoder_lens = decoder_seq.size(-1) - (decoder_seq == 0).sum(-1)

        decoder_seq = self.embedding(decoder_seq)

        decoder_seq = self.spatial_dropout(decoder_seq)

        decoder_seq = pack_padded_sequence(input=decoder_seq,
                                           lengths=decoder_lens,
                                           batch_first=True,
                                           enforce_sorted=False)

        decoder_seq, _ = self.lstm(decoder_seq, memory)

        encoder_seq = pad_packed_sequence(sequence=decoder_seq,
                                          batch_first=True)[0]

        if self.head:
            encoder_seq = self.fc(encoder_seq)

        return encoder_seq


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
                                         hidden_size=hidden_size,
                                         lstm_layers=lstm_layers,
                                         spatial_dropout=spatial_dropout,
                                         padding_index=padding_index,
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

        # TODO Add mask
        attention = scaled_dot_product_attention(query, key, value)

        output = decoder_seq + attention

        # TODO Add layer norm

        return self.fc(output)


class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):

        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden=None):

        self.check_forward_input(x)

        if hidden is None:
            hx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
            cx = x.new_zeros(x.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        self.check_forward_hidden(x, hx, '[0]')
        self.check_forward_hidden(x, cx, '[1]')

        gates = self.ln_ih(F.linear(x, self.weight_ih, self.bias_ih)) \
                + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))

        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)

        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()

        return hy, cy


class LayerNormLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 bidirectional=False):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1

        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, bias=bias)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, bias=bias)
                for layer in range(num_layers)
            ])

    def forward(self,
                x,
                hidden=None):

        seq_len, batch_size, hidden_size = x.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1

        if hidden is None:
            hx = x.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = x.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirectional:
            xs = x
            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(x):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])

        return y, (hy, cy)
