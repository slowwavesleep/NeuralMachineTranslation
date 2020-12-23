import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
            layer_dropout = 0

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

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.bidirectional = bidirectional

    def forward(self, encoder_seq):

        initial_len = encoder_seq.size(-1)

        encoder_lens = get_non_pad_lens(encoder_seq)

        encoder_seq = self.embedding(encoder_seq)

        encoder_seq = self.spatial_dropout(encoder_seq)

        encoder_seq = self.layer_norm(encoder_seq)

        encoder_seq = pack_padded_sequence(input=encoder_seq,
                                           lengths=encoder_lens,
                                           batch_first=True,
                                           enforce_sorted=False)

        encoder_seq, memory = self.lstm(encoder_seq)

        encoder_seq = pad_packed_sequence(sequence=encoder_seq,
                                          batch_first=True,
                                          total_length=initial_len)[0]

        if self.bidirectional:
            hidden, cell = memory
            hidden = hidden.permute(1, 0, 2)
            hidden = hidden.reshape(hidden.size(0), 1, -1).permute(1, 0, 2)
            cell = cell.permute(1, 0, 2)
            cell = cell.reshape(cell.size(0), 1, -1).permute(1, 0, 2)
            memory = hidden, cell

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

        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, decoder_seq, memory):

        initial_len = decoder_seq.size(-1)

        decoder_lens = get_non_pad_lens(decoder_seq)

        decoder_seq = self.embedding(decoder_seq)

        decoder_seq = self.spatial_dropout(decoder_seq)

        decoder_seq = self.layer_norm(decoder_seq)

        decoder_seq = pack_padded_sequence(input=decoder_seq,
                                           lengths=decoder_lens,
                                           batch_first=True,
                                           enforce_sorted=False)

        decoder_seq, _ = self.lstm(decoder_seq, memory)

        decoder_seq = pad_packed_sequence(sequence=decoder_seq,
                                          batch_first=True,
                                          total_length=initial_len)[0]

        if self.head:
            decoder_seq = self.fc(decoder_seq)

        return decoder_seq


def get_pad_mask(seq_1, seq_2):
    # (batch_size, seq_len_1), (batch_size, seq_len_2)  -> (batch_size, seq_len_2, seq_len_1)
    seq_len_1 = seq_1.size(-1)
    seq_len_2 = seq_2.size(-1)
    lens = get_non_pad_lens(seq_1)
    masks = [torch.arange(seq_len_1).expand(seq_len_2, seq_len_1) >= true_len for true_len in lens]
    return torch.stack(masks).cuda()


def get_non_pad_lens(seq):
    lens = seq.size(-1) - (seq == 0).sum(-1)
    return lens
