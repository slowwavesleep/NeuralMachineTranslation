from torch import nn
import torch


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


class BasicEncoderDecoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 model_dim: int,
                 model_layers: int,
                 model_dropout: float,
                 padding_index: int):

        super().__init__()

        if model_layers < 2 and model_dropout != 0:
            model_dropout = 0

        self.source_embedding = nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=emb_dim,
                                             padding_idx=padding_index)

        self.target_embedding = nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=emb_dim,
                                             padding_idx=padding_index)

        self.source_lstm = nn.LSTM(input_size=emb_dim,
                                   hidden_size=model_dim,
                                   num_layers=model_layers,
                                   dropout=model_dropout,
                                   batch_first=True)

        self.target_lstm = nn.LSTM(input_size=emb_dim,
                                   hidden_size=model_dim,
                                   num_layers=model_layers,
                                   dropout=model_dropout,
                                   batch_first=True)

        self.output = nn.Linear(in_features=model_dim,
                                out_features=vocab_size)

    def forward(self, encoder_seq, decoder_seq):

        encoder_seq = self.source_embedding(encoder_seq)
        _, (hidden, cell) = self.source_lstm(encoder_seq)

        decoder_seq = self.target_embedding(decoder_seq)
        decoder_seq, _ = self.target_lstm(decoder_seq, (hidden, cell))

        decoder_seq = self.output(decoder_seq)

        return decoder_seq
