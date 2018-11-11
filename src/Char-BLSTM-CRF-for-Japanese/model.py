import torch
import torch.nn as nn
from TorchCRF.crf import CRF


class blstm_crf(nn.Module):

    def __init__(self, vocab_size: int, num_labels: int,
                 hidden_size:int,
                 embed_size: int, pad_idx: int):
        super().__init__()


class blstm(nn.Module):

    def __init__(self, vocab_size: int, num_labels: int,
                 hidden_size:int, embed_size: int,
                 dropout_rate: int=0, pad_idx: int=0):

        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size= hidden_size // 2,
            num_layers=1,
            bias=True,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )
        self.out = nn.Linear(hidden_size, num_labels)

    def
