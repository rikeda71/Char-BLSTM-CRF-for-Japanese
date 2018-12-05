from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from TorchCRF.crf import CRF


class BLSTMCRF(nn.Module):
    CUDA = torch.cuda.is_available()

    def __init__(self, num_labels: int, hidden_size: int,
                 dropout_rate: float, pad_idx: int,
                 wordemb_dim: int, charemb_dim: int):
        """

        :param num_labels: number of label
        :param hidden_size: size of hidden state
        :param dropout_rate: dropout rate (0.0 <= dropout_rate < 1.0)
        :param pad_idx: index of padding character in word
        :param wordemb_dim: dimension of word embedding
        :param charemb_dim: dimension of character embedding
        """

        super().__init__()
        self.blstm = BLSTM(num_labels, hidden_size, dropout_rate, pad_idx, wordemb_dim, charemb_dim)
        self.crf = CRF(num_labels)
        self = self.cuda() if BLSTM.CUDA else self

    def forward(self, x: Dict[str, torch.Tensor], y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        :param x: word and character embedding
                   {'word': embedding of word including character, 'char': character embedding}
        :param y: labels of sequence in batch
        :param mask: masking of sequence (1 or 0)
        :return: score of Bidirectional LSTM CRF forward
        """

        # paddingの位置を導出
        h = self.blstm.forward(x, mask)
        score = self.crf.forward(h, y, mask)
        return -torch.mean(score)

    def decode(self, x: Dict[str, torch.Tensor], mask: torch.Tensor) -> List[int]:
        """

        :param x: word and character embedding
                   {'word': embedding of word including character, 'char': character embedding}
        :return: labels of x
        """

        h = self.blstm.forward(x, mask)
        labels = self.crf.viterbi_decode(h, mask)
        return labels

    def load(self, model_path: str) -> None:
        """
        load training model
        :param model_path: path of model file (.pth)
        :return: None
        """

        self.load_state_dict(torch.load(model_path))


class BLSTM(nn.Module):
    CUDA = torch.cuda.is_available()

    def __init__(self, num_labels: int, hidden_size: int,
                 dropout_rate: int, pad_idx: int,
                 wordemb_dim: int, charemb_dim: int):
        """

        :param num_labels: number of label
        :param hidden_size: size of hidden state
        :param dropout_rate: dropout rate (0.0 <= dropout_rate < 1.0)
        :param pad_idx: index of padding character in word
        :param wordemb_dim: dimension of word embedding
        :param charemb_dim: dimension of character embedding
        """

        super().__init__()

        self.hidden = None
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=wordemb_dim + charemb_dim,
            hidden_size=hidden_size // 2,
            num_layers=1,
            bias=True,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )

        self.out = nn.Linear(hidden_size, num_labels)

    def init_hidden(self, batch_size) -> Tuple[torch.Tensor]:
        """
        initialize hidden state
        :return: (hidden state, cell of LSTM)
        """

        h = self.zeros(2, batch_size, self.hidden_size // 2)
        c = self.zeros(2, batch_size, self.hidden_size // 2)
        return h, c

    def forward(self, x: Dict[str, np.ndarray], mask: torch.Tensor):
        """

        :param x: word and character embedding
                   {'word': embedding of word including character, 'char': character embedding}
                   (batch_size, sequence_len, embedding_size)
        :param mask: masking of sequence (1 or 0)
        :return: score of LSTM forward
        """

        batch_size = x['word'].shape[0]
        self.hidden = self.init_hidden(batch_size)
        x = torch.cat((x['word'], x['char']), 2)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.lstm(x, self.hidden)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        h *= mask.unsqueeze(-1)
        return h

    @staticmethod
    def zeros(*args):
        x = torch.zeros(*args)
        return x.cuda() if BLSTM.CUDA else x
