from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from TorchCRF.crf import CRF


class BLSTMCRF(nn.Module):
    CUDA = torch.cuda.is_available()

    def __init__(self, num_labels: int, hidden_size: int,
                 batch_size: int, dropout_rate: int, pad_idx: int,
                 wordemb: np.ndarray, charemb: np.ndarray):
        """

        :param num_labels: number of label
        :param hidden_size: size of hidden state
        :param batch_size: size of batch
        :param dropout_rate: dropout rate (0.0 <= dropout_rate < 1.0)
        :param pad_idx: index of padding character in word
        :param wordemb: word embedding
        :param charemb: character embedding
        """

        super().__init__()
        self.blstm = BLSTM(num_labels, hidden_size, batch_size, dropout_rate, pad_idx, wordemb, charemb)
        self.crf = CRF(num_labels)
        self = self.cuda() if BLSTM.CUDA else self

    def forward(self, x: Dict[str, torch.LongTensor], y: torch.LongTensor) -> torch.FloatTensor:
        """

        :param x: word and character embedding
                   {'word': embedding of word including character, 'char': character embedding}
        :param y: labels of sequence in batch
        :return: score of Bidirectional LSTM CRF forward
        """

        # paddingの位置を導出
        mask = x['char'].data.gt(0).float()
        h = self.lstm.forward(x, mask)
        score = self.crf.forward(h, y, mask)
        return score

    def decode(self, x: Dict[str, torch.LongTensor]) -> List[int]:
        """

        :param x: word and character embedding
                   {'word': embedding of word including character, 'char': character embedding}
        :return: labels of x
        """

        mask = x['char'].data.gt(0).float()
        h = self.lstm.forward(x, mask)
        labels = self.crf.viterbi_decode(h, mask)
        return labels


class BLSTM(nn.Module):
    CUDA = torch.cuda.is_available()

    def __init__(self, num_labels: int, hidden_size:int, 
                 batch_size: int, dropout_rate: int, pad_idx: int,
                 wordemb: np.ndarray, charemb: np.ndarray):
        """

        :param num_labels: number of label
        :param hidden_size: size of hidden state
        :param batch_size: size of batch
        :param dropout_rate: dropout rate (0.0 <= dropout_rate < 1.0)
        :param pad_idx: index of padding character in word
        :param wordemb: word embedding
        :param charemb: character embedding
        """

        super().__init__()

        self.hidden = None
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # 単語分散表現のセッティング
        # word embedding setting
        wordvocab_size, wordembed_size = wordemb.shape
        self.wordembed = nn.Embedding(wordvocab_size, wordembed_size, padding_idx=pad_idx)
        self.wordembed.weight = nn.Parameter(torch.from_numpy(wordemb))

        # 文字分散表現のセッティング
        # char embedding setting
        charvocab_size, charembed_size = charemb.shape
        self.charembed = nn.Embedding(charvocab_size, charembed_size, padding_idx=pad_idx)
        self.charembed.weight = nn.Parameter(torch.from_numpy(charemb))

        self.lstm = nn.LSTM(
            input_size=wordembed_size + charembed_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            bias=True,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=True,
        )

        self.out = nn.Linear(hidden_size, num_labels)

    def init_hidden(self) -> Tuple[torch.FloatTensor]:
        """
        initialize hidden state
        :return: (hidden state, cell of LSTM)
        """

        h = self.zeros(2, self.batch_size, self.hidden_size // 2)
        c = self.zeros(2, self.batch_size, self.hidden_size // 2)
        return h, c

    def forward(self, x: Dict[str, np.ndarray], mask: torch.FloatTensor):
        """

        :param x: word and character embedding
                   {'word': embedding of word including character, 'char': character embedding}
        :param mask: masking of sequence (1 or 0)
        :return: score of LSTM forward
        """

        self.hidden = self.init_hidden()
        word_x = self.wordembed(x['word'])
        char_x = self.charembed(x['char'])
        x = torch.cat((word_x, char_x), 0)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.lstm(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        h *= mask.unsqueeze(-1)
        return h

    @staticmethod
    def zeros(*args):
        x = torch.zeros(*args)
        return x.cuda() if BLSTM.CUDA else x
