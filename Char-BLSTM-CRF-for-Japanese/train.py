import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import BLSTMCRF
from dataset import Dataset


class Trainer():

    def __init__(self, optimizer,
                 hidden_size: int, batch_size: int,
                 dropout_rate: float, pad_idx: int,
                 wordemb_path: str, charemb_path: str,
                 text_path: str, save_path: str='weight.pth'):

        self.batch_size = batch_size
        self.dataset = Dataset(text_path, wordemb_path, charemb_path)
        dim_sizes = self.dataset.return_embedding_dim()
        num_labels = self.dataset.return_num_label_kind()
        self.model = BLSTMCRF(num_labels, hidden_size, dropout_rate, pad_idx, dim_sizes['word'], dim_sizes['char'])
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-3)

    def train(self, epoch_size: int=5):
        for epoch in range(epoch_size):
            # 学習データすべてのloss
            # loss of all of learning data
            all_loss = 0.0

            iterator = self.dataset.return_batch(self.batch_size)
            for i, data in enumerate(iterator):
                word = self.dataset.WORD.vocab.vectors[data.word]
                char = self.dataset.CHAR.vocab.vectors[data.char]
                # maskを作成
                mask = data.word != 1
                mask = mask.float()
                labels = data.label
                x = {'word': word, 'char': char}

                # reset gradient
                self.optimizer.zero_grad()

                loss = -torch.mean(self.model(x, labels, mask))

                loss.backward()
                self.optimizer.step()

                # print loss
                all_loss += loss.tolist()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, all_loss))
            all_loss = 0.0
        torch.save(self.model.state_dict(), save_path)

    def load(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))

    def predict(self):
        iterator = self.dataset.return_batch(self.batch_size)
        for i, data in enumerate(iterator):
            word = self.dataset.WORD.vocab.vectors[data.word]
            char = self.dataset.CHAR.vocab.vectors[data.char]
            # maskを作成
            mask = data.word != 1
            mask = mask.float()
            labels = data.label
            x = {'word': word, 'char': char}
            decode = self.model.decode(x, mask)
            for a, b, c in zip(decode, data.label, data.char):
                print('predict', a)
                print('answer', b[:len(a)])
                print('sentence', [self.dataset.CHAR.vocab.itos[i]
                                   for i in c])
