import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import BLSTMCRF
from dataset import Dataset


class Trainer():

    def __init__(self, optimizer, net: nn.Module,
                 num_labels: int, hidden_size: int,
                 batch_size: int, dropout_rate: int, pad_idx: int,
                 wordemb: np.ndarray, charemb: np.ndarray,
                 text_path: str):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=net, lr=1e-3)
        self.model = BLSTMCRF(num_labels, hidden_size, batch_size, dropout_rate, pad_idx, wordemb, charemb)
        self.dataset = Dataset(text_path, wordemb, charemb)
        self.iterator = self.dataset.return_batch(batch_size)

    def train(self, epoch_size: int=5):
        for epoch in range(epoch_size):
            # 学習データすべてのloss
            # loss of all of learning data
            all_loss = 0.0

            for i, data in enumerate(self.iterator):
                char, word, labels = data
                x = {'word': word, 'char': char}

                # reset gradient
                self.optimizer.zero_grad()

                output = self.model(x, labels)
                
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                # print loss
                all_loss += loss.data[0]
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, all_loss / 2000))
            all_loss = 0.0