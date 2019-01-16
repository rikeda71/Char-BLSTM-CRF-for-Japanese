import torch
from .model import BLSTMCRF
from .dataset import Dataset


class Trainer():

    def __init__(self, optimizer,
                 wordemb_path: str, charemb_path: str,
                 train_path: str, test_path: str, dev_path: str,
                 batch_size: int, hidden_size: int,
                 dropout_rate: float = 0.0, learning_rate: float = 1e-3,
                 clip_grad_num: float = 5.0,
                 save_path: str = 'weight.pth'):
        """
        :param optimizer: optimizer method
        :param wordemb_path: path of word embedding
        :param charemb_path: path of character embedding
        :param train_path: path of train dataset
        :param test_path: path of test dataset
        :param dev_path: path of development dataset
        :param batch_size: size of batch
        :param hidden_size: size of hidden matrix
        :param dropout_rate: rate of dropout (0 <= x < 1)
        :param learning_rate: rate of train
        :param clip_grad_num: using gradient clipping
        :param save_path: path of train result
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainset = Dataset(train_path, wordemb_path, charemb_path, self.device)
        self.testset = Dataset(test_path, wordemb_path, charemb_path, self.device)
        self.devset = Dataset(dev_path, wordemb_path, charemb_path, self.device)
        self.batch_size = batch_size
        self.save_path = save_path
        dim_sizes = self.trainset.return_embedding_dim()
        num_labels = self.trainset.return_num_label_kind()
        self.model = BLSTMCRF(num_labels, hidden_size, dropout_rate,
                              dim_sizes['word'], dim_sizes['char']).to(self.device)
        self.optimizer = optimizer(params=self.model.parameters(), lr=learning_rate)
        self.clip_grad_num = clip_grad_num
        self.learn_num = 0
        self.epoch_num = 0

    def train(self) -> float:
        """
        training
        :return:
        """

        # 学習データすべてのloss
        # loss of all of learning data
        all_loss = 0.0

        iterator = self.trainset.return_batch(self.batch_size)
        for i, data in enumerate(iterator):
            # maskを作成
            # making mask
            mask = data.word != 1
            mask = mask.float().to(self.device)

            # バッチ内に含まれる文の中の文字と文字を含む単語，文字に対しての固有表現ラベルを用意
            # prepare words and characters in sentence, and labels for characters
            word = self.trainset.WORD.vocab.vectors[data.word].to(self.device)
            char = self.trainset.CHAR.vocab.vectors[data.char].to(self.device)
            labels = data.label.to(self.device)
            x = {'word': word, 'char': char}

            # training
            self.optimizer.zero_grad()
            loss = self.model(x, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.clip_grad_num)
            self.optimizer.step()

            all_loss += loss.tolist()
            self.learn_num += 1

        self.epoch_num += 1
        # print loss
        print('[%d, %5d] loss: %.3f' %
              (self.epoch_num, self.learn_num, all_loss))

        return all_loss

    def save(self):
        """
        save training model
        """

        torch.save(self.model.state_dict(), self.save_path)
