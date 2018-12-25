import torch
import matplotlib.pyplot as plt
from .model import BLSTMCRF
from .dataset import Dataset
from seqeval.metrics import f1_score


class Trainer():

    def __init__(self, optimizer,
                 hidden_size: int, batch_size: int,
                 wordemb_path: str, charemb_path: str,
                 train_path: str, test_path: str,
                 dropout_rate: float=0.0, learning_rate: float=1e-3,
                 save_path: str='weight.pth'):
        """

        :param optimizer: optimizer method
        :param hidden_size: size of hidden matrix
        :param batch_size: size of batch
        :param wordemb_path: path of word embedding
        :param charemb_path: path of character embedding
        :param train_path: path of train dataset
        :param test_path: path of test dataset
        :param dropout_rate: rate of dropout (0 <= x < 1)
        :param learning_rate: rate of train
        :param save_path: path of train result
        """

        self.batch_size = batch_size
        self.dataset = Dataset(train_path, wordemb_path, charemb_path)
        self.test = Dataset(test_path, wordemb_path, charemb_path)
        self.save_path = save_path
        dim_sizes = self.dataset.return_embedding_dim()
        num_labels = self.dataset.return_num_label_kind()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BLSTMCRF(num_labels, hidden_size, dropout_rate, dim_sizes['word'], dim_sizes['char']).to(self.device)
        self.optimizer = optimizer(params=self.model.parameters(), lr=learning_rate)

    def train(self, epoch_size: int=5) -> None:
        """
        training
        :param epoch_size: size of epoch
        :return: None
        """

        learn_num = 0
        losses = []
        scores = []
        for epoch in range(epoch_size):
            # 学習データすべてのloss
            # loss of all of learning data
            all_loss = 0.0

            iterator = self.dataset.return_batch(self.batch_size)
            for i, data in enumerate(iterator):
                # maskを作成
                # making mask
                mask = data.word != 1
                mask = mask.float().to(self.device)

                # バッチ内に含まれる文の中の文字と文字を含む単語，文字に対しての固有表現ラベルを用意
                # prepare words and characters in sentence, and labels for characters
                word = self.dataset.WORD.vocab.vectors[data.word].to(self.device)
                char = self.dataset.CHAR.vocab.vectors[data.char].to(self.device)
                labels = data.label.to(self.device)
                x = {'word': word, 'char': char}

                # training
                self.optimizer.zero_grad()
                loss = self.model(x, labels, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                all_loss += loss.tolist()
                learn_num += 1

            # print loss
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, learn_num, all_loss))

            # 損失とF値を保持
            # save loss and F-measure
            losses.append(all_loss)
            scores.append(self.validate())

            # 学習モデルを保存
            # save training model
            torch.save(self.model.state_dict(), self.save_path)

        # 学習結果のプロット
        # plotting training result
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(range(len(losses)), losses, color="red")
        ax2.plot(range(len(scores)), scores, color="blue")
        ax1.legend(bbox_to_anchor=(0.9, 0.5), loc='upper right', borderaxespad=0.5, labels=['loss'], fontsize=10)
        ax2.legend(bbox_to_anchor=(0.9, 0.6), loc='upper right', borderaxespad=0.5, labels=['score'], fontsize=10)
        plt.xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('F-measure')
        plt.savefig('result.png')

    def validate(self) -> float:
        """
        validate for test dataset
        :return: f-measure of test dataset
        """

        iterator = self.test.return_batch(self.batch_size)
        predict = []
        answer = []
        # テストセットに対して，ラベルの予測を行う
        for i, data in enumerate(iterator):
            with torch.no_grad():
                word = self.dataset.WORD.vocab.vectors[data.word].to(trainer.device)
                char = self.dataset.CHAR.vocab.vectors[data.char].to(trainer.device)
                mask = data.word != 1
                mask = mask.float().to(self.device)
                x = {'word': word, 'char': char}
                decode = self.model.decode(x, mask)

            for pred, ans in zip(decode, data.label):
                answer.append([self.dataset.LABEL.vocab.itos[i] for i in ans[:len(pred)]])
                predict.append([self.dataset.LABEL.vocab.itos[i] for i in pred])

        # F値を返す
        return f1_score(answer, predict)
