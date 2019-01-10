import matplotlib.pyplot as plt
from .trainer import Trainer
from .reporter import Reporter
from .early_stopping import EarlyStopping


class Experimenter():

    def __init__(self, optimizer,
                 wordemb_path: str, charemb_path: str,
                 train_path: str, test_path: str, dev_path: str,
                 epoch_size: int, batch_size: int,
                 hidden_size: int, dropout_rate: float = 0.0,
                 learning_rate: float = 1e-3, clip_grad_num: float = 5.0,
                 save_path: str = 'weight.pth'):
        """
        :param optimizer: optimizer method
        :param wordemb_path: path of word embedding
        :param charemb_path: path of character embedding
        :param train_path: path of train dataset
        :param test_path: path of test dataset
        :param dev_path: path of development dataset
        :param epoch_size: size of epoch
        :param batch_size: size of batch
        :param hidden_size: size of hidden matrix
        :param dropout_rate: rate of dropout (0 <= x < 1)
        :param learning_rate: rate of train
        :param clip_grad_num: using gradient clipping
        :param save_path: path of train result
        """

        self.trainer = Trainer(optimizer,
                               wordemb_path, charemb_path,
                               train_path, test_path, dev_path,
                               batch_size, hidden_size,
                               dropout_rate, learning_rate,
                               clip_grad_num, save_path)

        self.epoch_size = epoch_size

    def run(self, label: str, target: str = 'all',
            measured_val: str = 'f1_score', patience: int = 3):
        """
        execute training method.
                argument values are used by early stopping and plotting the graph.
        :param label: NER label (PSN, LOC, PRO).
        :param target: named entity target (all, known, or unknown).
        :param measured_val: kind of measured value (precision, recall, or f1).
        :param patience: using early stopping. if patience <= 0, don't exec early stopping
        """

        assert target in ['all', 'known', 'unknown'],\
            '"target" must be "all", "known", or "unknown"'

        assert measured_val in ['precision', 'recall', 'f1_score'],\
            '"measured_val" must be "precision", "recall", or "f1_score"'

        losses = []
        scores = []
        es = EarlyStopping(patience)

        for epoch in range(self.epoch_size):
            # training
            loss = self.trainer.train()
            reporter = Reporter(self.trainer)
            reporter.predict_to_devset(self.trainer)
            if target == 'all':
                score = reporter.default_report(False)[label][measured_val]
            elif target == 'known':
                score = reporter.known_report(False)[label][measured_val]
            elif target == 'unknown':
                score = reporter.unknown_report(False)[label][measured_val]
            # 損失とF値を保持
            # save loss and F-measure
            losses.append(loss)
            scores.append(score)
            # early stopping
            if es.decision_stop(score):
                break
            # モデルの性能が更新されたら，その都度モデルのベクトルを保存する
            # if metrics is updated, save vectors of model
            if es.bad_epochs == 0:
                self.trainer.save()

        self.trainer.model.load(self.trainer.save_path)
        # 学習結果の表示
        self.reporter = Reporter(self.trainer)
        self.reporter.predict_to_testset(self.trainer)
        self.reporter.all_report()

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

        return {'all': self.reporter.default_report(False),
                'unknown': self.reporter.unknown_report(False),
                'known': self.reporter.known_report(False)}
