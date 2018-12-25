import torch.optim
from time import time
from miner import Miner
from deepjapaner.trainer import Trainer
from deepjapaner.reporter import Reporter

t = Trainer(optimizer=torch.optim.Adam, hidden_size=300, batch_size=32,
            wordemb_path='word_vectors.txt',
            charemb_path='char_vectors.txt',
            train_path='train.txt', test_path='test.txt', dropout_rate=0.5)
now = time()
t.train(10)
print(str(time() - now) + "秒")
t.model.load('weight.pth')

reporter = Reporter(t)
reporter.all_report()
