import torch.optim
from time import time
from deepjapaner.experimenter import Experimenter

e = Experimenter(optimizer=torch.optim.Adam,
                 wordemb_path='word_vectors.txt', charemb_path='char_vectors.txt',
                 train_path='train.txt', test_path='test.txt', dev_path='test.txt',
                 hidden_size=300, batch_size=32, epoch_size=30,
                 )

now = time()
e.run(label='PRO', target='unknown', measured_val='f1_score', patience=3)
print(str(time() - now) + "秒")
