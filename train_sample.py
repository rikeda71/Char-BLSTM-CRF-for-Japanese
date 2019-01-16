import torch.optim
from time import time
from deepjapaner.experimenter import Experimenter

e = Experimenter(optimizer=torch.optim.Adam,
                 wordemb_path='wordvectors',
                 charemb_path='charvectors',
                 train_path='train.txt',
                 test_path='dev.txt',
                 dev_path='dev.txt',
                 hidden_size=300, batch_size=32, epoch_size=30,
                 )

now = time()
e.run(label='PSN', target='unknown', measured_val='f1_score', patience=3)
print(str(time() - now) + "秒")
