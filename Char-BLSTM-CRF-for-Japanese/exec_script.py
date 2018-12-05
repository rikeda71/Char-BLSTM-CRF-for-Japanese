from trainer import Trainer
import torch.optim
t = Trainer(optimizer=torch.optim.Adam, hidden_size=300, batch_size=32, pad_idx=1,
            wordemb_path='/home/s14t284/Desktop/wordembed50/all_vectors.txt',
            charemb_path='/home/s14t284/Desktop/charembed50_5/all_vectors.txt',
            train_path='train.txt', test_path='test.txt')
t.train(240)
