from trainer import Trainer
import torch.optim
from time import time
from miner import Miner

t = Trainer(optimizer=torch.optim.Adam, hidden_size=300, batch_size=32, pad_idx=1,
            wordemb_path='/home/s14t284/Desktop/wordembed500/all_vectors.txt',
            charemb_path='/home/s14t284/Desktop/charembed50_5/all_vectors.txt',
            train_path='train.txt', test_path='test.txt', dropout_rate=0.5)
now = time()
t.train(100)
print(str(time() - now) + "秒")
t.model.load('weight.pth')

iterator = t.test.return_batch(t.batch_size)
predict = []
answer = []
sentence = []
# テストセットに対して，ラベルの予測を行う
for i, data in enumerate(iterator):
    word = t.dataset.WORD.vocab.vectors[data.word]
    char = t.dataset.CHAR.vocab.vectors[data.char]
    mask = data.word != 1
    mask = mask.float()
    x = {'word': word, 'char': char}
    decode = t.model.decode(x, mask)

    for pred, ans, c in zip(decode, data.label, data.char):
        answer.append([t.dataset.LABEL.vocab.itos[i] for i in ans[:len(pred)]])
        predict.append([t.dataset.LABEL.vocab.itos[i] for i in pred])
        sentence.append([t.dataset.CHAR.vocab.itos[i] for i in c[:len(pred)]])

# 既知単語の取得のため，学習データの単語も格納
train_iterator = t.dataset.return_batch(t.batch_size)
known_answer = []
known_sentence = []
for i, data in enumerate(train_iterator):
    word = t.dataset.WORD.vocab.vectors[data.word]
    char = t.dataset.CHAR.vocab.vectors[data.char]
    mask = data.word != 1
    mask = mask.float()
    x = {'word': word, 'char': char}
    decode = t.model.decode(x, mask)

    for pred, ans, c in zip(decode, data.label, data.char):
        known_answer.append([t.dataset.LABEL.vocab.itos[i] for i in ans[:len(pred)]])
        known_sentence.append([t.dataset.CHAR.vocab.itos[i] for i in c[:len(pred)]])

known_m = Miner(known_answer, [['']], known_sentence, {'PRO': [], 'SHO': []})
knowns = known_m.return_answer_named_entities()['unknown']

m = Miner(answer, predict, sentence, knowns)
m.default_report(True)
m.unknown_only_report(True)
m.known_only_report(True)
