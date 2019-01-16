from .model import BLSTMCRF
from .dataset import Dataset
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
import MeCab


class ModelAPI():

    def __init__(self, model_path: str, train_path: str,
                 wordemb_path: str, charemb_path: str,
                 hidden_size: int):
        """
        :param model_path: trained model file path (.pth)
        :param train_path: file path used training
        :param wordemb_path: path of word embedding used training
        :param charemb_path: path of char embedding used training
        :param hidden_size: size of hidden layer
        """

        self.mecab = MeCab.Tagger('-Owakati')
        self.WORD = data.Field(batch_first=True)
        self.CHAR = data.Field(batch_first=True)
        self.LABEL = data.Field(batch_first=True)
        self.fields = [('char', self.CHAR), ('word', self.WORD), ('label', self.LABEL)]
        self.dataset = datasets.SequenceTaggingDataset(path=train_path, fields=self.fields,
                                                       separator='\t')
        self.CHAR.build_vocab(self.dataset, vectors=Vectors(charemb_path))
        self.WORD.build_vocab(self.dataset, vectors=Vectors(wordemb_path))
        self.LABEL.build_vocab(self.dataset)
        self.model = BLSTMCRF(len(self.LABEL.vocab.itos), hidden_size, 0.0,
                              self.WORD.vocab.vectors.size()[1],
                              self.CHAR.vocab.vectors.size()[1])
        self.model.load(model_path)

    def predict(self, sentence: str):
        """
        :param sentence: A japanese sentence
        """

        morphs = self.mecab.parse(sentence)
        morphs = morphs.split(' ')[:-1]
        words = []
        chars = []
        for morph in morphs:
            for c in morph:
                words.append(morph)
                chars.append(c)

        word = torch.Tensor([self.WORD.vocab.stoi[w] for w in words]).long()
        word = self.WORD.vocab.vectors[word].view(1, -1, 50)
        mask = word != 1
        mask = mask.float()[:, :, 0]
        char = torch.Tensor([self.CHAR.vocab.stoi[c] for c in chars]).long()
        char = self.CHAR.vocab.vectors[char].view(1, -1, 50)
        x = {'word': word, 'char': char}
        decode = self.model.decode(x, mask)
        return [self.LABEL.vocab.itos[d] for d in decode[0]]
