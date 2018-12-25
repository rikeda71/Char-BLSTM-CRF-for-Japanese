from typing import Dict
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors


class Dataset():

    WORD = data.Field(batch_first=True)
    CHAR = data.Field(batch_first=True)
    LABEL = data.Field(batch_first=True)

    def __init__(self, text_path: str,
                 wordembed_path: str, charembed_path: str):
        """
        The form of dataset
        想定しているデータセットの形
        私は白い恋人を食べました
        私  私  O
        は  は  O
        白  白い    B-PRO
        い  白い    I-PRO
        恋  恋人    I-PRO
        人  恋人    I-PRO
        を  を  O
        食  食べ    O
        べ  食べ    O
        ま  まし    O
        し  まし    O
        た  た  O
        """

        self.fields = [('char', self.CHAR), ('word', self.WORD), ('label', self.LABEL)]
        self.dataset = datasets.SequenceTaggingDataset(path=text_path, fields=self.fields)
        self.CHAR.build_vocab(self.dataset, vectors=Vectors(charembed_path))
        self.WORD.build_vocab(self.dataset, vectors=Vectors(wordembed_path))
        self.LABEL.build_vocab(self.dataset)

    def return_dataset(self):
        """
        return dataset variable
        """

        return self.dataset

    def return_batch(self, batch_size: int):
        """
        return dataset in batches (iterator form)
        :param batch_size: size of batches
        """

        return data.BucketIterator(dataset=self.dataset,
                                   batch_size=batch_size,
                                   sort=True,
                                   sort_key=lambda x: len(x.char),
                                   repeat=False)

    def return_embedding_dim(self) -> Dict[str, int]:
        """
        return size of embedding dimension (word embedding dimension and char embedding dimension)
        """

        char_dim = self.CHAR.vocab.vectors.shape[1]
        word_dim = self.WORD.vocab.vectors.shape[1]
        return {'char': char_dim, 'word': word_dim}

    def return_num_label_kind(self) -> int:
        """
        return number of kind of label
        """

        return len(self.LABEL.vocab.itos)
