import torch
from miner import Miner
from typing import Dict


class Reporter():

    def __init__(self, trainer: torch.nn.Module):
        """
        :param trainer: Neural Network Model
        """

        self._answers = []
        self._predicts = []
        self._sentences = []
        self._correct_known_words(trainer)

    def all_report(self):
        """
        showing default, unknown only, and known only reports in the console
        """

        print('all')
        self.default_report()
        print('\nunknown')
        self.unknown_report()
        print('\nknown')
        self.known_report()

    def default_report(self, print_: bool = True) \
            -> Dict[str, Dict[str, float]]:
        """
        return a report of named entity recognition experiment
        :param print_: if True, showing a report in the console
        """

        return self.miner.default_report(print_)

    def known_report(self, print_: bool = True) -> Dict[str, Dict[str, float]]:
        """
        return a report of known named entity recognition experiment
        :param print_: if True, showing a report in the console
        """

        return self.miner.known_only_report(print_)

    def unknown_report(self, print_: bool = True) \
            -> Dict[str, Dict[str, float]]:
        """
        return a report of unknown named entity recognition experiment
        :param print_: if True, showing a report in the console
        """

        return self.miner.unknown_only_report(print_)

    def predict_to_devset(self, trainer):
        """
        predicting labels of dev dataset
        :param trainer: Neural Network Model
        """

        iterator = trainer.devset.return_batch(trainer.batch_size)
        vecs = trainer.devset
        for i, data in enumerate(iterator):
            with torch.no_grad():
                word = vecs.WORD.vocab.vectors[data.word].to(trainer.device)
                char = vecs.CHAR.vocab.vectors[data.char].to(trainer.device)
                mask = data.word != 1
                mask = mask.float().to(trainer.device)
                x = {'word': word, 'char': char}
                decode = trainer.model.decode(x, mask)

                for pred, ans, c in zip(decode, data.label, data.char):
                    self._answers.append([trainer.devset.LABEL.vocab.itos[i]
                                          for i in ans[:len(pred)]])
                    self._predicts.append([trainer.devset.LABEL.vocab.itos[i]
                                           for i in pred])
                    self._sentences.append([trainer.devset.CHAR.vocab.itos[i]
                                            for i in c[:len(pred)]])
        self.miner = Miner(self._answers, self._predicts,
                           self._sentences, self._known_words)

    def predict_to_testset(self, trainer):
        """
        predicting labels of test dataset
        :param trainer: Neural Network Model
        """

        iterator = trainer.testset.return_batch(trainer.batch_size)
        vecs = trainer.testset
        for i, data in enumerate(iterator):
            with torch.no_grad():
                word = vecs.WORD.vocab.vectors[data.word].to(trainer.device)
                char = vecs.CHAR.vocab.vectors[data.char].to(trainer.device)
                mask = data.word != 1
                mask = mask.float().to(trainer.device)
                x = {'word': word, 'char': char}
                decode = trainer.model.decode(x, mask)

                for pred, ans, c in zip(decode, data.label, data.char):
                    self._answers.append([trainer.testset.LABEL.vocab.itos[i]
                                          for i in ans[:len(pred)]])
                    self._predicts.append([trainer.testset.LABEL.vocab.itos[i]
                                           for i in pred])
                    self._sentences.append([trainer.testset.CHAR.vocab.itos[i]
                                            for i in c[:len(pred)]])
        self.miner = Miner(self._answers, self._predicts,
                           self._sentences, self._known_words)

    def _correct_known_words(self, trainer):
        """
        correcting known named entities (named entities in train dataset)
        :param trainer: Neural Network Model
        """

        train_iterator = trainer.trainset.return_batch(trainer.batch_size)
        vecs = trainer.trainset
        known_answer = []
        known_sentence = []
        for i, data in enumerate(train_iterator):
            with torch.no_grad():
                word = vecs.WORD.vocab.vectors[data.word].to(trainer.device)
                char = vecs.CHAR.vocab.vectors[data.char].to(trainer.device)
                mask = data.word != 1
                mask = mask.float().to(trainer.device)
                x = {'word': word, 'char': char}
                decode = trainer.model.decode(x, mask)

                for pred, ans, c in zip(decode, data.label, data.char):
                    known_answer.append([trainer.trainset.LABEL.vocab.itos[i]
                                         for i in ans[:len(pred)]])
                    known_sentence.append([trainer.trainset.CHAR.vocab.itos[i]
                                           for i in c[:len(pred)]])

        miner = Miner(known_answer, [['']],
                      known_sentence, {'PRO': [], 'SHO': []})
        self._known_words = miner.return_answer_named_entities()['unknown']
