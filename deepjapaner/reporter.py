import torch
from miner import Miner
from typing import List, Dict


class Reporter():

    def __init__(self, trainer: torch.nn.Module):
        """
        :param trainer: Neural Network Model
        """

        self._answers = []
        self._predicts = []
        self._sentences = []
        self._predict_to_testset(trainer)
        self._correct_known_words(trainer)
        self.miner = Miner(self._answers, self._predicts, self._sentences, self._known_words)

    def all_report(self):
        """
        showing default, unknown only, and known only reports in the console
        """

        print('all')
        self.default_report()
        print('unknown')
        self.unknown_report()
        print('known')
        self.known_report()

    def default_report(self, print_: bool=True) -> Dict[str, Dict[str, float]]:
        """
        return a report of named entity recognition experiment
        :param print_: if True, showing a report in the console
        """

        return self.miner.default_report(print_)

    def known_report(self, print_: bool=True) -> Dict[str, Dict[str, float]]:
        """
        return a report of known named entity recognition experiment
        :param print_: if True, showing a report in the console
        """

        return self.miner.known_only_report(print_)

    def unknown_report(self, print_: bool=True) -> Dict[str, Dict[str, float]]:
        """
        return a report of unknown named entity recognition experiment
        :param print_: if True, showing a report in the console
        """

        return self.miner.unknown_only_report(print_)

    def _predict_to_testset(self, trainer):
        """
        predicting labels of test dataset
        :param trainer: Neural Network Model
        """

        iterator = trainer.test.return_batch(trainer.batch_size)
        for i, data in enumerate(iterator):
            word = trainer.test.WORD.vocab.vectors[data.word]
            char = trainer.test.CHAR.vocab.vectors[data.char]
            mask = data.word != 1
            mask = mask.float()
            x = {'word': word, 'char': char}
            decode = trainer.model.decode(x, mask)

            for pred, ans, c in zip(decode, data.label, data.char):
                self._answers.append([trainer.test.LABEL.vocab.itos[i]
                                      for i in ans[:len(pred)]])
                self._predicts.append([trainer.test.LABEL.vocab.itos[i]
                                       for i in pred])
                self._sentences.append([trainer.test.CHAR.vocab.itos[i]
                                        for i in c[:len(pred)]])

    def _correct_known_words(self, trainer):
        """
        correcting known named entities (named entities in train dataset)
        :param trainer: Neural Network Model
        """

        train_iterator = trainer.dataset.return_batch(trainer.batch_size)
        known_answer = []
        known_sentence = []
        for i, data in enumerate(train_iterator):
            word = trainer.dataset.WORD.vocab.vectors[data.word]
            char = trainer.dataset.CHAR.vocab.vectors[data.char]
            mask = data.word != 1
            mask = mask.float()
            x = {'word': word, 'char': char}
            decode = trainer.model.decode(x, mask)

            for pred, ans, c in zip(decode, data.label, data.char):
                known_answer.append([trainer.dataset.LABEL.vocab.itos[i]
                                     for i in ans[:len(pred)]])
                known_sentence.append([trainer.dataset.CHAR.vocab.itos[i]
                                       for i in c[:len(pred)]])

        miner = Miner(known_answer, [['']], known_sentence, {'PRO': [], 'SHO': []})
        self._known_words = miner.return_answer_named_entities()['unknown']
