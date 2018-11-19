from torchtext import data, datasets


class Dataset():

    WORD = data.Field(init_token='<bos>', eos_token='<eos>')
    CHAR = data.Field(init_token='<bos>', eos_token='<eos>')
    LABEL = data.Field(init_token='<bos>', eos_token='<eos>')

    def __init__(self, text_path: str, wordembed, charembed):
        """
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
        self.CHAR.build_vocab(self.dataset, vectors=charembed)
        self.WORD.build_vocab(self.dataset, vectors=wordembed)
        self.LABEL.build_vocab(self.dataset)
        print(self.WORD.vocab.freqs)
        print(self.CHAR.vocab.freqs)
        print(self.LABEL.vocab.freqs)

    def return_dataset(self):
        return self.dataset

    def return_batch(self, batch_size: int):
        return data.BucketIterator(dataset=self.dataset,
                                   batch_size=batch_size,
                                   sort_key=lambda x: len(x))
