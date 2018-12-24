# Char-BLSTM-CRF-for-Japanese

An Inplementation of Character based Bidirectional LSTM CRF for Japanese.

This library is a library of name entity recognition (NER) for Japanese and reproduces the [S Misawa et al., 2017](http://www.aclweb.org/anthology/W17-4114) with pytorch.


## Requirements

- python3 (>=3.5)
- pytorch 1.0
- [TorchCRF](https://github.com/s14t284/TorchCRF)
- [miNER](https://github.com/Andolab/miNER)

## Installation

```shell
pip install git+https://github.com/s14t284/TorchCRF#egg=TorchCRF
pip install git+https://github.com/Andolab/miNERegg=miNER
pip install git+https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese#egg=CBCJ
```


## Usage

I prepared sample codes. Please see [exec\_script](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/exec_script.py) or [exec\_sample.ipynb](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/exec_sample.pynb) .


## Reference
- S Misawa, M Taniguchi, et al. [Character-based Bidirectional LSTM-CRF with words and characters for Japanese Named Entity Recognition](http://www.aclweb.org/anthology/W17-4114)


## License

MIT

