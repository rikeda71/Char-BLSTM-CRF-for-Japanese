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
pip install git+https://github.com/Andolab/miNER#egg=miNER
pip install git+https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese#egg=deepjapaner
```


## Usage

I prepared sample codes. Please see [exec\_script.py](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/exec_script.py) or [exec\_sample.ipynb](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/exec_sample.ipynb).

### Class parameter description

- Trainer(optiminzer, hidden\_size, batch\_size, wordemb\_path, charemb\_path, train\_path, test\_path, dropout\_rate, learning\_rate, save\_path)

|  parameter  |  description  |
| ---- | ---- |
|  optimizer  |  setting pytorch optimizer method(torch.optim.\*). For example, torch.optim.Adam, torch.optim.SGD, etc...  |
|  hidden\_size  |  hidden layer size of Bidirectional LSTM.  |
|  batch\_size  |  batch size using training Neural Network. |
|  wordemb\_path  |  file path of word embedding (.txt) |
|  charemb\_path  |  file path of char embedding (.txt)  |
|  train\_path  |  file path of train dataset  |
|  test\_path  |  file path of test dataset  |
|  dropout\_rate  |  dropout rate (0 <= rate < 1). defaut 0.0  |
|  learning\_rate  |  learning rate. default 1e-3  |
|  save\_path  |  model save path (.pth)  |

- Reporter(trainer)

|  parameter  |  description  |
| ---- | ---- |
|  trainer  |  setting Trainer() class  |


## Reference
- S Misawa, M Taniguchi, et al. [Character-based Bidirectional LSTM-CRF with words and characters for Japanese Named Entity Recognition](http://www.aclweb.org/anthology/W17-4114) (ACL2017)


## License

MIT

