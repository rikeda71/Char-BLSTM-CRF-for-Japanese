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

I prepared sample codes. Please see [train\_sample.py](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/train_sample.py), [predict\_sample.py](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/predict_sample.py) or [exec\_sample.ipynb](https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese/blob/master/exec_sample.ipynb).

### Class parameter description

- Experiment(optiminzer, wordemb\_path, charemb\_path, train\_path, test\_path, dropout\_rate,
             epoch\_size, batch\_size, hidden\_size, learning\_rate, clip\_grad\_num, save\_path)

|  parameter  |  description  |
| ---- | ---- |
|  optimizer  |  setting pytorch optimizer method(torch.optim.\*). For example, torch.optim.Adam, torch.optim.SGD|
|  wordemb\_path  |  file path of word embedding (.txt) |
|  charemb\_path  |  file path of char embedding (.txt)  |
|  train\_path  |  file path of train dataset  |
|  test\_path  |  file path of test dataset  |
|  dev\_path  |  file path of develop dataset  |
|  epoch\_size  |  epoch size using training Neural Network. |
|  batch\_size  |  batch size using training Neural Network. |
|  hidden\_size  |  hidden layer size of Bidirectional LSTM.  |
|  dropout\_rate  |  dropout rate (0 <= rate < 1). \[0.0\]  |
|  learning\_rate  |  learning rate. \[1e-3\]  |
|  clip\_grad\_num  |  using gradient clipping. \[5.0\] |
|  save\_path  |  model save path (.pth)  |

| method | description |
| ---- | ---- |
| run(label, target, measured\_value, patience) | execute a Named Entity Recognition experiment. Please give name of named entity label to "label", and give value of int type to "patience". "label" and "patience" are used in early stopping.


- ModelAPI(model\_path, train\_path, wordemb\_path, charemb\_path, hidden\_size)

| parameter  |  description  |
| ---- | ---- |
| model\_path | trained model file path (.pth) |
| train\_path | file path used training |
| wordemb\_path | path of word embedding used training |
| charemb\_path | path of char embedding used training |
| hidden\_size | size of hidden layer |

| method | description |
| ---- | ---- |
| predict(sentence) | predict Named Entity label for sentence. Please give a Japanese sentence to sentence of argument parameter


## Reference
- S Misawa, M Taniguchi, et al. [Character-based Bidirectional LSTM-CRF with words and characters for Japanese Named Entity Recognition](http://www.aclweb.org/anthology/W17-4114) (ACL2017)


## License

MIT

