from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='deepjapaner',
    version='1.0.0',
    description='An Implementation of Character based Bidirectional LSTM CRF for Japanese',
    long_description=readme,
    author='Ryuya Ikeda',
    author_email='rikeda71@gmail.com',
    install_requires=['torch==1.0.0', 'torchtext', 'numpy', 'matplotlib', 'seqeval'],
    url='https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese',
    license=license,
    packages=['deepjapaner'],
    python_requires='>=3.5',
    test_suite='tests',
)
