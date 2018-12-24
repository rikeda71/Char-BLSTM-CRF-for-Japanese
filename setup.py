from setuptools import setup, find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='CBCJ',
    version='1.0.0',
    description='An Implementation of Character based Bidirectional LSTM CRF for Japanese',
    long_description=readme,
    author='Ryuya Ikeda',
    author_email='rikeda71@gmail.com',
    install_requires=_requires_from_file('requirements.txt'),
    url='https://github.com/s14t284/Char-BLSTM-CRF-for-Japanese',
    license=license,
    packages=['src'],
    python_requires='>=3.5',
    test_suite='tests',
)
