import random
from random import Random
import re
from typing import List, Tuple, Union, NoReturn
import gzip
import math
from enum import Enum, auto
import os


class SplitType(Enum):
    train = auto()
    dev = auto()
    test = auto()


class LangDirection(Enum):
    source = auto()
    target = auto()


def load_ted(file_path: str,
             lowercase: bool = True) -> List[str]:
    """
    Loads and cleans data from TED Multilingual Corpus.
    It is expected that the specified file contains data only for a single language.
    :param file_path: path to the file containing single language sentences
    :param lowercase: whether the output should be lowercase
    :return: list of sentences
    """
    with open(file_path) as file:
        sentences = file.readlines()
        cleaned = []
        for sentence in sentences:
            sentence = clean_ted(sentence)
            if lowercase:
                sentence = sentence.lower()
            cleaned.append(sentence)
        return cleaned


def clean_ted(text: str) -> str:
    """
    Helper function that removes unnecessary information from lines in TED Corpus
    :param text: a line in TED Corpus
    :return: cleaned line
    """
    return re.findall(r"^\d+:[a-z]{2}:(.+$)", text)[0].strip('\n')


def load_anki(file_path: str):
    with open(file_path) as file:

        source_sentences = []
        target_sentences = []

        for line in file:
            line = line.split('\t')
            source_sentences.append(line[0])
            target_sentences.append(line[1])

        return source_sentences, target_sentences


def clean_anki(sentences: List[str]) -> List[str]:
    return [re.sub(r"[^а-яёa-z ]", "", sentence.lower()) for sentence in sentences]


def basic_load(file_path: str) -> List[str]:
    with open(file_path) as file:
        return file.readlines()


def basic_gzip_load(file_path: str, max_lines: int = 100_000) -> List[str]:
    result = []
    with gzip.open(file_path) as file:
        for index, line in enumerate(file):
            result.append(line.decode('utf-8'))
            if index > max_lines:
                break
    return result


def shuffle_sentences(source: List[str],
                      target: List[str],
                      seed: Union[int, None] = None) -> Tuple[List[str], List[str]]:
    """
    Shuffles lists of source and target sentences simultaneously.
    :param source: list of source sentences
    :param target: list of target sentences
    :param seed:
    :return: shuffled lists of source and target sequences (source -> translation pairs are kept)
    """

    assert len(source) == len(target), '`source` and `target` must have the same number of elements!'

    local_random = Random()

    if seed:
        local_random.seed(seed)

    pairs = list(zip(source, target))
    local_random.shuffle(pairs)
    source, target = zip(*pairs)
    return source, target


def split_data(source: List[str],
               target: List[str],
               data_path: str = '/data',
               seed: Union[None, int] = None,
               train_size: float = 0.85,
               dev_size: float = 0.1,
               test_size: float = 0.05):

    assert train_size + dev_size + test_size == 1
    assert len(source) == len(target)

    source, target = shuffle_sentences(source, target, seed)

    dev_start = math.floor(len(source) * train_size)
    test_start = math.floor(len(source) * (train_size + dev_size))

    source_train, source_dev, source_test = source[:dev_start], source[dev_start:test_start], source[test_start:]
    target_train, target_dev, target_test = target[:dev_start], target[dev_start:test_start], target[test_start:]

    split_helper(source_train, LangDirection.source, data_path, SplitType.train)
    split_helper(source_dev, LangDirection.source, data_path, SplitType.dev)
    split_helper(source_train, LangDirection.source, data_path, SplitType.test)

    split_helper(target_train, LangDirection.target, data_path, SplitType.train)
    split_helper(target_dev, LangDirection.target, data_path, SplitType.dev)
    split_helper(target_test, LangDirection.target, data_path, SplitType.test)


def split_helper(data: List[str],
                 direction: LangDirection,
                 folder: str,
                 split_part: SplitType) -> NoReturn:

    if not os.path.exists(folder):
        os.makedirs(folder)

    path = f'{folder}/{direction.name}.{split_part.name}'

    with open(path, 'w') as file:
        for sentence in data:
            file.write(sentence)

