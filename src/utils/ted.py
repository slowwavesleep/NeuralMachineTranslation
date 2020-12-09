import re
from typing import List


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

