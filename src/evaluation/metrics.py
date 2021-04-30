from typing import Tuple, List

import sacrebleu


def read_translations(file_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Function to read a text file with the following structure:

    <original sentence>\n
    <reference translation>\n
    <translation done by MT model>\n
    <empty line>

    :param file_path: a string specifying path to file
    :return: a tuple containing a list of source sentences, a list of reference translations,
             and a list of translations done by MT model
    """
    sources, targets, hypotheses = [], [], []

    with open(file_path) as file:
        for index, line in enumerate(file):
            position = index % 4
            if position == 0:
                sources.append(line)
            elif position == 1:
                targets.append(line)
            elif position == 2:
                hypotheses.append(line)

    return sources, targets, hypotheses


def evaluate_corpus_bleu(file_path: int) -> float:
    """
    Reads a file with translations and evaluates BLEU score on given sentences.
    BLEU score has the following range of values: [0, 100] where 0 is the
    worst possible score and 100 is the best one.

    :param file_path: a string specifying path to file
    :return: float value between 0 and 100
    """

    sources, references, hypotheses = read_translations(file_path)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])

    return bleu.score













