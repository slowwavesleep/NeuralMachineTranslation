import youtokentome as yttm
from typing import List, NoReturn
import os
import math


# TODO Add dir check


def make_bpe_data(bpe_text_path: str, sentences: List[str]) -> NoReturn:
    """
    Helper function required to train BPE tokenizer
    :param bpe_text_path: path to temporary file
    :param sentences: list of sentences to write
    :return: does not return anything
    """
    with open(bpe_text_path, 'w') as file:
        for sentence in sentences:
            file.write(sentence + '\n')


def train_bpe(sentences: List[str],
              bpe_text_path: str,
              bpe_model_path: str,
              vocab_size: int) -> NoReturn:
    """
    Trains BPE tokenizer. The model itself is stored in file at the specified location.
    :param sentences: sentences to train the tokenizer or
    :param bpe_text_path: path to write temporary text file
    :param bpe_model_path: path to store the tokenizer
    :param vocab_size: vocabulary size of the resulting tokenizer
    :return: does not return anything
    """
    print("\nAttempting to create temporary data...")
    make_bpe_data(bpe_text_path, sentences)
    print("Temporary data successfully created!")
    yttm.BPE.train(data=bpe_text_path, vocab_size=vocab_size, model=bpe_model_path)
    print("BPE model successfully trained!")
    print("Attempting to remove temporary data...")
    if os.path.exists(bpe_text_path):
        os.remove(bpe_text_path)
        print("Temporary data successfully removed!")
    else:
        print("Failed to remove temporary data")


def batch_tokenize(sentences: List[str],
                   tokenizer: yttm.BPE,
                   batch_size: int = 256,
                   bos: bool = True,
                   eos: bool = True) -> List[List[int]]:
    """
    Tokenize input sentences in batches.
    :param sentences: sentences to tokenize
    :param tokenizer: trained tokenizer model
    :param batch_size: amount of sentences in each batch
    :param bos: whether to add <BOS> symbol at the beginning of each sentence
    :param eos: whether to add <EOS> symbol at the end of each sentence
    :return: a list of tokenized sentences, where each sentence is represented as a list of integers
    """

    tokenized = []

    for i_batch in range(math.ceil(len(sentences) / batch_size)):
        tokenized.extend(
            tokenizer.encode(
                list(sentences[i_batch * batch_size:(i_batch + 1) * batch_size]), bos=bos, eos=eos)
        )
    return tokenized
