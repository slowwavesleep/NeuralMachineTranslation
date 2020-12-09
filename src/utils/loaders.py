from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import List, Tuple
import random


class MTData(Dataset):

    def __init__(self,
                 source: List[List[int]],
                 target: List[List[int]],
                 max_len_source: int = 32,
                 max_len_target: int = 32,
                 pad_index: int = 0,
                 bos_index: int = 2,
                 eos_index: int = 3):

        assert len(source) == len(target), "There must the same amount of source and target texts!"

        self.source = source
        self.target = target
        self.max_len_source = max_len_source
        self.max_len_target = max_len_target
        self.pad_index = pad_index
        self.bos_index = bos_index
        self.eos_index = eos_index

    def __len__(self) -> int:
        return len(self.source)

    def pad_seq(self,
                seq: List[int],
                max_len: int,
                pre_pad: bool = False) -> List[int]:
        """
        Pads input sequence to max length with pad index specified during initialization.
        :param seq: tokenized sequence to pad, i.e. a list of indices
        :param max_len: maximum possible length of a sequence
        :param pre_pad: whether to add at the beginning of the sequence
        :return: a list of indices padded to maximum length
        """
        pads = [self.pad_index] * (max_len - len(seq))

        if pre_pad:
            return pads + seq
        else:
            return seq + pads

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        # it is necessary to subtract 1 to prevent long examples
        # from having mismatched length because <BOS> and <EOS>
        # tags are added after clipping an example to the max size
        encoder_seq = self.source[index][:self.max_len_source - 1]  # <BOS> + ...
        decoder_seq = self.target[index][:self.max_len_target - 1]  # <BOS> + ...
        target_seq = self.target[index][:self.max_len_target - 1]  # ... + <EOS>

        encoder_seq = self.pad_seq(seq=[self.bos_index] + encoder_seq,
                                   max_len=self.max_len_source)

        decoder_seq = self.pad_seq(seq=[self.bos_index] + decoder_seq,
                                   max_len=self.max_len_target)

        target_seq = self.pad_seq(seq=target_seq + [self.eos_index],
                                  max_len=self.max_len_target)

        encoder_seq = Tensor(encoder_seq).long()
        decoder_seq = Tensor(decoder_seq).long()
        target_seq = Tensor(target_seq).long()

        return encoder_seq, decoder_seq, target_seq


def shuffle_sentences(source: List[str], target: List[str]) -> Tuple[List[str], List[str]]:
    """
    Shuffles lists of source and target sentences simultaneously.
    :param source: list of source sentences
    :param target: list of target sentences
    :return: shuffled lists of source and target sequences (source -> translation pairs are kept)
    """

    assert len(source) == len(target), '`source` and `target` must have the same number of elements!'

    pairs = list(zip(source, target))
    random.shuffle(pairs)
    source, target = zip(*pairs)
    return source, target


def get_loaders(source: List[List[int]],
                target: List[List[int]],
                train_size: float = 0.9,
                batch_size: int = 256,
                **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and valid DataLoaders from given lists of tokenized source and target sentences.
    :param source: lists of tokens in source language
    :param target: lists of tokens in target language
    :param train_size: fraction of the data to use for training
    :param batch_size: number of elements in a single batch
    :return: DataLoaders for source and target sentences
    """
    assert len(source) == len(target), '`source` and `target` must have the same number of elements!'

    validation_start_index = int(len(source) * train_size)

    train_ds = MTData(source[:validation_start_index],
                      target[:validation_start_index],
                      **kwargs)

    valid_ds = MTData(source[validation_start_index:],
                      target[validation_start_index:],
                      **kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)

    return train_loader, valid_loader
