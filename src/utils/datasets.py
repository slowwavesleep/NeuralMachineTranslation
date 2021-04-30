from typing import List, Tuple

from torch.utils.data import Dataset
from torch import Tensor


class MTData(Dataset):
    """
    Dataset class for machine translation data.
    Pads or truncates sentences to specified length.
    Adds <bos> tokens to encoder and decoder sentences.
    Adds <eos> tokens to target sentences.
    """

    def __init__(self,
                 source: List[List[int]],
                 target: List[List[int]],
                 max_len_source: int = 32,
                 max_len_target: int = 32,
                 pad_index: int = 0,
                 bos_index: int = 2,
                 eos_index: int = 3):
        """
        :param source: a list of sentences to translate presented as lists of indices
        :param target: a list of reference translated sentenced presented as lists of indices
        :param max_len_source: resulting length of all encoder sequences
        :param max_len_target: resulting length of all decoder and target sequences
        :param pad_index: index of <pad> token
        :param bos_index: index of <bos> token
        :param eos_index: index of <eos> token
        """

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
        """
        Returns a tuple containing tensors for encoder, decoder, and target sequences respectively.
        """
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
