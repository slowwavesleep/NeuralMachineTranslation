import torch
from torch.nn import Module
import youtokentome as yttm
from typing import List, NoReturn
from tqdm import tqdm


class Translator:

    def __init__(self,
                 source_tokenizer: yttm.BPE,
                 target_tokenizer: yttm.BPE,
                 model: Module,
                 device: object,
                 bos_index: int = 2,
                 eos_index: int = 3,
                 max_sequence: int = 32):

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.model = model
        self.device = device
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.max_sequence = max_sequence

    def translate(self, source: str) -> str:
        """
        Translate one given sentence into target language.
        :param source: sentence to translate
        """

        # if max_sequence > MAX_LEN:
        #     raise ValueError

        tokenized = self.source_tokenizer.encode(source, eos=False, bos=True)

        encoder_sequence = torch.tensor([tokenized]).long().to(self.device)
        decoder_sequence = torch.tensor([self.bos_index]).long().unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for timestamp in range(self.max_sequence):
                predictions = self.model(encoder_sequence, decoder_sequence)
                current_token = predictions[:, -1, :].argmax(dim=-1)
                if current_token == self.eos_index:
                    break
                decoder_sequence = torch.cat([decoder_sequence, current_token.unsqueeze(0)], dim=-1)

        translation = self.target_tokenizer.decode(decoder_sequence.squeeze(0).tolist())
        translation = translation[0].lstrip('<BOS> ').rstrip('<EOS>')

        return translation

    def to_file(self,
                source_sentences: List[str],
                target_sentences: List[str],
                file_path: str) -> NoReturn:
        """
        Translate given sentences in source language and write results to file in the following format:
        <source sentence>
        <target sentence>
        <translation by the model>
        :param source_sentences: sentences to translate
        :param target_sentences: reference sentences in target language
        :param file_path: path to resulting file
        """


        # TODO CLEAN nextline when loading

        with open(file_path, 'w') as file:
            for source, target in tqdm(zip(source_sentences, target_sentences),
                                       total=len(source_sentences),
                                       desc='Translating sentences...'):

                translation = self.translate(source)
                file.write(source)# + '\n')
                file.write(target)# + '\n')
                file.write(translation + '\n')
                file.write('\n')
