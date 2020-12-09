import torch
from torch.nn import Module
import youtokentome as yttm


def greedy_translate(source: str,
                     source_tokenizer: yttm.BPE,
                     target_tokenizer: yttm.BPE,
                     model: Module,
                     device: object,
                     bos_index: int = 2,
                     eos_index: int = 3,
                     max_sequence: int = 32):

    # if max_sequence > MAX_LEN:
    #     raise ValueError

    tokenized = source_tokenizer.encode(source, eos=False, bos=True)

    encoder_sequence = torch.tensor([tokenized]).long().to(device)
    decoder_sequence = torch.tensor([bos_index]).long().unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        for timestamp in range(max_sequence):
            predictions = model(encoder_sequence, decoder_sequence)
            current_token = predictions[:, -1, :].argmax(dim=-1)
            if current_token == eos_index:
                break
            decoder_sequence = torch.cat([decoder_sequence, current_token.unsqueeze(0)], dim=-1)

    translation = target_tokenizer.decode(decoder_sequence.squeeze(0).tolist())
    translation = translation[0].lstrip('<BOS> ').rstrip('<EOS>')
    return translation
