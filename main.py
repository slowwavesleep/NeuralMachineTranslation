import youtokentome as yttm
import torch
from src.utils.ted import load_ted
from src.utils.tokenization import train_bpe, batch_tokenize
from src.utils.loaders import get_loaders
from src.nn.models import EncoderDecoder
from src.nn.training import train, evaluate
from src.utils.loaders import shuffle_sentences


SOURCE_DATA_PATH = "data/ru_ted.txt"
TARGET_DATA_PATH = "data/fr_ted.txt"
BPE_TEXT_PATH = "tmp/bpe_text.tmp"
SOURCE_BPE_PATH = "models/source_bpe.model"
TARGET_BPE_PATH = "models/target_bpe.model"
VOCAB_SIZE = 7000
PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3
SOURCE_MAX_LEN = 16
TARGET_MAX_LEN = 18
RETRAIN_BPE = False

source_sentences = load_ted(SOURCE_DATA_PATH)
target_sentences = load_ted(TARGET_DATA_PATH)

if RETRAIN_BPE:
    train_bpe(source_sentences, BPE_TEXT_PATH, SOURCE_BPE_PATH, VOCAB_SIZE)
    train_bpe(target_sentences, BPE_TEXT_PATH, TARGET_BPE_PATH, VOCAB_SIZE)

source_bpe = yttm.BPE(model=SOURCE_BPE_PATH)
target_bpe = yttm.BPE(model=TARGET_BPE_PATH)

source_sentences, target_sentences = shuffle_sentences(source_sentences, target_sentences)

source_tokenized = batch_tokenize(source_sentences, source_bpe, bos=False, eos=False)
target_tokenized = batch_tokenize(target_sentences, target_bpe, bos=False, eos=False)

train_loader, valid_loader = get_loaders(source=source_tokenized,
                                         target=target_tokenized,
                                         batch_size=256,
                                         train_size=0.9,
                                         max_len_source=SOURCE_MAX_LEN,
                                         max_len_target=TARGET_MAX_LEN)

GPU = torch.cuda.is_available()

if GPU:
    print('Using GPU...')
    device = torch.device('cuda')
else:
    print('Using CPU...')
    device = torch.device('cpu')

model = EncoderDecoder(vocab_size=VOCAB_SIZE,
                       emb_dim=100,
                       model_dim=100,
                       model_layers=1,
                       model_dropout=0.3,
                       padding_index=PAD_INDEX)

model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
optimizer = torch.optim.Adam(params=model.parameters())

train(model, train_loader, criterion, optimizer, device)
# evaluate(model, valid_loader, criterion, device)


def greedy_translate(source,
                     source_tokenizer,
                     target_tokenizer,
                     bos_index=2,
                     eos_index=3,
                     max_sequence=32):

    # if max_sequence > MAX_LEN:
    #     raise ValueError

    tokenized = source_tokenizer.encode(source, eos=True, bos=True)

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

    answer = target_tokenizer.decode(decoder_sequence.squeeze(0).tolist())
    answer = answer[0].lstrip('<BOS> ').rstrip('<EOS>')
    return answer


print(source_sentences[0])
print(greedy_translate(source_sentences[0], source_bpe, target_bpe))
print(source_sentences[1])
print(greedy_translate(source_sentences[1], source_bpe, target_bpe))
print(source_sentences[2])
print(greedy_translate(source_sentences[2], source_bpe, target_bpe))
print(source_sentences[3])
print(greedy_translate(source_sentences[3], source_bpe, target_bpe))
