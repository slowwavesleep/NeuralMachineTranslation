import youtokentome as yttm
import torch
from src.utils.ted import load_ted
from src.utils.tokenization import train_bpe, batch_tokenize
from src.utils.loaders import get_loaders
from src.nn.models import BasicEncoderDecoder
from src.nn.training import train, evaluate
from src.utils.loaders import shuffle_sentences
from src.nn.translation import greedy_translate


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

model = BasicEncoderDecoder(vocab_size=VOCAB_SIZE,
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

for sentence in source_sentences[:10]:
    print(sentence)
    print(greedy_translate(sentence, source_bpe, target_bpe, model, device))