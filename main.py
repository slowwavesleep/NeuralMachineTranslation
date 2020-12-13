import youtokentome as yttm
import torch
from src.utils.data import load_ted, load_anki, clean_anki, basic_load, basic_gzip_load
from src.utils.tokenization import train_bpe, batch_tokenize
from src.utils.loaders import get_loaders
from src.nn.models import BasicEncoderDecoder
from src.nn.training import training_cycle
from src.nn.translation import Translator

# file paths
SOURCE_TRAIN_PATH = "data/rus.txt"
SOURCE_DEV_PATH = "data/fra-rus/dev.src"
TARGET_TRAIN_PATH = "data/fra-rus/train.trg.gz"
TARGET_DEV_PATH = "data/fra-rus/dev.trg"
BPE_TEXT_PATH = "tmp/bpe_text.tmp"
SOURCE_BPE_PATH = "models/source_bpe.model"
TARGET_BPE_PATH = "models/target_bpe.model"
TRANSLATIONS_PATH = "results/translations.txt"

# model parameters
VOCAB_SIZE = 7000
PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3
SOURCE_MAX_LEN = 16
TARGET_MAX_LEN = 18

# miscellaneous
TRAIN_BPE = False
TRAIN_NET = False

# source_sentences = basic_gzip_load(SOURCE_TRAIN_PATH)
# target_sentences = basic_gzip_load(TARGET_TRAIN_PATH)

source_sentences, target_sentences = load_anki(SOURCE_TRAIN_PATH)


if TRAIN_BPE:
    train_bpe(source_sentences, BPE_TEXT_PATH, SOURCE_BPE_PATH, VOCAB_SIZE)
    train_bpe(target_sentences, BPE_TEXT_PATH, TARGET_BPE_PATH, VOCAB_SIZE)

source_bpe = yttm.BPE(model=SOURCE_BPE_PATH)
target_bpe = yttm.BPE(model=TARGET_BPE_PATH)


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
                            emb_dim=256,
                            model_dim=256,
                            model_layers=2,
                            model_dropout=0.3,
                            padding_index=PAD_INDEX)

model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
optimizer = torch.optim.Adam(params=model.parameters())

if TRAIN_NET:
    training_cycle(model, train_loader, valid_loader, optimizer, criterion, device, 5)

model.load_state_dict(torch.load('models/best_language_model_state_dict.pth'))

translator = Translator(source_bpe, target_bpe, model, device)
translator.to_file(source_sentences[-1000:], target_sentences[-1000:], TRANSLATIONS_PATH)
