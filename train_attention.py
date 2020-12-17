import youtokentome as yttm
import torch
from torch.utils.data import DataLoader
from src.utils.data import basic_load, basic_gzip_load
from src.utils.tokenization import train_bpe, batch_tokenize
from src.utils.datasets import MTData
from src.nn.models import LstmAttentionModel
from src.nn.training import training_cycle
from src.nn.translation import Translator


# data paths
# SOURCE_TRAIN_PATH = "data/eng-rus/source.train"
# SOURCE_DEV_PATH = "data/eng-rus/source.dev"
# TARGET_TRAIN_PATH = "data/eng-rus/target.train"
# TARGET_DEV_PATH = "data/eng-rus/target.dev"

SOURCE_TRAIN_PATH = "data/rus-ukr/train.src.gz"
SOURCE_DEV_PATH = "data/rus-ukr/dev.src"
TARGET_TRAIN_PATH = "data/rus-ukr/train.trg.gz"
TARGET_DEV_PATH = "data/rus-ukr/dev.trg"
SOURCE_TEST_PATH = "data/rus-ukr/test.src"
TARGET_TEST_PATH = "data/rus-ukr/test.trg"
TRANSLATIONS_PATH = "results/main/translations.txt"

# models paths
BPE_TEXT_PATH = "tmp/bpe_text.tmp"
SOURCE_BPE_PATH = "models/main/source_bpe.model"
TARGET_BPE_PATH = "models/main/target_bpe.model"

# model parameters
VOCAB_SIZE = 7000
PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3
SOURCE_MAX_LEN = 20
TARGET_MAX_LEN = 20

# flow control
TRAIN_BPE = True
TRAIN_NET = True
TRANSLATE_TEST = True


source_train = basic_gzip_load(SOURCE_TRAIN_PATH)
target_train = basic_gzip_load(TARGET_TRAIN_PATH)
source_dev = basic_load(SOURCE_DEV_PATH)
target_dev = basic_load(TARGET_DEV_PATH)


if TRAIN_BPE:
    train_bpe(source_train, BPE_TEXT_PATH, SOURCE_BPE_PATH, VOCAB_SIZE)
    train_bpe(target_train, BPE_TEXT_PATH, TARGET_BPE_PATH, VOCAB_SIZE)

source_bpe = yttm.BPE(model=SOURCE_BPE_PATH)
target_bpe = yttm.BPE(model=TARGET_BPE_PATH)

# <BOS> and <EOS> tags are added by dataset class
source_train_tokenized = batch_tokenize(source_train, source_bpe, bos=False, eos=False)
source_dev_tokenized = batch_tokenize(source_dev, source_bpe, bos=False, eos=False)
target_train_tokenized = batch_tokenize(target_train, target_bpe, bos=False, eos=False)
target_dev_tokenized = batch_tokenize(target_dev, target_bpe, bos=False, eos=False)

train_ds = MTData(source_train_tokenized, target_train_tokenized, SOURCE_MAX_LEN, TARGET_MAX_LEN)
valid_ds = MTData(source_train_tokenized, target_train_tokenized, SOURCE_MAX_LEN, TARGET_MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=512)

GPU = torch.cuda.is_available()

if GPU:
    print('Using GPU...')
    device = torch.device('cuda')
else:
    print('Using CPU...')
    device = torch.device('cpu')


model = LstmAttentionModel(vocab_size=VOCAB_SIZE,
                           emb_dim=300,
                           hidden_size=1000,
                           layer_dropout=0.3,
                           lstm_layers=2,
                           bidirectional=False,
                           spatial_dropout=0.3,
                           padding_index=PAD_INDEX)

model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
optimizer = torch.optim.Adam(params=model.parameters())

if TRAIN_NET:

    training_cycle(model, train_loader, valid_loader, optimizer, criterion, device, 1)

if TRANSLATE_TEST:

    source_test = basic_load(SOURCE_TEST_PATH)
    target_test = basic_load(TARGET_TEST_PATH)

    model.load_state_dict(torch.load('models/best_language_model_state_dict.pth'))

    translator = Translator(source_bpe, target_bpe, model, device)
    translator.to_file(source_test, target_test, TRANSLATIONS_PATH)
