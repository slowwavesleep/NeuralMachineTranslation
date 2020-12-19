from yaml import safe_load
import youtokentome as yttm
import torch
from torch.utils.data import DataLoader
from src.utils.data import basic_load, smart_load
from src.utils.tokenization import train_bpe, batch_tokenize
from src.utils.datasets import MTData
from src.nn.training import training_cycle
from src.nn.translation import Translator
import argparse

parser = argparse.ArgumentParser(description='Run model with specified settings.')

parser.add_argument(dest='config', type=str, help='Path to config file.')
args = parser.parse_args()

with open(args.config) as file:
    config = safe_load(file)

data_paths = config['data_paths']
models_paths = config['models_paths']
parameters = config['parameters']
flow = config['flow_control']
net_params = config['net_parameters']

source_train = smart_load(data_paths['source_train_path'], max_lines=parameters['max_lines_train'])
target_train = smart_load(data_paths['target_train_path'], max_lines=parameters['max_lines_train'])
source_dev = smart_load(data_paths['source_dev_path'])
target_dev = smart_load(data_paths['target_dev_path'])

assert len(source_train) == len(target_train)
assert len(source_dev) == len(target_dev)


if flow['train_bpe']:

    train_bpe(sentences=source_train,
              bpe_text_path=models_paths['bpe_text_path'],
              bpe_model_path=models_paths['source_bpe_path'],
              vocab_size=parameters['vocab_size'])

    train_bpe(sentences=target_train,
              bpe_text_path=models_paths['bpe_text_path'],
              bpe_model_path=models_paths['target_bpe_path'],
              vocab_size=parameters['vocab_size'])

source_bpe = yttm.BPE(model=models_paths['source_bpe_path'])
target_bpe = yttm.BPE(model=models_paths['target_bpe_path'])

# <BOS> and <EOS> tags are added by dataset class
source_train_tokenized = batch_tokenize(source_train, source_bpe, bos=False, eos=False)
source_dev_tokenized = batch_tokenize(source_dev, source_bpe, bos=False, eos=False)
target_train_tokenized = batch_tokenize(target_train, target_bpe, bos=False, eos=False)
target_dev_tokenized = batch_tokenize(target_dev, target_bpe, bos=False, eos=False)


train_ds = MTData(source_train_tokenized,
                  target_train_tokenized,
                  parameters['source_max_len'],
                  parameters['target_max_len'])

valid_ds = MTData(source_train_tokenized,
                  target_train_tokenized,
                  parameters['source_max_len'],
                  parameters['target_max_len'])

train_loader = DataLoader(train_ds, batch_size=parameters['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=parameters['batch_size'])

GPU = torch.cuda.is_available()

if GPU:
    print('Using GPU...')
    device = torch.device('cuda')
else:
    print('Using CPU...')
    device = torch.device('cpu')

if config['model'] == 'baseline':

    from src.nn.models import BaselineModel

    model = BaselineModel(vocab_size=parameters['vocab_size'],
                          padding_index=parameters['pad_index'],
                          **net_params)

elif config['model'] == 'main':

    from src.nn.models import LstmAttentionModel

    model = LstmAttentionModel(vocab_size=parameters['vocab_size'],
                               padding_index=parameters['pad_index'],
                               **net_params)

else:

    raise NotImplementedError

model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=parameters['pad_index'])
optimizer = torch.optim.Adam(params=model.parameters())


if flow['train_net']:

    training_cycle(model, train_loader, valid_loader, optimizer, criterion, device, parameters['num_epochs'])


if flow['translate_test']:

    source_test = basic_load(data_paths['source_test_path'])
    target_test = basic_load(data_paths['target_test_path'])

    model.load_state_dict(torch.load('models/best_language_model_state_dict.pth'))

    translator = Translator(source_bpe, target_bpe, model, device)
    translator.to_file(source_test, target_test, data_paths['translations_path'])

