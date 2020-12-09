import numpy as np
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


def train(model: Module,
          loader: DataLoader,
          criterion: Module,
          optimizer: Optimizer,
          device: object,
          last_n_losses: int = 500,
          verbose: bool = True):
    losses = []

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc='Train')

    model.train()

    for encoder_seq, decoder_seq, target_seq in loader:

        encoder_seq = encoder_seq.to(device)
        decoder_seq = decoder_seq.to(device)
        target_seq = target_seq.to(device)

        pred = model(encoder_seq, decoder_seq)

        loss = criterion(pred.view(-1, pred.size(-1)), target_seq.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                 perplexity=np.exp(np.mean(losses[-last_n_losses:])))

        progress_bar.update()

    progress_bar.close()

    return losses


def evaluate(model: Module,
             loader: DataLoader,
             criterion: Module,
             device: object,
             last_n_losses: int = 500,
             verbose: bool = True):

    losses = []

    progress_bar = tqdm(total=len(loader), disable=not verbose, desc='Evaluate')

    model.eval()

    for encoder_seq, decoder_seq, target_seq in loader:

        encoder_seq = encoder_seq.to(device)
        decoder_seq = decoder_seq.to(device)
        target_seq = target_seq.to(device)

        with torch.no_grad():
            pred = model(encoder_seq, decoder_seq)

        loss = criterion(pred.view(-1, pred.size(-1)), target_seq.view(-1))

        losses.append(loss.item())

        progress_bar.set_postfix(loss=np.mean(losses[-last_n_losses:]),
                                 perplexity=np.exp(np.mean(losses[-last_n_losses:])))

        progress_bar.update()

    progress_bar.close()

    return losses
