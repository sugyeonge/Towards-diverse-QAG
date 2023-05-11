from functools import reduce
from functools import partial
from itertools import product
from pathlib import Path
from random import sample
import argparse
import os
import sys

import numpy as np
from torch.utils.data.dataset import T_co
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import trange, tqdm
from setproctitle import setproctitle


# hyperparams
model_name = [
    'bert-base-uncased',
    'roberta-base'
][1]
lr = 5e-07
only_one_negative = False
ckpt_name = 'model_ckpt'

class ContrastPQA(TensorDataset):
    def __init__(self, hf_fairytale_qa, tokenizer):
        super(ContrastPQA, self).__init__()
        self.tokenizer = tokenizer
        def pair_by_content(acc, cur):
            c = cur['content']
            if c not in acc:
                acc[c] = []
            acc[c].append((cur['question'], cur['answer']))
            return acc

        d = reduce(pair_by_content, hf_fairytale_qa, {})
        self.examples = []
        negative_dummy_answer = None
        for content in d:
            questions, answers = zip(*d[content])
            if only_one_negative:
                for i in range(len(questions)):
                    neg_indexes = list(range(len(answers)))
                    if len(neg_indexes) == 1:
                        neg_answer = negative_dummy_answer
                    else:
                        neg_indexes.remove(i)
                        neg_index = sample(neg_indexes, 1)[0]  # randomly sample one
                        neg_answer = answers[neg_index]
                    self.examples.append((content, questions[i], answers[i], 1))  # (p, q, a, label)
                    if neg_answer is None:
                        raise AssertionError
                    self.examples.append((content, questions[i], neg_answer, 0))  # (p, q, a, label)
                    negative_dummy_answer = answers[i]
            else:
                for i, j in product(range(len(questions)), repeat=2):
                    self.examples.append((content, questions[i], answers[j], int(i == j)))  # (p, q, a, label)

    def __getitem__(self, index) -> T_co:
        return make_feature(self.tokenizer, self.examples[index])
        # return self.examples[index]

    def __len__(self):
        return len(self.examples)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file',
                        default=None,
                        type=str,)
    parser.add_argument('--onlyinfer',
                        action='store_true',)
    return parser.parse_args()

def make_feature(tokenizer, example, use_context=True, fixed_token_len=False):
    if fixed_token_len:
        ...
    elif use_context:  # use (P, Q, A) else (Q, A)
        sep_token = tokenizer.sep_token
        s = f' {sep_token} '.join(example[:-1])
        # return tokenizer(s, return_tensors='pt', padding='max_length'), example[-1]
        return tokenizer(
            s,
            return_tensors='pt',
            padding='max_length',  # TODO debug code
            max_length=512,
            truncation=True
        ), example[-1]
    else:
        q, a, label = example[1:]
        return tokenizer(
            q, a, return_tensors='pt', padding='max_length'
        ), label

def save(model, optimizer, cur_epoch):
    model_lr = str(model_name).split('_')[0] + '_' + str(lr)
    # path = os.path.join(ckpt_name, model_lr)
    path = Path(ckpt_name) / model_lr
    if not path.exists():
        os.mkdir(path)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, Path(ckpt_name) / model_lr / (str(cur_epoch) + ".pth"))

def load(model: torch.nn.Module, optimizer:None, model_file):
    path = Path(model_file)
    if not path.exists():
        raise FileNotFoundError
    with open(path) as f:
        save_dict = torch.load(path)
    model.load_state_dict(save_dict['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

def append_result_(text, model_lr):
    path = Path(ckpt_name) / model_lr
    if not path.exists():
        path.mkdir(parents=True)
    with open(path / 'result.txt', 'at') as f:
        f.write(str(text) + '\n')
    print(text)

def validation(model, dataloader, **kwargs):
    device = kwargs['device']
    # Validation
    hits = []
    val_steps = 0
    val_loss = 0
    model.eval()
    for step, batch in enumerate(tqdm(dataloader)):
        input_dict = batch[0].to(device)
        for key in input_dict:
            input_dict[key] = input_dict[key].squeeze(0)
        label = batch[-1].clone().detach().to(device)
        with torch.no_grad():
            eval_output = model(**input_dict)
        logits = eval_output.logits
        val_loss += CrossEntropyLoss()(logits, label)
        val_steps += 1

        # accumulate accuracy
        logits = logits.view(-1, 2).detach().cpu().numpy()
        label = label.view(-1).cpu().numpy()
        pred = np.argmax(logits, axis=1)
        hits.append(pred == label)
        if step < 10:
            print(logits)
            print('label: ', label)
            print('pred: ', pred)
            print()
    val_loss = val_loss / val_steps
    val_accuracy = float(sum(hits)) / len(hits)
    return val_loss, val_accuracy

def only_inference(model, valid_dataloader, **kwargs):
    print()
    print('---------------------------')
    print('--- Inference Only Mode ---')
    print('---------------------------')
    print()
    val_loss, val_acc = validation(model, valid_dataloader, device=kwargs['device'])
    print('validation loss: {:.4f}'.format(val_loss))
    print('validation accuracy: {:.4f}\n'.format(val_acc))
    sys.exit()  # end script

def main():
    args = parse_args()

    model_lr = str(model_name).split('_')[0] + '_' + str(lr)
    # cur_time = datetime.now().strftime('%d-%H-%M-%S')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-08
    )

    if args.model_file:
        load(model, optim, args.model_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')

    ds = load_dataset("GEM/FairytaleQA")
    train, valid, test = ds['train'], ds['validation'], ds['test']

    # concatenate train dataset with test dataset since we have other test dataset
    train = concatenate_datasets([train ,test])

    contrast_ds = ContrastPQA(train, tokenizer)
    train_dataloader = DataLoader(contrast_ds, shuffle=True)
    valid_dataloader = DataLoader(ContrastPQA(valid, tokenizer), shuffle=True)

    if args.onlyinfer:
        only_inference(model, valid_dataloader, device=device)

    append_result = partial(append_result_, model_lr=model_lr)
    append_result('model: {}, lr: {}'.format(model_name, lr))

    epochs = 20

    for epoch in trange(epochs, desc='Epoch'):
        # Train
        model.train()

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader)):
            input_dict = batch[0].to(device)
            for key in input_dict:
                input_dict[key] = input_dict[key].squeeze(0)
            label = batch[-1].clone().detach().to(device)

            optim.zero_grad()
            train_output = model(**input_dict, labels=label)
            train_output.loss.backward()
            optim.step()

            tr_loss += train_output.loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

        val_loss, val_accuracy = validation(model, valid_dataloader, device=device)

        append_result('epoch: {}'.format(epoch))
        append_result('train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        append_result('validation loss: {:.4f}'.format(val_loss))
        append_result('validation accuracy: {:.4f}\n'.format(val_accuracy))
        save(model, optim, epoch)
        print('model saved')

if __name__ == '__main__':
    setproctitle("training ranker")
    main()



















