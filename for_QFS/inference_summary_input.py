import os
import nltk
from nltk.tokenize import sent_tokenize
import json
import pandas as pd
import glob
import argparse

nltk.download('punkt')
argparser = argparse.ArgumentParser()
argparser.add_argument('--file', type=str, required=True, help='fname')
#argparser.add_argument('--write', type=str, required=True, help='write folder')
args = argparser.parse_args()

def write_jsonl(fpath, fname, idx):
    with open(os.path.join(fpath, fname+'.jsonl'), 'w', encoding='utf-8') as idx:
        for i in range(len(sentence)):
            dic = {"source":passages[i],"query":sentence[i], "target":target[i]}
            idx.write(json.dumps(dic) + "\n")

with open(args.file, 'r') as f:
    book = f.read().splitlines()
    book = [i.strip() for i in book]

passages, sentence = [], []
for passage in book:
    tok_sent = sent_tokenize(passage)
    for sent in tok_sent:
        passages.append(passage.strip())
        sentence.append(sent.strip())

target = ['None' for i in range(len(sentence))]
assert len(passages) == len(sentence) == len(target)
fpath = '/'.join(args.file.split('/')[:-1])
fname = args.file.split('/')[-1]
fname = fname.split('.')[0]
write_jsonl(fpath,fname, 'w1')

with open(os.path.join(fpath, fname)+'.passage', 'w') as w:
    for i in passages:
        w.write(i)
        w.write('\n')
