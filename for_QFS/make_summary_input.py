import nltk
from nltk.tokenize import sent_tokenize
import json
import pandas as pd
import glob
import argparse

nltk.download('punkt')
argparser = argparse.ArgumentParser()
argparser.add_argument('--fname', type=str, required=True, help='fname')
argparser.add_argument('--write', type=str, required=True, help='write fname')
args = argparser.parse_args()

def split_qp(data):
    q = [i.split(' <SEP> ')[0] for i in data]
    p = [i.split(' <SEP> ')[1] for i in data]
    return q, p

def write_jsonl(fname, idx):
    with open('{}.jsonl'.format(fname), 'w', encoding='utf-8') as idx:
        for i in range(len(question)):
            dic = {"source":passage[i],"query":question[i], "target":'None'}
            idx.write(json.dumps(dic) + "\n")

# file_open
with open(args.fname, 'r') as f:
    f = f.read().splitlines()
    split = [i.strip() for i in f]

# split q/p
question, passage = split_qp(split)
write_jsonl('{}'.format(args.write), 'w')