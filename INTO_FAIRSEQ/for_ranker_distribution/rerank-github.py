from typing import List
import os
import json
import logging

import numpy as np
import pandas as pd
import sacrebleu
from torchmetrics.text.rouge import ROUGEScore
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
ROUGE = ROUGEScore()


def score(hyp: List[str], ref: str, type='rouge'):
    output = []
    for h in hyp:
        if type == 'bleu':
            tmp_bleu = sacrebleu.sentence_bleu(h, [ref]).score
            output.append(tmp_bleu / 100.0)
        elif type == 'rouge':
            tmp_rouge = ROUGE(h, ref)['rougeL_fmeasure'].tolist()
            output.append(tmp_rouge)
        else:
            raise Exception('type definition error')
    return max(output)


def self_score(hyp: List[str]):
    self_ref = [[i for i in hyp[:hyp.index(h)] + hyp[hyp.index(h)+1:]] for h in hyp]
    self_bleu, self_rouge = [], []

    for h, r in zip(hyp, self_ref):
        tmp_bleu = [sacrebleu.sentence_bleu(h, [r_each]).score for r_each in r]
        tmp_rouge = [ROUGE(h, r_each)['rougeL_fmeasure'].tolist() for r_each in r]
        self_bleu.append(max(tmp_bleu) / 100.0)
        self_rouge.append(max(tmp_rouge))

    return self_bleu, self_rouge


def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    output = {}
    for key in obj:
        tmp = obj[key]
        df = []
        for (q, a), score in tmp:
            df.append([q, a, score])
        df = pd.DataFrame(df, columns=['q', 'a', 'score'])
        df['a_lem'] = df['a'].apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in x.split(' ') if i not in stopwords]))
        df['q_lem'] = df['q'].apply(lambda x: ' '.join([lemmatizer.lemmatize(i) for i in x.split(' ') if i not in stopwords]))
        print(f'original len: {len(df)}, '
              f'a_trimmed_len: {len(df["a_lem"].drop_duplicates())}, '
              f'q_trimmed_len: {len(df["q_lem"].drop_duplicates())}')
        output[key] = df

    return output


def find_scale(hyp: List[str], ref, type='rouge'):
    if not hyp:
        return 0
    else:
        return score(hyp, ref, type)


def rank_by_EM(df: pd.DataFrame, column_name='a_lem', return_df=False):
    df.sort_values(by='score', ascending=False, inplace=True)

    output = []
    key = df[column_name].drop_duplicates().tolist()
    if not return_df:
        for k in key:
            tmp = df[df[column_name] == k].iloc[0]
            output.append(((tmp['q'], tmp['a']), tmp['score']))
        return output
    else:
        for k in key:
            tmp = df[df[column_name] == k].iloc[0]
            output.append(tmp.name)
        output = df.iloc[output]
        return output


def rank_by_overlap(original: pd.DataFrame, column_name='a_lem', type='rouge'):
    df = original.sort_values(by='score', ascending=False)

    negatives = df[df['score'] < 0]
    df = df.drop(negatives.index)

    output = []
    overlap_check = []
    dropped = []

    while len(df) > 1:
        #print(len(df))

        if column_name == 'qa_lem':
            df['score_now'] = df.apply(
                lambda x: x['score'] -
                          find_scale(hyp=overlap_check, ref=x['a_lem'], type=type) *
                          find_scale(hyp=overlap_check, ref=x['q_lem'], type=type) *
                          abs(x['score']),
                axis=1
            )
        else:
            df['score_now'] = df.apply(
                lambda x: x['score'] - find_scale(hyp=overlap_check, ref=x[column_name], type=type) * abs(x['score']),
                axis=1
            )

        tmp = df.iloc[df['score_now'].argmax()].copy()
        df.drop(tmp.name, inplace=True) 
        drop_index = df[df['score_now'] == 0].index.tolist()
        if drop_index:
            dropped.extend(drop_index) 
            df.drop(drop_index, inplace=True)

        if column_name == 'qa_lem':
            overlap_check.append(tmp['q_lem'] + '/' + tmp['a_lem'])
        else:
            overlap_check.append(tmp[column_name])
        output.append(((tmp['q'], tmp['a']), tmp['score']))

    df = original.sort_values(by='score', ascending=False)
    if dropped:
        print('\ndropped: ', dropped)
        print('len df:', len(df))
        print(df)
        dropped = df.loc[dropped]
        for tmp in dropped.iloc:
            output.append(((tmp['q'], tmp['a']), tmp['score']))

    if negatives.index.tolist():
        for tmp in negatives.iloc:
            output.append(((tmp['q'], tmp['a']), tmp['score']))
    return output


def rerank(obj: dict):
    output = {}
    for key in obj:
        tmp = obj[key]
        tmp = rank_by_EM(tmp, column_name='a_lem', return_df=True)
        output[key] = rank_by_overlap(tmp, column_name='a_lem', type='rouge')
    return output


def run(source, target):
    def write_json(obj, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    os.makedirs(target, exist_ok=True)
    print(f'\n\n ############ now processing: {source} ############ \n\n')
    for f in os.listdir(source):
        print('\n filename: ', f)
        if not os.path.exists(os.path.join(target, f)):
            data = read_json(os.path.join(source, f))
            data = rerank(data)
            write_json(data, os.path.join(target, f))


if __name__ == "__main__":
    print(':)')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        required=True,
                        help='data source dir')
    parser.add_argument('--target',
                        default='target',
                        help='data target dir')

    args = parser.parse_args()

    run(args.source, args.target)

