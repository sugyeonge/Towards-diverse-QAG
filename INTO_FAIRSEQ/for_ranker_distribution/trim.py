import glob
import json
import argparse
import os.path
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--source', required=True, type=str)
argparser.add_argument('--topk', default=5, type=str)
args = argparser.parse_args()

def file_open(fname, idx):
    with open(fname, 'r') as idx:
        idx = idx.read().splitlines()
    return idx

def open_json(fname, idx):
    with open(fname, 'r') as idx:
        file = json.load(idx)
    return file

for i in os.listdir(args.source):
    story = open_json(os.path.join(args.source,i), 'f')
    fname = i

if not os.path.exists('inference_result'):
    os.makedirs('inference_result')
    
with open('inference_result/{}.{}'.format(fname, args.topk), 'w') as w:
    for idx, (passage, pairs) in enumerate(story.items()):
        pairs = pairs[:int(args.topk)]
        for i in pairs:
            question = i[0][0].strip()
            answer = i[0][1].strip()
            merge = answer + ' </s> ' + passage.strip() + ' </s> ' + question
            w.write(merge)
            w.write('\n')
