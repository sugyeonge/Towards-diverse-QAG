import os
import argparse
import spacy

nlp = spacy.load('en_core_web_sm')
argparser = argparse.ArgumentParser()
argparser.add_argument('--file', required=True, type=str)
arg = argparser.parse_args()

fpath = '/'.join(arg.file.split('/')[:-1])
fname = arg.file.split('/')[-1]
fname = fname.split('.')[0]

with open(arg.file, 'r') as f, open(os.path.join(fpath, fname+'.write'), 'w') as w:
    f = f.read().splitlines()
    for i in f:
        prcs = nlp(i)
        prcs = [tok.text for tok in prcs]
        prcsd = ' '.join(prcs)
        w.write(prcsd.lower().strip())
        w.write('\n')

    
    
