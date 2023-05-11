import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--passage', type=str, required=True)
argparser.add_argument('--question', type=str, required=True)
argparser.add_argument('--write', type=str, required=True)
arg = argparser.parse_args()

def concat(pas, que):
    c = []
    for i, j in zip(pas, que):
        low_i = i.lower()
        low_j = j.lower()
        c.append(low_i + ' </s> ' + low_j)
    return c

with open(arg.passage, 'r') as f, open(arg.question, 'r') as f2:
   passage = f.read().splitlines()
   question = f2.read().splitlines()

data = concat(passage, question)

with open(arg.write, 'w') as w:
    for i in data:
        w.write(i)
        w.write('\n')
