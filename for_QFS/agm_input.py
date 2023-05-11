# agm input
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--pq_file', type=str, required=True)
argparser.add_argument('--summary', type=str, required=True)
argparser.add_argument('--write', type=str, required=True)
arg = argparser.parse_args()

def concat(pas, que):
    c = []
    for i, j in zip(pas, que):
        low_i = i.lower()
        low_j = j.lower()
        c.append(low_i + ' </s> ' + low_j)
    return c

with open(arg.pq_file, 'r') as f, open(arg.summary, 'r') as f2:
   qp = f.read().splitlines()
   p = [i.split(' <SEP> ')[1] for i in qp]
   summary = f2.read().splitlines()

data = concat(p, summary)

with open(arg.write, 'w') as w:
    for i in data:
        w.write(i)
        w.write('\n')
