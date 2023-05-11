import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--pa_file', type=str, required=True)
argparser.add_argument('--q_file', type=str, required=True)
argparser.add_argument('--write', type=str, required=True)
args = argparser.parse_args()

def append_wh(pa, q):
    new_pass = []
    for passage, question in zip(pa, q):
        passages = passage.split(' <SEP> ')
        new_p = passages[1] + ' </s> ' + passages[0] + '</s> ' + question.split()[0]
        new_pass.append(new_p.strip())
    return new_pass

with open(args.pa_file, 'r') as pa, open(args.q_file, 'r') as q, open(args.write, 'w') as w:
    pa = pa.read().splitlines()
    q = q.read().splitlines()

    split = append_wh(pa, q)

    for i in split:
        w.write(i)
        w.write('\n')