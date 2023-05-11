import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--qp_file', type=str, required=True)
argparser.add_argument('--write', type=str, required=True)
args = argparser.parse_args()

def swap(pa):
    new_pass = []
    for passage in pa:
        passages = passage.split(' <SEP> ')
        new_p = passages[1] + ' </s> ' + passages[0]
        new_pass.append(new_p.strip())
    return new_pass

with open(args.qp_file, 'r') as qp, open(args.write, 'w') as w:
    qp = qp.read().splitlines()
    
    for i in swap(qp):
        w.write(i)
        w.write('\n')