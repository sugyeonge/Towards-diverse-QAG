import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--passage', type=str, required=True)
argparser.add_argument('--answer', type=str, required=True)
argparser.add_argument('--write', type=str, required=True)
arg = argparser.parse_args()

def concat_wh(pas, ans):
    c = []
    p = []
    for i, j in zip(pas, ans):
        c.append(i + ' </s> ' + j + ' </s> ' + 'who')
        c.append(i + ' </s> ' + j + ' </s> ' + 'when')
        c.append(i + ' </s> ' + j + ' </s> ' + 'where')
        c.append(i + ' </s> ' + j + ' </s> ' + 'what')
        c.append(i + ' </s> ' + j + ' </s> ' + 'how')
        c.append(i + ' </s> ' + j + ' </s> ' + 'why')
        for idx in range(6):
            p.append(i)
    return c, p

with open(arg.passage, 'r') as f, open(arg.answer, 'r') as f2:
   passage = f.read().splitlines()
   answer = f2.read().splitlines()

data, aug_pas = concat_wh(passage, answer)

assert len(data) == len(aug_pas)
with open(arg.write, 'w') as w, open(arg.write+'_aug_passage', 'w') as w2:
    for i in data:
        w.write(i)
        w.write('\n')

    for i in aug_pas:
        w2.write(i)
        w2.write('\n')
