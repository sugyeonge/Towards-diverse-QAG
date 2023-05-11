# python
from functools import reduce
from pathlib import Path
import json
import argparse
# 3rd-party
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from setproctitle import setproctitle
# framework
from train import load

class GeneratedLoader:
    """
    reduced by filename(bookname) and passage(context)
    ex)self.collections = {
        'sections-alleleiraugh-or-the-many-furred-creature':{
            'context1': [
                'context1 <SEP> question <SEP> answer',
                'context1 <SEP> question <SEP> answer',...
            ],
            'context2': [
                'context2 <SEP> question <SEP> answer',
                'context2 <SEP> question <SEP> answer',...
            ], ...
        }
    }
    """
    def __init__(self, root, tokenizer, clean):
        questions_without_the = []
        answers_without_the = []
        def _pair_by_passage(acc, cur):
            context = cur.split(tokenizer.sep_token)[0].strip()
            if context not in acc:
                acc[context] = []
            if not cur in acc[context]:
                acc[context].append(cur)
            return acc
        def _pair_by_passage_exceptthe(acc, cur):
            context = cur.split(tokenizer.sep_token)[0].strip()
            if context not in acc:
                acc[context] = []
            if not cur in acc[context]:
                question, answer = tuple(cur.split(tokenizer.sep_token)[1:])
                question_without_the = question.replace('the ', '').replace('that ', '')
                answer_without_the = answer.replace('the ', '').replace('that ', '')
                if question_without_the not in questions_without_the and answer_without_the not in answers_without_the:
                    acc[context].append(cur)
                    questions_without_the.append(question_without_the)
                    answers_without_the.append(answer_without_the)
            return acc
        root = Path(root)
        if not root.is_dir():
            raise ValueError

        #assert len(list(root.iterdir())) == 23
        self.collection = {}
        for path in root.iterdir():
            with open(path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
            lines = [line.replace('<SEP>', tokenizer.sep_token).strip() for line in lines]
            if clean:
                self.collection[path.stem] = reduce(_pair_by_passage_exceptthe, lines, {})
            else:
                self.collection[path.stem] = reduce(_pair_by_passage, lines, {})
            questions_without_the = []
            answers_without_the = []


class GeneratedLoader2:  # for iter1, iter2, iter3, iter4 directory
    def __init__(self, root, tokenizer):
        root = Path(root)
        sep = tokenizer.sep_token
        p = root / 'passage'
        passages = {}
        for path in p.iterdir():
            with open(path, 'rt', encoding='utf-8') as f:
                passages[path.stem] = f.readlines()

        p = root / 'question'
        questions = {}
        for path in p.iterdir():
            with open(path, 'rt', encoding='utf-8') as f:
                questions[path.stem] = f.readlines()

        p = root / 'answer'
        answers = {}
        for path in p.iterdir():
            with open(path, 'rt', encoding='utf-8') as f:
                answers[path.stem] = f.readlines()

        #assert len(list(passages.keys())) == len(list(questions.keys())) == len(list(answers.keys())) == 23
        assert set(list(passages.keys())) == set(list(questions.keys())) == set(list(answers.keys()))

        def _pair_by_passage(acc, cur):
            context = cur[0]
            if context not in acc:
                acc[context] = []
            if not cur in acc[context]:
                acc[context].append(f" {sep} ".join([c.strip() for c in cur]))
            return acc

        self.collection = {}
        for story_name in list(passages.keys()):
            self.collection[story_name] = list(zip(passages[story_name], questions[story_name], answers[story_name]))
            self.collection[story_name] = reduce(_pair_by_passage, self.collection[story_name], {})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        default=None,
                        type=str,)
    parser.add_argument('--output_dir',
                        default=None,
                        type=str,)
    return parser.parse_args()

# preference variable
clean = False

setproctitle('applying reranker')
model_name = 'roberta-base'
model_file = 'model_ckpt/roberta-base_5e-07/5.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
print(f'clean: {clean}')
print(f'code version: {53}')

args = parse_args()
source = args.source
output_dir = args.output_dir
assert source is not None

print('model loading..')
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
).to(device).eval()
load(model, None, model_file)
print('model loading completed')

tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left')

try:
    loader = GeneratedLoader(source, tokenizer, clean)
except:
    loader = GeneratedLoader2(source, tokenizer)

save_to = Path(output_dir)
print('save to ' + output_dir)

# inference
collections = loader.collection
sorted_collections = {key:[] for key in collections.keys()}
for i, (book, val) in enumerate(collections.items()):
    print(f'processing book: {book}')
    for passage, examples in val.items():
        scores = []
        for example in examples:
            with torch.no_grad():
                logits = model(**tokenizer(example, return_tensors='pt').to(device)).logits.detach().cpu()
            score = float(logits[0][1] - logits[0][0])
            scores.append(score)
        scored_examples = [(example, scores[i]) for i, example in enumerate(examples)]
        sorted_examples = sorted(scored_examples, key=lambda x:x[1], reverse=True)
        val[passage] = sorted_examples

# save to json
sep = tokenizer.sep_token
for book in collections:
    for passage in collections[book]:
        # collections[book][passage] = [qag_pair[0].split(sep)[1:] for qag_pair in collections[book][passage]]
        collections[book][passage] = [(qag_pair[0].split(sep)[1:], qag_pair[1]) for qag_pair
                                      in collections[book][passage]]
if not save_to.is_dir():
    save_to.mkdir(parents=True)
for book in collections.keys():
    filename = book + '.json'
    with open(save_to / filename, 'wt', encoding='utf-8') as f:
        json.dump(collections[book], f, ensure_ascii=False, indent=2)
    # for passage in collections[book].keys():
    #     collections[book][passage] = [text for text, _ in collections[book][passage]]

print("program finished")
