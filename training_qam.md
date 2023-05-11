# Question Answering (Training)

1. **Format data**

```bash
# Data preprocessing
split="train"/"val"/"test"

mkdir data/QAM_raw
python swap.py --qp_file data/BART_QA/${split}.source --write data/QAM_raw/${split}.source

cp data/BART_QA/${split}.target data/QAM_raw/
```

2. **Train QAM**

```bash
# Data Naming
for SPLIT in train val test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "data/QAM_raw/$SPLIT.$LANG" \
    --outputs "data/QAM_raw/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "data/QAM_raw/train.bpe" \
  --validpref "data/QAM_raw/val.bpe" \
	--testpref "data/QAM_raw/test.bpe" \
  --destdir "QAM-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt

sh ./train_qam.sh
```

3. **Checkpoint output:** `./checkpoint_qa/checkpoint_best.pt`
