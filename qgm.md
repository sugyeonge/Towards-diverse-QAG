# Question Generation (Training)
###
1. **Add interrogative-word indicator**

```bash
cd path/to/fairseq
mkdir data/QGM_raw

split="train"/"val"/"test"
python add_wh.py --pa_file data/BART_QG/${split}.source --q_file data/BART_QG/${split}.target --write data/QGM_raw/${split}.source

cp data/BART_QG/${split}.target data/QGM_raw/
```
#
2. **Train QGM**

```bash
# Data Naming
for SPLIT in train val test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "data/QGM_raw/$SPLIT.$LANG" \
    --outputs "data/QGM_raw/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "data/QGM_raw/train.bpe" \
  --validpref "data/QGM_raw/val.bpe" \
	--testpref "data/QGM_raw/test.bpe" \
  --destdir "QGM-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt

sh ./train_qgm.sh
```
#
3. **Checkpoint output:** `./checkpoint_qg/checkpoint_best.pt`
