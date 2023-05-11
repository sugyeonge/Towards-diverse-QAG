# Answer Generation (Training)

<br/> 
1. **Get summaries**

```bash
# QFS model download
cp -r /path/to/fairseq/data/BART_QA /path/to/query-focused-sum

sh ./fairytale_summary.sh ${train/val/test}
```

```bash
# Make format (output file -> train/val/test.jsonl)
split="train"/"val"/"test"
python make_summary_input.py --fname BART_QA/${split}.source --split ${split}

# Inference QFS model
CUDA_VISIBLE_DEVICES=${device} python train.py   --do_predict \
  --test_file ${split}.jsonl   --model_name_or_path \
../segenc-qmsum-16384-512-wikisum-1   --multiencoder_type bart \
  --multiencoder_max_num_chunks 32   --multiencoder_stride \
  --max_source_len 512   --output_dir temp \
  --generation_max_len 256   --val_max_target_length 256 \
  --per_device_eval_batch_size 2   --predict_with_generate \
  --prediction_path ${split}.summary

mkdir AGM_raw

python agm_input.py --pq_file ${split}.source --summary ${split}.summary --write AGM_raw/${split}.source

mv BART_QA/${split}.target AGM_raw #answer
mv AGM_raw /path/to/fairseq/data
```
<br/> 

2. **Train AGM**

```bash
cd /path/to/fairseq

# Data Naming
for SPLIT in train val test
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "data/AGM_raw/$SPLIT.$LANG" \
    --outputs "data/AGM_raw/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

# Fairseq preprocess
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "data/AGM_raw/train.bpe" \
  --validpref "data/AGM_raw/val.bpe" \
	--testpref "data/AGM_raw/test.bpe" \
  --destdir "SUM_A-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt

# train
sh ./train_agm.sh
```

<br/> 
3. **Checkpoint output:** `./checkpoint_bart_sum_a/checkpoint_best.pt`
