# Make format (output file -> train/val/test.jsonl)
split=$1 # train/dev/test
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

python agm_input.py --pq_file BART_QA/${split}.source --summary ${split}.summary --write AGM_raw/${split}.source

mv BART_QA/${split}.target AGM_raw #answer
