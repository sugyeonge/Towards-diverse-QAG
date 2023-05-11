data=$1
device=$2

fname=`echo ${data} | cut -d '.' -f1`
echo ${fname}

mkdir infer_result
mkdir infer_result/passage
mkdir infer_result/question
mkdir infer_result/answer

# data_preprocessing -> ${data}.write
python data_preprocess.py --file ${data}
python ../query-focused-sum/multiencoder/inference_summary_input.py --file ${fname}'.write'


# get summary
CUDA_VISIBLE_DEVICES=${device} python ../query-focused-sum/multiencoder/train.py   --do_predict \
  --test_file ${fname}.jsonl   --model_name_or_path \
../query-focused-sum/segenc-qmsum-16384-512-wikisum-1   --multiencoder_type bart \
  --multiencoder_max_num_chunks 32   --multiencoder_stride \
  --max_source_len 512   --output_dir temp \
  --generation_max_len 256   --val_max_target_length 256 \
  --per_device_eval_batch_size 2   --predict_with_generate \
  --prediction_path ${fname}.summary


# initial answer generation
python answer_prompt.py --passage ${fname}.passage --question ${fname}.summary --write ${fname}.0input

CUDA_VISIBLE_DEVICES=${device} python ./examples/bart/inference.py --model-dir ./checkpoint_bart_sum_a --model-file checkpoint_best.pt --src ${fname}.0input --out ${fname}.0output


# question generation
python wh-prompt.py --passage ${fname}.passage --answer ${fname}.0output --write ${fname}.1input

CUDA_VISIBLE_DEVICES=${device} python ./examples/bart/inference.py --model-dir ./checkpoint_qg/ --model-file checkpoint_best.pt --src ${fname}.1input --out ${fname}.1output


# answer adjustment
python answer_prompt.py --passage ${fname}.1input_aug_passage --question ${fname}.1output --write ${fname}.2input

CUDA_VISIBLE_DEVICES=${device} python ./examples/bart/inference.py --model-dir ./checkpoint_qa --model-file checkpoint_best.pt --src ${fname}.2input --out ${fname}.2output


# finalize
mv ${fname}.1input_aug_passage infer_result/passage
mv ${fname}.2output infer_result/answer
mv ${fname}.1output infer_result/question

rm ${fname}.*output*
rm ${fname}.*input*
rm ${fname}.jsonl
rm ${fname}.summary
rm ${fname}.passage
rm ${fname}.write

# ranking
python for_ranker_distribution/load_generated.py --source ./infer_result --output_dir rank_output
python for_ranker_distribution/rerank-github.py --source rank_output --target rerank_output
python for_ranker_distribution/trim.py --source rerank_output --topk 5

rm -r rank_output
rm -r rerank_output
rm -r infer_result

# final generated output is in 'inference_result'
