# Ranking model (Training)
###
1. **Training ranker model with FairytaleQA**

```bash
python train.py #--model_file ${model_file}
```

This produces the model checkpoint at `./model_ckpt`
#
2 **Apply ranker model for QA-pairs**

Following code generates checkpoint `./model_ckpt/roberta-base_5e-07/5.pth` automatically. OR You can download the model checkpoints

```bash
python load_generated.py --source ${source} --output_dir ${output_dir}
```

`${source}` : pre-constructed QA pairs for ranking

`${output_dir}` : directory for ranked QA-pairs
