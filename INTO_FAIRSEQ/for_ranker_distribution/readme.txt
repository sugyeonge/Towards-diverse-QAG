### Training ranker model with FairytaleQA

```bash
python train.py --model_file ${model_file}
```

This produces the model checkpoint at `./model_ckpt` 

### Applying ranker model for QA-pairs that already been made

Following code utilizes checkpoint `./model_ckpt/roberta-base_5e-07/5.pth` automatically.

`${source}` : pre-constructed QA pairs for ranking

`${output_dir}` : directory for ranked QA-pairs