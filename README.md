# MCCWS

Works on MCCWS.

## Installation

```sh
# Some datasets are compressed by rar.
apt install unrar

# Python dependency.
pipenv install
```

## Development Installation.

```sh
pipenv install --dev
```

## Execution

```sh
# First run preprocessing.
python -m src.preprocess \
  --dev_ratio 0.1 \
  --exp_name my_pre_exp \
  --model_name bert-base-chinese \
  --max_len 60 \
  --seed 42 \
  --use_dset as \
  --use_dset cityu \
  --use_dset msr \
  --use_dset pku \
  --use_width_norm 1 \
  --use_num_norm 1 \
  --use_alpha_norm 1 \
  --use_mix_alpha_num_norm 1 \
  --use_unc 1

# Then use the preprocessing result to train MCCWS.
python -m src.train_mccws \
  --batch_size 64 \
  --ckpt_step 5000 \
  --exp_name my_model_exp \
  --gpu 0 \
  --log_step 1000 \
  --lr 2e-5 \
  --max_norm 10.0 \
  --pre_exp_name my_pre_exp \
  --p_drop 0.1 \
  --seed 42 \
  --total_step 200000 \
  --use_unc 1 \
  --warmup_step 50000 \
  --weight_decay 0.0

# After training, generate model inference on dev sets.
python -m src.infer_mccws \
  --batch_size 512 \
  --exp_name my_dev_infer_exp \
  --first_ckpt 100000 \
  --gpu 0 \
  --last_ckpt 200000 \
  --model_exp_name my_model_exp \
  --seed 42 \
  --split dev \
  --use_unc 0

# Run dev F1 evaluation and find the checkpoint with highest dev F1.
python -m src.eval_mccws_f1 \
  --exp_name my_dev_infer_exp \
  --first_ckpt 100000 \
  --gpu 0 \
  --last_ckpt 200000 \
  --split dev \
  --use_unc 0

# Use the checkpoint with highest F1 on dev to inference on test sets.
python -m src.infer_mccws \
  --batch_size 512 \
  --exp_name my_test_infer_exp \
  --first_ckpt 175000 \
  --gpu 0 \
  --last_ckpt 200000 \
  --model_exp_name my_model_exp \
  --seed 42 \
  --split test \
  --use_unc 0

# Run test F1 evaluation.
python -m src.eval_mccws_f1 \
  --exp_name my_test_infer_exp \
  --first_ckpt 175000 \
  --gpu 0 \
  --last_ckpt 200000 \
  --split test \
  --use_unc 0

# Run test F1 evaluation with UNC.
python -m src.eval_mccws_f1 \
  --exp_name my_test_unc_infer_exp \
  --first_ckpt 175000 \
  --gpu 0 \
  --last_ckpt 200000 \
  --split test \
  --use_unc 1
```

