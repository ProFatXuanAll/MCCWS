"""MCCWS model inference script.

.. code-block:: shell

  python -m src.eval_mccws_f1 \
    --exp_name my_infer_exp \
    --first_ckpt 1 \
    --gpu 0 \
    --last_ckpt 200000 \
    --seed 42 \
    --split dev \
    --use_unc 0
"""

import argparse
import distutils.util
import json
import logging
import os
import re
import sys
from collections import deque
from typing import List

import torch.utils.tensorboard
from tqdm import tqdm

import src.dset
import src.utils.model
import src.utils.rand
import src.vars

logger = logging.getLogger(__name__)


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.eval_mccws_f1', description='MCCWS model inference.')
  parser.add_argument(
    '--exp_name',
    help='Inference experiment name.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--first_ckpt',
    help='First checkpoint to be evaluated.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--last_ckpt',
    help='Last checkpoint to be evaluated.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--gpu',
    help='GPU id.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--split',
    choices=['train', 'dev', 'test'],
    help='Which split of the dataset to evaluate.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--use_unc',
    help='''
    Whether to use [unc] tokens.
    Set to `1` to use.
    Set to `0` to not use.
    ''',
    required=True,
    type=distutils.util.strtobool,
  )

  args = parser.parse_args(argv)

  # Checkpoints must be positive.
  if args.first_ckpt <= 0 or args.last_ckpt <= 0:
    logger.error('Steps must be positive.')
    exit(1)
  # First checkpoint must be less than the last checkpoint step.
  if args.first_ckpt > args.last_ckpt:
    logger.error('Warmup step must be less than total step')
    exit(1)

  return args


@torch.no_grad()
def main(argv: List[str]) -> None:
  args = parse_args(argv=argv)

  exp_dir_path = os.path.join(src.vars.EXP_PATH, args.exp_name)
  log_dir_path = os.path.join(src.vars.LOG_PATH, args.exp_name)
  if not os.path.exists(exp_dir_path):
    logger.error(f'Experiment {args.exp_name} does not exist.')
    exit(1)
  if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

  infer_cfg = json.load(open(
    os.path.join(src.vars.EXP_PATH, args.exp_name, 'infer_cfg.json'),
    'r',
    encoding='utf-8',
  ))

  model_exp_name = infer_cfg['model_exp_name']
  infer_first_ckpt = infer_cfg['first_ckpt']
  infer_last_ckpt = infer_cfg['last_ckpt']

  train_cfg = json.load(
    open(
      os.path.join(src.vars.EXP_PATH, model_exp_name, 'train_cfg.json'),
      'r',
      encoding='utf-8',
    )
  )

  pre_exp_name = train_cfg['pre_exp_name']
  criterion_encode = json.load(
    open(
      os.path.join(src.vars.EXP_PATH, pre_exp_name, 'criterion_encode.json'),
      'r',
      encoding='utf-8',
    )
  )

  logger.info('Start calculating F1.')

  ckpts = src.utils.model.list_ckpts(exp_name=model_exp_name, first_ckpt=infer_first_ckpt, last_ckpt=infer_last_ckpt)
  ckpts = list(filter(lambda ckpt: args.first_ckpt <= ckpt <= args.last_ckpt, ckpts))
  if not ckpts:
    logger.error('No checkpoint is selected.')
    exit(1)

  for ckpt in ckpts:
    for dset_name in criterion_encode.keys():
      if not os.path.exists(os.path.join(exp_dir_path, f'{dset_name}_{args.split}.{ckpt}.txt')):
        logger.error(f'Inference result on checkpoint {ckpt} does not exist.')
        exit(1)

  writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir_path)

  score_script_path = os.path.join(src.vars.PROJECT_ROOT, 'src', 'score')
  for ckpt in tqdm(ckpts):
    for dset_name in criterion_encode.keys():
      train_file_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, pre_exp_name, f'{dset_name}_train.txt')
      eval_file_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, pre_exp_name, f'{dset_name}_{args.split}.txt')
      infer_file_path = os.path.join(exp_dir_path, f'{dset_name}_{args.split}.{ckpt}.txt')
      out_f1_file_path = os.path.join(exp_dir_path, f'{dset_name}_{args.split}.{ckpt}.f1')

      # Run perl script provided by SIGHAN 2005.
      script = f'perl {score_script_path} {train_file_path} {eval_file_path} {infer_file_path} > {out_f1_file_path}'
      exit_code = os.system(script)
      if exit_code != 0:
        logger.error('something wrong with perl script.')
        exit(1)
      # Read the last 7 lines from the output file of the perl script.
      # Source: https://splunktool.com/copy-the-last-three-lines-of-a-text-file-in-python
      with open(out_f1_file_path, 'r', encoding='utf-8') as f:
        lines = deque(f, 7)

      # Parse metric scores from the output file.
      recall = float(re.match(r'={3}\s+TOTAL\s+TRUE\s+WORDS\s+RECALL:\s+(\d\.\d+)', lines[0]).group(1))
      precision = float(re.match(r'={3}\s+TOTAL\s+TEST\s+WORDS\s+PRECISION:\s+(\d\.\d+)', lines[1]).group(1))
      f1 = float(re.match(r'={3}\s+F\s+MEASURE:\s+(\d\.\d+)', lines[2]).group(1))
      oov_recall = float(re.match(r'={3}\s+OOV\s+Recall\s+Rate:\s+(\d\.\d+)', lines[4]).group(1))
      iv_recall = float(re.match(r'={3}\s+IV\s+Recall\s+Rate:\s+(\d\.\d+)', lines[5]).group(1))

      if args.use_unc:
        writer.add_scalar(f'{dset_name}/{args.split}/unc/recall', recall, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/unc/precision', precision, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/unc/f1', f1, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/unc/oov_recall', oov_recall, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/unc/iv_recall', iv_recall, ckpt)
      else:
        writer.add_scalar(f'{dset_name}/{args.split}/{dset_name}/recall', recall, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/{dset_name}/precision', precision, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/{dset_name}/f1', f1, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/{dset_name}/oov_recall', oov_recall, ckpt)
        writer.add_scalar(f'{dset_name}/{args.split}/{dset_name}/iv_recall', iv_recall, ckpt)

  writer.close()
  logger.info('Finish calculating F1.')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
