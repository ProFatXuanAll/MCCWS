"""MCCWS model eval script.

.. code-block:: shell

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
"""

import argparse
import distutils.util
import json
import logging
import os
import re
import sys
from typing import List

import torch
import torch.utils.data
import torch.utils.tensorboard
from tqdm import tqdm

import src.dset
import src.utils.rand
import src.vars

logger = logging.getLogger(__name__)


def list_ckpts(exp_name: str, first_ckpt: int, last_ckpt: int) -> List[int]:
  ckpts = []
  for file_name in os.listdir(os.path.join(src.vars.EXP_PATH, exp_name)):
    match = re.match(r'model-(\d+).pt', file_name)
    if not match:
      continue

    ckpt = int(match.group(1))
    if first_ckpt <= ckpt <= last_ckpt:
      ckpts.append(ckpt)

  ckpts.sort()
  return ckpts


def load_model(ckpt: int, exp_name: str) -> torch.nn.Module:
  file_path = os.path.join(src.vars.EXP_PATH, exp_name, f'model-{ckpt}.pt')
  return torch.load(file_path)


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.eval_mccws', description='Evaluate MCCWS model.')
  parser.add_argument(
    '--model_exp_name',
    help='model experiment name to be evaluated.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--exp_name',
    help='eval experiment name.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--batch_size',
    help='Mini-batch size.',
    required=True,
    type=int,
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
    '--seed',
    help='Control random seed.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--split',
    choices=['train', 'dev', 'test'],
    help='Which split of the dataset to evaluate.',
    required=True,
    type=int,
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
    exit(0)
  # First checkpoint must be less than the last checkpoint step.
  if args.first_ckpt > args.last_ckpt:
    logger.error('Warmup step must be less than total step')
    exit(0)

  return args


@torch.no_grad()
def main(argv: List[str]) -> None:
  args = parse_args(argv=argv)

  src.utils.rand.set_seed(seed=args.seed)

  model_exp_dir_path = os.path.join(src.vars.EXP_PATH, args.model_exp_name)
  eval_exp_dir_path = os.path.join(src.vars.EXP_PATH, args.exp_name)
  log_dir_path = os.path.join(src.vars.LOG_PATH, args.exp_name)
  if not os.path.exists(model_exp_dir_path):
    logger.error(f'Experiment {args.exp_name} does not exist.')
    exit(0)
  if not os.path.exists(eval_exp_dir_path):
    os.makedirs(eval_exp_dir_path)
  if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

  # Save evaluation configuration.
  json.dump(
    args.__dict__,
    open(os.path.join(eval_exp_dir_path, 'eval_cfg.json'), 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )

  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')

  logger.info('Start loading datasets.')

  # Load criterion encoding.
  train_cfg = json.load(open(os.path.join(src.vars.EXP_PATH, args.model_exp, 'train_cfg.json')))
  pre_exp_name = train_cfg['pre_exp_name']
  criterion_encode = json.load(open(os.path.join(src.vars.EXP_PATH, pre_exp_name, 'criterion_encode.json')))

  data_loader_map = {}
  for dset_name in criterion_encode.keys():
    data_loader = torch.utils.data.DataLoader(
      batch_size=args.batch_size,
      dataset=src.dset.EvalDset(
        dset_name=dset_name,
        pre_exp_name=pre_exp_name,
        split=args.split,
        use_unc=args.use_unc,
      ),
      shuffle=False,
    )
    data_loader_map[dset_name] = data_loader

  logger.info('Finish loading datasets.')
  logger.info('Start evaluating model.')

  writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir_path)

  for ckpt in tqdm(
    list_ckpts(exp_name=args.model_exp_name, first_ckpt=args.first_ckpt, last_ckpt=args.last_ckpt),
    dynamic_ncols=True,
  ):
    model = load_model(ckpt=ckpt, exp_name=args.model_exp_name)
    model.eval()
    model = model.to(device)

    for dset_name, data_loader in data_loader_map.items():

      for attention_mask, input_ids, token_type_ids, answer in data_loader:
        criterion_pred, _, bmes_pred, _ = model(
          attention_mask=attention_mask.to(device),
          input_ids=input_ids.to(device),
          token_type_ids=token_type_ids.to(device),
          answer=None,
        )

      # writer.add_scalar('criterion_loss', avg_criterion_loss, step)
      # writer.add_scalar('bmes_loss', avg_bmes_loss, step)
      # writer.add_scalar('total_loss', avg_total_loss, step)

  writer.flush()
  writer.close()

  logger.info('Finish evaluating model.')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
