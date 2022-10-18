"""MCCWS model training script.

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
    --use_dset as \
    --use_dset cityu \
    --use_dset msr \
    --use_dset pku \
    --use_unc 1 \
    --warmup_step 50000 \
    --weight_decay 0.0
"""

import argparse
import distutils.util
import json
import logging
import os
import sys
from typing import List

import torch
import torch.utils.data
import torch.utils.tensorboard
import transformers
from tqdm import tqdm

import src.dset
import src.model
import src.utils.rand
import src.vars

logger = logging.getLogger(__name__)


def save_model(step: int, exp_name: str, model: torch.nn.Module):
  file_path = os.path.join(src.vars.EXP_PATH, exp_name, f'model-{step}.pt')
  torch.save(model, file_path)


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.train', description='Train MCCWS model.')
  parser.add_argument(
    '--pre_exp_name',
    help='''
    Name of the preprocess experiment.
    We use this as model name to load pre-trained model with criterion tokens.
    ''',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--exp_name',
    help='Training experiment name.',
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
    '--ckpt_step',
    help='Checkpoint interval.',
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
    '--log_step',
    help='Log interval.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--lr',
    help='Learning rate.',
    required=True,
    type=float,
  )
  parser.add_argument(
    '--p_drop',
    help='Dropout probability.',
    required=True,
    type=float,
  )
  parser.add_argument(
    '--max_norm',
    help='Gradient clipping max norm.',
    required=True,
    type=float,
  )
  parser.add_argument(
    '--total_step',
    help='Total training step.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--warmup_step',
    help='Warmup training step.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--weight_decay',
    help='Weight decay coefficient.',
    required=True,
    type=float,
  )
  parser.add_argument(
    '--seed',
    help='Control random seed.',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--use_dset',
    action='append',
    choices=src.vars.ALL_DSETS,
    help='Select datasets to load.',
    required=True,
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

  # Learning rate must be between 0 and 1 but exclusive from both end points.
  if not (0.0 < args.lr < 1.0):
    logger.error('Learning rate must be between 0 and 1 (exclusive).')
    exit(0)
  # Dropout probability must be between 0 and 1.
  if not (0.0 <= args.p_drop <= 1.0):
    logger.error('Dropout probability must be between 0 and 1 (inclusive).')
    exit(0)
  # Steps must be positive.
  if args.warmup_step <= 0 or args.total_step <= 0 or args.ckpt_step <= 0 or args.log_step <= 0:
    logger.error('Steps must be positive.')
    exit(0)
  # Warmup step must be less than total step.
  if args.warmup_step > args.total_step:
    logger.error('Warmup step must be less than total step')
    exit(0)

  # Must choose at least one dataset.
  if not args.use_dset:
    logger.error('Must choose at least one dataset.')
    exit(0)
  args.use_dset.sort()

  return args


def main(argv: List[str]) -> None:
  args = parse_args(argv=argv)

  src.utils.rand.set_seed(seed=args.seed)

  exp_dir_path = os.path.join(src.vars.EXP_PATH, args.exp_name)
  log_dir_path = os.path.join(src.vars.LOG_PATH, args.exp_name)
  if not os.path.exists(exp_dir_path):
    os.makedirs(exp_dir_path)
  if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

  # Save preprocess configuration.
  json.dump(
    args.__dict__,
    open(os.path.join(exp_dir_path, 'training_cfg.json'), 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )

  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')

  logger.info('Start loading datasets.')

  data_loader = torch.utils.data.DataLoader(
    batch_size=args.batch_size,
    dataset=src.dset.Dset(
      dset_names=args.use_dset,
      pre_exp_name=args.pre_exp_name,
      split='train',
      use_unc=args.use_unc,
    ),
    shuffle=True,
  )

  logger.info('Finish loading datasets.')
  logger.info('Start loading model.')

  model = src.model.WithCriterion(
    pre_exp_name=args.pre_exp_name,
    p_drop=args.p_drop,
    dset_names=args.use_dset,
  )
  model.train()
  model = model.to(device)

  logger.info('Finish loading model.')

  no_decay = ['bias', 'LayerNorm.weight']
  optim_group_params = [
    {
      'params': [param for name, param in model.named_parameters() if not any(nd in name for nd in no_decay)],
      'weight_decay': args.weight_decay,
    },
    {
      'params': [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay)],
      'weight_decay': 0.0,
    },
  ]

  optim = torch.optim.AdamW(optim_group_params, betas=(0.9, 0.999), eps=1e-8, lr=args.lr)
  schlr = transformers.get_linear_schedule_with_warmup(
    optimizer=optim,
    num_warmup_steps=args.warmup_step,
    num_training_steps=args.total_step,
  )

  writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir_path)
  cli_logger = tqdm(range(args.total_step), desc=f'loss: {0:.6f}', dynamic_ncols=True)

  logger.info('Start training model.')

  step = 0
  avg_criterion_loss = 0.0
  avg_bmes_loss = 0.0
  avg_total_loss = 0.0
  while step < args.total_step:
    for attention_mask, input_ids, token_type_ids, answer in data_loader:
      _, criterion_loss, _, bmes_loss = model(
        attention_mask=attention_mask.to(device),
        input_ids=input_ids.to(device),
        token_type_ids=token_type_ids.to(device),
        answer=answer.to(device),
      )

      total_loss = criterion_loss + bmes_loss

      avg_criterion_loss += criterion_loss.item()
      avg_bmes_loss += bmes_loss.item()
      avg_total_loss += total_loss.item()

      total_loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)

      optim.step()
      schlr.step()

      optim.zero_grad()

      step += 1

      if step % args.ckpt_step == 0:
        save_model(step=step, exp_name=args.exp_name, model=model)

      if step % args.log_step == 0:
        avg_criterion_loss = avg_criterion_loss / args.log_step
        avg_bmes_loss = avg_bmes_loss / args.log_step
        avg_total_loss = avg_total_loss / args.log_step

        cli_logger.set_description(f'loss: {avg_total_loss:.6f}')
        cli_logger.update()

        writer.add_scalar('criterion_loss', avg_criterion_loss, step)
        writer.add_scalar('bmes_loss', avg_bmes_loss, step)
        writer.add_scalar('total_loss', avg_total_loss, step)

        avg_criterion_loss = 0.0
        avg_bmes_loss = 0.0
        avg_total_loss = 0.0

      if step >= args.total_step:
        break

  save_model(step=step, exp_name=args.exp_name, model=model)
  writer.flush()
  writer.close()
  cli_logger.close()

  logger.info('Finish training model.')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
