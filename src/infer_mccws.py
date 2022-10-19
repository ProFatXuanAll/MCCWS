"""MCCWS model inference script.

.. code-block:: shell

  python -m src.infer_mccws \
    --batch_size 512 \
    --exp_name my_infer_exp \
    --first_ckpt 100000 \
    --gpu 0 \
    --last_ckpt 200000 \
    --model_exp_name my_model_exp \
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
from typing import List

import torch
import torch.utils.data
from tqdm import tqdm

import src.dset
import src.utils.model
import src.utils.rand
import src.vars

logger = logging.getLogger(__name__)


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.infer_mccws', description='MCCWS model inference.')
  parser.add_argument(
    '--model_exp_name',
    help='Model experiment name to inference.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--exp_name',
    help='Inference experiment name.',
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
  infer_exp_dir_path = os.path.join(src.vars.EXP_PATH, args.exp_name)
  log_dir_path = os.path.join(src.vars.LOG_PATH, args.exp_name)
  if not os.path.exists(model_exp_dir_path):
    logger.error(f'Experiment {args.exp_name} does not exist.')
    exit(0)
  if not os.path.exists(infer_exp_dir_path):
    os.makedirs(infer_exp_dir_path)
  if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

  # Save evaluation configuration.
  json.dump(
    args.__dict__,
    open(os.path.join(infer_exp_dir_path, 'infer_cfg.json'), 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )

  device = torch.device('cpu')
  if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu}')

  logger.info('Start loading datasets.')

  # Load criterion encoding.
  train_cfg = json.load(
    open(
      os.path.join(src.vars.EXP_PATH, args.model_exp_name, 'train_cfg.json'),
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
  logger.info('Start inferencing model.')

  ckpts = src.utils.model.list_ckpts(exp_name=args.model_exp_name, first_ckpt=args.first_ckpt, last_ckpt=args.last_ckpt)

  for ckpt in tqdm(ckpts, dynamic_ncols=True):
    model = src.utils.model.load_model(ckpt=ckpt, exp_name=args.model_exp_name)
    model.eval()
    model = model.to(device)

    for dset_name, data_loader in data_loader_map.items():
      out_txt_file_name = f'{dset_name}_{args.split}.{ckpt}.txt'
      out_acc_file_name = f'{dset_name}_{args.split}.{ckpt}.acc'

      out_txt_file = open(os.path.join(infer_exp_dir_path, out_txt_file_name), 'w', encoding='utf-8')
      out_acc_file = open(os.path.join(infer_exp_dir_path, out_acc_file_name), 'w', encoding='utf-8')
      for attention_mask, input_ids, token_type_ids, answer, sents in data_loader:
        criterion_pred, _, bmes_pred, _ = model(
          attention_mask=attention_mask.to(device),
          input_ids=input_ids.to(device),
          token_type_ids=token_type_ids.to(device),
          answer=None,
        )

        # Decode criterion.
        # (B, n_dsets) -> (B)
        for cid in criterion_pred.argmax(dim=1).tolist():
          out_acc_file.write(f'{cid}\n')

        # Decode words by BMES.
        # (B, S, n_tags) -> (B, S)
        for bmes_list, sent in zip(bmes_pred.argmax(dim=2).tolist(), sents):
          pred_sent = ''
          merge_word = ''
          for char, bmes_id in zip(list(re.sub(r'\s+', r'', sent)), bmes_list):
            # TODO: always split punctuation.
            # In the middle of a word.
            if merge_word:
              # Word is now finish.
              if bmes_id == src.vars.TAG_SET_E_ID:
                pred_sent += f'{merge_word}{char} '
                merge_word = ''
              # Still in the middle of a word.
              elif bmes_id == src.vars.TAG_SET_M_ID:
                merge_word += char
              # Model predict new multi-characters word without closing previous word.
              # In this case we simply treat the previous word is end and start a new multi-characters word.
              elif bmes_id == src.vars.TAG_SET_B_ID:
                pred_sent += f'{merge_word} '
                merge_word = char
              # Model predict new single-character word without closing previous word.
              # In this case we simply treat the previous word is end and start a new single-character word.
              elif bmes_id == src.vars.TAG_SET_S_ID:
                pred_sent += f'{merge_word} {char} '
                merge_word = ''
              # Model predict something we never optimized for.
              # This must not happen, but in case it happens, we treat it like a single-character word.
              else:
                pred_sent += f'{merge_word} {char} '
                merge_word = ''
            # Seeing a new word.
            else:
              # Seeing a new single-character word.
              if bmes_id == src.vars.TAG_SET_S_ID:
                pred_sent += f'{char} '
              # Seeing a new multi-characters word.
              elif bmes_id == src.vars.TAG_SET_B_ID:
                merge_word = char
              # Model predict middle word but no word to be merged.
              # In this case we treat it as new multi-characters word.
              elif bmes_id == src.vars.TAG_SET_M_ID:
                merge_word = char
              # Model predict end word but no word to be merged.
              # In this case we treat it as new single-character word.
              elif bmes_id == src.vars.TAG_SET_E_ID:
                pred_sent += f'{char} '
              # Model predict something we never optimized for.
              # This must not happen, but in case it happens, we treat it like a single-character word.
              else:
                pred_sent += f'{char} '

          # Model must predict the last character as E or S.
          # But if model failed to do that, then we simply close the multi-characters word.
          if merge_word:
            pred_sent += f'{merge_word} '

          # Remove the last whitespace.
          pred_sent = pred_sent[:-1]
          out_txt_file.write(f'{pred_sent}\n')

      out_txt_file.close()
      out_acc_file.close()

  logger.info('Finish inferencing model.')


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
