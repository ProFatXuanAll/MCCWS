r"""Text preprocess script.

.. code-block:: shell

  python -m src.get_preprocess_stats --exp_name my_pre_exp --tablefmt latex
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import List, Union

import pandas as pd

import src.utils.download_data
import src.utils.rand
import src.vars

logger = logging.getLogger(__name__)


def get_sents_stats(sents: List[str], train_sents: List[str]) -> List[Union[int, float]]:
  # Calculate length statistics.
  n_sents = len(sents)
  n_unique_sents = len(set(sents))

  sents_lens = list(map(lambda sent: len(re.sub(r'\s+', r'', sent)), sents))
  sents_max_len = max(sents_lens)
  sents_min_len = min(sents_lens)
  sents_avg_len = sum(sents_lens) / n_sents

  chars = []
  words = []
  for sent in sents:
    for word in re.split(r'\s+', sent):
      words.append(word)
      for char in word:
        chars.append(char)

  n_words = len(words)
  n_unique_words = len(set(words))

  words_lens = list(map(len, words))
  words_max_len = max(words_lens)
  words_avg_len = sum(words_lens) / n_words

  n_chars = len(chars)
  n_unique_chars = len(set(chars))

  # Calculate OOV statistics.
  train_words = set()
  for sent in train_sents:
    train_words.update(re.split(r'\s+', sent))

  n_oov = 0
  for word in words:
    if word not in train_words:
      n_oov += 1
  oov_rate = n_oov / n_words * 100

  return [
    n_chars,
    n_sents,
    n_words,
    n_unique_chars,
    n_unique_sents,
    n_unique_words,
    oov_rate,
    sents_avg_len,
    sents_max_len,
    sents_min_len,
    words_avg_len,
    words_max_len,
  ]


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.preprocess', description='Preprocess text.')
  parser.add_argument(
    '--exp_name',
    help='Preprocess experiment name.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--tablefmt',
    choices=['grid', 'latex'],
    help='Output table format.',
    required=True,
    type=str,
  )

  args = parser.parse_args(argv)

  return args


def read_preprocess_cfg_from_file(exp_name: str) -> argparse.Namespace:
  """Read preprocess configuration."""
  preprocess_exp_model_dir_path = os.path.join(src.vars.EXP_PATH, exp_name)
  if not os.path.exists(preprocess_exp_model_dir_path):
    raise ValueError('Preprocess experiment does not exist.  Run preprocess script first.')

  cfg = json.load(open(os.path.join(preprocess_exp_model_dir_path, 'preprocess_cfg.json'), 'r', encoding='utf-8'))
  return argparse.Namespace(**cfg)


def read_sents_from_file(exp_name: str, file_name: str) -> List[str]:
  preprocess_exp_data_dir_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, exp_name)
  if not os.path.exists(preprocess_exp_data_dir_path):
    raise ValueError('Preprocess experiment does not exist.  Run preprocess script first.')

  with open(os.path.join(preprocess_exp_data_dir_path, file_name), 'r') as f:
    sents = list(map(lambda sent: sent.strip(), f.readlines()))
  return sents


def main(argv: List[str]) -> None:
  args = parse_args(argv=argv)

  preprocess_cfg = read_preprocess_cfg_from_file(exp_name=args.exp_name)

  src.utils.rand.set_seed(seed=preprocess_cfg.seed)

  stats_map = {}
  all_train_sents = []
  all_dev_sents = []
  all_test_sents = []

  for dset_name in preprocess_cfg.use_dset:
    train_sents = read_sents_from_file(exp_name=args.exp_name, file_name=f'{dset_name}_train.norm.txt')
    dev_sents = read_sents_from_file(exp_name=args.exp_name, file_name=f'{dset_name}_dev.norm.txt')
    test_sents = read_sents_from_file(exp_name=args.exp_name, file_name=f'{dset_name}_test.norm.txt')

    all_train_sents.extend(train_sents)
    all_dev_sents.extend(dev_sents)
    all_test_sents.extend(test_sents)

    stats_map[f'{dset_name}_train'] = get_sents_stats(sents=train_sents, train_sents=train_sents)
    stats_map[f'{dset_name}_dev'] = get_sents_stats(sents=dev_sents, train_sents=train_sents)
    stats_map[f'{dset_name}_test'] = get_sents_stats(sents=test_sents, train_sents=train_sents)

  stats_map['all_train'] = get_sents_stats(sents=all_train_sents, train_sents=all_train_sents)
  stats_map['all_dev'] = get_sents_stats(sents=all_dev_sents, train_sents=all_train_sents)
  stats_map['all_test'] = get_sents_stats(sents=all_test_sents, train_sents=all_train_sents)

  df = pd.DataFrame(
    stats_map,
    index=[
      '#.chars.',
      '#.sents.',
      '#.words',
      '#.u.chars.',
      '#.u.sents.',
      '#.u.words',
      'OOV%',
      'sents.avg.len.',
      'sents.max.len.',
      'sents.min.len.',
      'words.avg.len.',
      'words.max.len.',
    ],
  )

  # These are not needed to be shown on paper.
  df = df.drop(index=['#.u.sents.', 'sents.max.len.', 'sents.min.len.', 'words.avg.len.', 'words.max.len.'])

  print(df.transpose().to_markdown(tablefmt=args.tablefmt, floatfmt='10.2f'))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
