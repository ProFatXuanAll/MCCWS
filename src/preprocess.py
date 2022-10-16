r"""Text preprocess script.

.. code-block:: shell

  python -m src.preprocess \
    --dev_ratio 0.1 \
    --exp_name my_preproc_exp \
    --max_len 60 \
    --use_dset as \
    --use_dset cityu \
    --use_dset msr \
    --use_dset pku \
    --use_width_norm 1 \
    --use_num_norm 1 \
    --use_alpha_norm 1 \
    --use_mix_alpha_num_norm 1
"""

import argparse
import distutils.util
import logging
import sys
from typing import List

logger = logging.getLogger(__name__)


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.utils.preprocess', description='Preprocess text.')
  parser.add_argument(
    '--dev_ratio',
    help='''
    Spliting ratio for development set.
    Only work for SIGHAN 2005 and 2008 bakeoff datasets.
    ''',
    required=True,
    type=float,
  )
  parser.add_argument(
    '--exp_name',
    help='Preprocess experiment name.',
    required=True,
    type=str,
  )
  parser.add_argument(
    '--max_len',
    help='''
    Maximum number of characters in a sentence.
    For training set, each sentence are chunked into subsequences with length not longer than --max_len.
    For development and test sets, sentences are chunked before inference and merged back after inference.
    ''',
    required=True,
    type=int,
  )
  parser.add_argument(
    '--use_dset',
    action='append',
    choices=['as', 'cityu', 'ctb8', 'msr', 'pku', 'sxu', 'weibo'],
    help='Select datasets to preprocess.',
    required=True,
  )
  parser.add_argument(
    '--use_width_norm',
    help='''
    Convert full-width characters into half-width.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.str2bool,
  )
  parser.add_argument(
    '--use_num_norm',
    help='''
    Convert consecutive digits into one representative digit.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.str2bool,
  )
  parser.add_argument(
    '--use_alpha_norm',
    help='''
    Convert consecutive alphabets into one representative alphabet.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.str2bool,
  )
  parser.add_argument(
    '--use_mix_alpha_num_norm',
    help='''
    Convert consecutive alphanumeric into one representative character.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.str2bool,
  )

  args = parser.parse_args(argv)

  # Ratio must between 0 and 1 but exclusive from both end points.
  if not (0.0 < args.dev_ratio < 1.0):
    logger.error('Development splitting ratio must between 0 and 1 (exclusive).')
    exit(0)

  # Must be both true or false.
  if args.use_num_norm and not args.use_width_norm:
    logger.warn('Full width digits are not normalized.')
  if args.use_alpha_norm and not args.use_width_norm:
    logger.warn('Full width alphabets are not normalized.')
  if args.use_mix_alpha_num_norm and not args.use_num_norm:
    logger.error('Alphanumerics are normalized only after digits are normalized.')
    exit(0)
  if args.use_mix_alpha_num_norm and not args.use_alpha_norm:
    logger.error('Alphanumerics are normalized only after alphabets are normalized.')
    exit(0)

  return args


def main(argv: List[str]) -> None:
  args = parse_args(argv=argv)
  print(args)


if __name__ == '__main__':
  main(argv=sys.argv[1:])
