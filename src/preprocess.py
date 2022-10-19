r"""Text preprocess script.

.. code-block:: shell

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
"""

import argparse
import copy
import distutils.util
import json
import logging
import os
import pickle
import re
import sys
from typing import Callable, Dict, List

import torch
import transformers

import src.utils.download_data
import src.utils.rand
import src.vars
from src.vars import TAG_SET

logger = logging.getLogger(__name__)

NUM_NORM_PTTN = re.compile(r'((-|\+)?\d+((\.|·)\d+)?%?)+')
ALPHA_NORM_PTTN = re.compile(r'[A-Za-z_.]+')
MIX_ALPHA_NUM_NORM_PTTN = re.compile(r'([0x]){2,}')
SPLIT_PUNC_PTTN = re.compile(r'^[。！？：；…、，（）”’,;!?、,…]+$')
NO_DEV_DSETS = ['as', 'cityu', 'msr', 'pku']


def width_norm(sent: str) -> str:
  """Convert full-width characters into half-width.

  Source: https://github.com/acphile/MCCWS/blob/51f4a76c3f0e5cf760dc60e4b6bc779baaa5e108/prepoccess.py
  """
  out_sent = ''
  for char in sent:
    char_unicode = ord(char)
    if char_unicode == 12288:  # 全角空格直接转换
      char_unicode = 32
    elif 65281 <= char_unicode <= 65374:  # 全角字符（除空格）根据关系转化
      char_unicode -= 65248
    out_sent += chr(char_unicode)
  return out_sent


def num_norm(sent: str) -> str:
  """Convert consecutive digits into one digit.

  We use `0` as representative digit.
  Source: https://github.com/acphile/MCCWS/blob/51f4a76c3f0e5cf760dc60e4b6bc779baaa5e108/prepoccess.py
  """
  return NUM_NORM_PTTN.sub(r'0', sent).strip()


def alpha_norm(sent: str) -> str:
  """Convert consecutive alphabets into one alphabet.

  We use `x` as representative alphabet.
  Source: https://github.com/acphile/MCCWS/blob/51f4a76c3f0e5cf760dc60e4b6bc779baaa5e108/prepoccess.py
  """
  return ALPHA_NORM_PTTN.sub(r'x', sent).strip()


def mix_alpha_norm(sent: str) -> str:
  """Convert consecutive alphanumerics into one character.
  Must have at least two consecutive alphanumerics to perform conversion.

  We use `c` as representative alphabet.
  """
  return sent
  # return MIX_ALPHA_NUM_NORM_PTTN.sub(r'c', sent).strip()


def parse_args(argv: List[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser('python -m src.preprocess', description='Preprocess text.')
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
    '--model_name',
    help='''
    Pretrained Chinese model to use.
    We need this for tokenizing text.
    ''',
    choices=['bert-base-chinese'],
    required=True,
    type=str,
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
    type=distutils.util.strtobool,
  )
  parser.add_argument(
    '--use_num_norm',
    help='''
    Convert consecutive digits into one representative digit.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.strtobool,
  )
  parser.add_argument(
    '--use_alpha_norm',
    help='''
    Convert consecutive alphabets into one representative alphabet.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.strtobool,
  )
  parser.add_argument(
    '--use_mix_alpha_num_norm',
    help='''
    Convert consecutive alphanumeric into one representative character.
    Set to `1` to convert.
    Set to `0` to not convert.
    ''',
    required=True,
    type=distutils.util.strtobool,
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

  # Ratio must be between 0 and 1 but exclusive from both end points.
  if not (0.0 < args.dev_ratio < 1.0):
    logger.error('Development splitting ratio must be between 0 and 1 (exclusive).')
    exit(0)

  # Must choose at least one dataset.
  if not args.use_dset:
    logger.error('Must choose at least one dataset.')
    exit(0)
  args.use_dset.sort()

  # Must be both true or false.
  if args.use_num_norm and not args.use_width_norm:
    logger.warning('Full width digits are not normalized.')
  if args.use_alpha_norm and not args.use_width_norm:
    logger.warning('Full width alphabets are not normalized.')
  if args.use_mix_alpha_num_norm and not args.use_num_norm:
    logger.error('Alphanumerics are normalized only after digits are normalized.')
    exit(0)
  if args.use_mix_alpha_num_norm and not args.use_alpha_norm:
    logger.error('Alphanumerics are normalized only after alphabets are normalized.')
    exit(0)

  return args


def read_sents_from_file(file_name: str, norm_funcs: List[Callable[[str], str]]) -> List[str]:
  logger.info(f'Start reading from {file_name}.')

  input_txt_file = open(os.path.join(src.vars.RAW_DATA_PATH, file_name), 'r', encoding='utf-8')
  lines = input_txt_file.readlines()
  input_txt_file.close()

  sents = []
  for line in lines:
    # Perform normalization.
    for norm_func in norm_funcs:
      line = norm_func(line)

    # Split by length and by punctuations.
    words = []
    for word in re.split(r'\s+', line):
      # Discard empty string.
      if not word:
        continue
      # Add word to form sentence.
      words.append(word)

      # Split on pre-defined punctuations characters.
      if SPLIT_PUNC_PTTN.match(word):
        sent = ' '.join(words).strip()
        words = []
        # Discard empty string.
        if sent:
          sents.append(sent)

    # Add remaining words.
    if words:
      sent = ' '.join(words).strip()
      # Discard empty string.
      if sent:
        sents.append(sent)

  logger.info(f'Finish reading from {file_name}.')

  return sents


def write_sents_to_file(exp_name: str, file_name: str, sents: List[str]) -> None:
  preprocess_exp_data_dir_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, exp_name)
  out_txt_file_path = os.path.join(preprocess_exp_data_dir_path, file_name)

  logger.info(f'Start writing to {out_txt_file_path}.')

  # Make sure path exist.
  if not os.path.exists(preprocess_exp_data_dir_path):
    os.makedirs(preprocess_exp_data_dir_path)

  if os.path.exists(out_txt_file_path):
    logger.warning(f'Overwritting existing file: {out_txt_file_path}')

  with open(out_txt_file_path, 'w', encoding='utf-8') as out_txt_file:
    for sent in sents:
      out_txt_file.write(f'{sent}\n')

  logger.info(f'Finish writing to {out_txt_file_path}.')


def split_sent_by_len(max_len: int, sents: List[str]) -> List[str]:
  new_sents = []
  for sent in sents:
    new_sent = []
    new_sent_len = 0
    for word in re.split(r'\s+', sent):
      if new_sent_len + len(word) >= max_len:
        new_sents.append(' '.join(new_sent))
        new_sent = []
        new_sent_len = 0
      new_sent.append(word)
      new_sent_len += len(word)

    if new_sent:
      new_sents.append(' '.join(new_sent))

  return new_sents


def load_tknzr(
  dset_names: List[str],
  exp_name: str,
  model_name: str,
  use_unc: bool,
) -> transformers.PreTrainedTokenizerBase:
  if model_name == 'bert-base-chinese':
    cfg = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name)
    tknzr = transformers.BertTokenizer.from_pretrained(model_name)
  else:
    raise ValueError(f'Invalid model name: {model_name}.')

  # Create criterion specific tokens.
  # Token are in the form [tk], for example: [as] and [pku].
  dset_tks = [f'[{dset_name}]' for dset_name in dset_names]

  if use_unc:
    dset_tks.append('[unc]')

  # Add criterion specific tokens to tokenizer and model.
  tknzr.add_special_tokens({'additional_special_tokens': dset_tks})
  model.resize_token_embeddings(cfg.vocab_size + len(dset_tks))

  preprocess_exp_model_dir_path = os.path.join(src.vars.EXP_PATH, exp_name)
  if not os.path.exists(preprocess_exp_model_dir_path):
    os.makedirs(preprocess_exp_model_dir_path)

  tknzr.save_pretrained(preprocess_exp_model_dir_path)
  model.save_pretrained(preprocess_exp_model_dir_path)

  return tknzr


def encode_sents(
  criterion_encode: Dict[str, int],
  dset_name: str,
  max_len: int,
  sents: List[str],
  tknzr: transformers.PreTrainedTokenizerBase,
) -> Dict[str, List[torch.Tensor]]:
  tensor_data = {
    'input_ids': [],
    'token_type_ids': [],
    'attention_mask': [],
    'answer': [],
  }

  for sent in sents:
    # BEMS tagging.
    answer: List[int] = []
    for word in re.split(r'\s+', sent):
      if len(word) == 1:
        answer.append(TAG_SET['s'])
      elif len(word) == 2:
        answer.extend([TAG_SET['b'], TAG_SET['e']])
      else:
        answer.extend([TAG_SET['b']] + [TAG_SET['m']] * (len(word) - 2) + [TAG_SET['e']])
    answer = [criterion_encode[dset_name]] + answer

    # First remove spaces between "words", then insert spaces between "characters".
    # This is need for sentencepiece to split sentences into characters.
    char_seq = ' '.join(list(re.sub(r'\s+', r'', sent)))
    # Add criterion specific token at front.
    char_seq = f'[{dset_name}] {char_seq}'

    # Encode with tokenizer.
    enc = tknzr.encode_plus(
      text=char_seq,
      add_special_tokens=True,
      padding='max_length',
      truncation=False,
      max_length=max_len + 3,  # for [CLS], [criterion] and [SEP].
      is_split_into_words=False,
      return_attention_mask=True,
      return_token_type_ids=True,
    )

    tensor_data['input_ids'].append(enc['input_ids'])
    tensor_data['token_type_ids'].append(enc['token_type_ids'])
    tensor_data['attention_mask'].append(enc['attention_mask'])

    answer = answer + [TAG_SET['pad']] * max(0, max_len + 2 - len(answer))
    tensor_data['answer'].append(answer)

  return tensor_data


def write_tensor_data_to_file(exp_name: str, file_name: str, tensor_data: Dict[str, List[torch.Tensor]]) -> None:
  preprocess_exp_data_dir_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, exp_name)
  out_pickle_file_path = os.path.join(preprocess_exp_data_dir_path, file_name)

  logger.info(f'Start writing to {out_pickle_file_path}.')

  # Make sure path exist.
  if not os.path.exists(preprocess_exp_data_dir_path):
    os.makedirs(preprocess_exp_data_dir_path)

  if os.path.exists(out_pickle_file_path):
    logger.warning(f'Overwritting existing file: {out_pickle_file_path}')

  pickle.dump(tensor_data, open(out_pickle_file_path, 'wb'))

  logger.info(f'Finish writing to {out_pickle_file_path}.')


def replace_with_unc(
  tensor_data: Dict[str, List[torch.Tensor]],
  tknzr: transformers.PreTrainedTokenizerBase,
) -> Dict[str, List[torch.Tensor]]:
  unc_tkid = tknzr.encode('[unc]')[0]
  out_tensor_data = copy.deepcopy(tensor_data)
  for idx in range(len(tensor_data['input_ids'])):
    out_tensor_data['input_ids'][idx][1] = unc_tkid
  return out_tensor_data


def main(argv: List[str]) -> None:
  args = parse_args(argv=argv)

  src.utils.rand.set_seed(seed=args.seed)

  preprocess_exp_model_dir_path = os.path.join(src.vars.EXP_PATH, args.exp_name)
  if not os.path.exists(preprocess_exp_model_dir_path):
    os.makedirs(preprocess_exp_model_dir_path)

  # Save preprocess configuration.
  json.dump(
    args.__dict__,
    open(os.path.join(preprocess_exp_model_dir_path, 'preprocess_cfg.json'), 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )

  # Make sure dataset exist.
  src.utils.download_data.download_all()

  # Load tokenizer for preprocessing.
  tknzr = load_tknzr(
    dset_names=args.use_dset,
    exp_name=args.exp_name,
    model_name=args.model_name,
    use_unc=args.use_unc,
  )

  norm_funcs = []
  if args.use_width_norm:
    norm_funcs.append(width_norm)
  if args.use_num_norm:
    norm_funcs.append(num_norm)
  if args.use_alpha_norm:
    norm_funcs.append(alpha_norm)
  if args.use_mix_alpha_num_norm:
    norm_funcs.append(mix_alpha_norm)

  criterion_encode = {dset_name: idx for idx, dset_name in enumerate(args.use_dset)}

  json.dump(
    criterion_encode,
    open(os.path.join(preprocess_exp_model_dir_path, 'criterion_encode.json'), 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )

  for dset_name in args.use_dset:
    test_sents = read_sents_from_file(file_name=f'{dset_name}_test.txt', norm_funcs=norm_funcs)
    # If no development set, then split development set from training set.
    if dset_name in NO_DEV_DSETS:
      sents = read_sents_from_file(file_name=f'{dset_name}_train.txt', norm_funcs=norm_funcs)
      split_idx = int(len(sents) * args.dev_ratio)
      dev_sents = sents[:split_idx]
      train_sents = sents[split_idx:]
    else:
      train_sents = read_sents_from_file(file_name=f'{dset_name}_train.txt', norm_funcs=norm_funcs)
      dev_sents = read_sents_from_file(file_name=f'{dset_name}_dev.txt', norm_funcs=norm_funcs)

    train_sents = split_sent_by_len(max_len=args.max_len, sents=train_sents)
    dev_sents = split_sent_by_len(max_len=args.max_len, sents=dev_sents)
    test_sents = split_sent_by_len(max_len=args.max_len, sents=test_sents)

    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_train.txt', sents=train_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_dev.txt', sents=dev_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_test.txt', sents=test_sents)

    train_tensor_data = encode_sents(
      criterion_encode=criterion_encode,
      dset_name=dset_name,
      max_len=args.max_len,
      sents=train_sents,
      tknzr=tknzr,
    )
    dev_tensor_data = encode_sents(
      criterion_encode=criterion_encode,
      dset_name=dset_name,
      max_len=args.max_len,
      sents=dev_sents,
      tknzr=tknzr,
    )
    test_tensor_data = encode_sents(
      criterion_encode=criterion_encode,
      dset_name=dset_name,
      max_len=args.max_len,
      sents=test_sents,
      tknzr=tknzr,
    )

    write_tensor_data_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_train.pkl', tensor_data=train_tensor_data)
    write_tensor_data_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_dev.pkl', tensor_data=dev_tensor_data)
    write_tensor_data_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_test.pkl', tensor_data=test_tensor_data)

    if args.use_unc:
      train_tensor_data = replace_with_unc(tensor_data=train_tensor_data, tknzr=tknzr)
      dev_tensor_data = replace_with_unc(tensor_data=dev_tensor_data, tknzr=tknzr)
      test_tensor_data = replace_with_unc(tensor_data=test_tensor_data, tknzr=tknzr)

      write_tensor_data_to_file(
        exp_name=args.exp_name,
        file_name=f'{dset_name}_train.unc.pkl',
        tensor_data=train_tensor_data,
      )
      write_tensor_data_to_file(
        exp_name=args.exp_name,
        file_name=f'{dset_name}_dev.unc.pkl',
        tensor_data=dev_tensor_data,
      )
      write_tensor_data_to_file(
        exp_name=args.exp_name,
        file_name=f'{dset_name}_test.unc.pkl',
        tensor_data=test_tensor_data,
      )


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
