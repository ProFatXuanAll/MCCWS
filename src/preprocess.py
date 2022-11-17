r"""Text preprocess script.

.. code-block:: shell

  python -m src.preprocess \
    --dev_ratio 0.1 \
    --exp_name my_pre_exp \
    --max_len 64 \
    --model_name bert-base-chinese \
    --seed 42 \
    --use_dset as \
    --use_dset cityu \
    --use_dset cnc \
    --use_dset ctb6 \
    --use_dset msr \
    --use_dset pku \
    --use_dset sxu \
    --use_dset ud \
    --use_dset wtb \
    --use_dset zx
"""

import argparse
import copy
import json
import logging
import os
import pickle
import re
import sys
from typing import Dict, List

import torch
import transformers

import src.utils.download_data
import src.utils.rand
import src.vars
from src.vars import TAG_SET

logger = logging.getLogger(__name__)

NO_DEV_DSETS = ['as', 'cityu', 'msr', 'pku', 'sxu']

ALPHA_NORM_PTTN = re.compile(r'[A-Za-z_.]+')
NUM_NORM_PTTN = re.compile(r'((-|\+)?(\d+)([\.|·/∶:]\d+)?%?)+')
PUNC_PTTN = re.compile(r'^[。！？：；…+、，（）“”’,;!?、,()『』]+$')
SPACE_PTTN = re.compile(r'\s+')


def full_width_norm(sent: str) -> str:
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
  return out_sent.strip()


def num_norm(sent: str) -> str:
  """Convert consecutive digits into one digit.

  We use `0` as representative digit.

  Source:
  https://github.com/acphile/MCCWS/blob/51f4a76c3f0e5cf760dc60e4b6bc779baaa5e108/prepoccess.py
  https://github.com/koukaiu/dlut-nihao/blob/master/src/utils.py
  """
  return NUM_NORM_PTTN.sub(r'0', sent).strip()


def alpha_norm(sent: str) -> str:
  """Convert consecutive alphabets into one alphabet.

  We use `x` as representative alphabet.

  Source:
  https://github.com/acphile/MCCWS/blob/51f4a76c3f0e5cf760dc60e4b6bc779baaa5e108/prepoccess.py
  https://github.com/koukaiu/dlut-nihao/blob/master/src/utils.py
  """
  return ALPHA_NORM_PTTN.sub(r'x', sent).strip()


def find_trunc_idx(max_len: int, words: List[str]) -> int:
  """Find truncate index in the sentence.

  First try to find the last punctuation in the sentence.
  If there is no punctuation, then truncate to max length.

  Return index of the last word before truncated.
  """
  # Find last punctuation.
  for idx in range(len(words) - 1, -1, -1):
    if PUNC_PTTN.match(words[idx]):
      return idx

  # Fail to find punctuation.
  # Find length truncation index instead.
  char_count = 0
  for idx, word in enumerate(words):
    if char_count + len(word) > max_len:
      return idx
    char_count += len(word)

  # This cannot happen.
  raise ValueError('No truncation index can be found.')


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
    Each sentence are chunked into subsequences with length not longer than --max_len.
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

  args = parser.parse_args(argv)

  # Ratio must be between 0 and 1 but exclusive from both end points.
  if not (0.0 < args.dev_ratio < 1.0):
    logger.error('Development splitting ratio must be between 0 and 1 (exclusive).')
    exit(1)

  # Must choose at least one dataset.
  if not args.use_dset:
    logger.error('Must choose at least one dataset.')
    exit(1)
  args.use_dset.sort()

  return args


def read_sents_from_file(file_name: str, max_len: int) -> List[str]:
  logger.info(f'Start reading from {file_name}.')

  input_txt_file = open(os.path.join(src.vars.RAW_DATA_PATH, file_name), 'r', encoding='utf-8')
  lines = input_txt_file.readlines()
  input_txt_file.close()

  ori_short_sents = []
  norm_short_sents = []
  for line in lines:
    ori_long_sent = full_width_norm(sent=line)
    norm_long_sent = num_norm(sent=ori_long_sent)
    norm_long_sent = alpha_norm(sent=norm_long_sent)

    ori_words = SPACE_PTTN.split(ori_long_sent)
    norm_words = SPACE_PTTN.split(norm_long_sent)

    assert len(ori_words) == len(norm_words)

    # Perform normalization and split by length.
    start_idx = 0
    char_count = 0
    for end_idx in range(len(norm_words)):
      word = norm_words[end_idx]

      # Record number of characters.
      char_count += len(word)

      # If exceed length limit, then truncate.
      if char_count > max_len:
        # Find truncation index.
        # Use `+ start_idx` to shift starting point.
        trunc_idx = find_trunc_idx(max_len=max_len, words=norm_words[start_idx:end_idx + 1]) + start_idx

        # Perform truncation.
        # The word in the truncation point is included.
        ori_short_sent = ' '.join(ori_words[start_idx:trunc_idx + 1]).strip()
        ori_short_sents.append(ori_short_sent)

        norm_short_sent = ' '.join(norm_words[start_idx:trunc_idx + 1]).strip()
        norm_short_sents.append(norm_short_sent)

        # Shift start index.
        start_idx = trunc_idx + 1
        # Reestimate character counts.
        char_count = sum(map(len, norm_words[start_idx:end_idx + 1]))

    # Add remaining words.
    if norm_words[start_idx:]:
      ori_short_sent = ' '.join(ori_words[start_idx:]).strip()
      ori_short_sents.append(ori_short_sent)

      norm_short_sent = ' '.join(norm_words[start_idx:]).strip()
      norm_short_sents.append(norm_short_sent)

  assert len(ori_short_sents) == len(norm_short_sents)

  logger.info(f'Finish reading from {file_name}.')

  return ori_short_sents, norm_short_sents


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
) -> transformers.PreTrainedTokenizerBase:
  if model_name == 'bert-base-chinese':
    cfg = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.BertModel.from_pretrained(model_name)
    tknzr = transformers.BertTokenizer.from_pretrained(model_name)
  else:
    raise ValueError(f'Invalid model name: {model_name}.')

  # Create criterion specific tokens and unknown criterion token.
  # Criterion specific tokens are in the form [m], for example: [as] and [pku].
  dset_tks = [f'[{dset_name}]' for dset_name in dset_names] + ['[unc]']

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
  # tknzr = load_tknzr(
  #   dset_names=args.use_dset,
  #   exp_name=args.exp_name,
  #   model_name=args.model_name,
  # )

  criterion_encode = {dset_name: idx for idx, dset_name in enumerate(args.use_dset)}

  json.dump(
    criterion_encode,
    open(os.path.join(preprocess_exp_model_dir_path, 'criterion_encode.json'), 'w', encoding='utf-8'),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )

  for dset_name in args.use_dset:
    test_ori_sents, test_norm_sents = read_sents_from_file(file_name=f'{dset_name}_test.txt', max_len=args.max_len)
    train_ori_sents, train_norm_sents = read_sents_from_file(file_name=f'{dset_name}_train.txt', max_len=args.max_len)
    # If no development set, then split development set from training set.
    if dset_name in NO_DEV_DSETS:
      split_idx = int(len(train_norm_sents) * args.dev_ratio)
      dev_ori_sents = train_ori_sents[:split_idx]
      dev_norm_sents = train_norm_sents[:split_idx]
      train_ori_sents = train_ori_sents[split_idx:]
      train_norm_sents = train_norm_sents[split_idx:]
    else:
      dev_ori_sents, dev_norm_sents = read_sents_from_file(file_name=f'{dset_name}_dev.txt', max_len=args.max_len)

    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_train.ori.txt', sents=train_ori_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_train.norm.txt', sents=train_norm_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_dev.ori.txt', sents=dev_ori_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_dev.norm.txt', sents=dev_norm_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_test.ori.txt', sents=test_ori_sents)
    write_sents_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_test.norm.txt', sents=test_norm_sents)

    # train_tensor_data = encode_sents(
    #   criterion_encode=criterion_encode,
    #   dset_name=dset_name,
    #   max_len=args.max_len,
    #   sents=train_sents,
    #   tknzr=tknzr,
    # )
    # dev_tensor_data = encode_sents(
    #   criterion_encode=criterion_encode,
    #   dset_name=dset_name,
    #   max_len=args.max_len,
    #   sents=dev_sents,
    #   tknzr=tknzr,
    # )
    # test_tensor_data = encode_sents(
    #   criterion_encode=criterion_encode,
    #   dset_name=dset_name,
    #   max_len=args.max_len,
    #   sents=test_sents,
    #   tknzr=tknzr,
    # )

    # write_tensor_data_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_train.pkl', tensor_data=train_tensor_data)
    # write_tensor_data_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_dev.pkl', tensor_data=dev_tensor_data)
    # write_tensor_data_to_file(exp_name=args.exp_name, file_name=f'{dset_name}_test.pkl', tensor_data=test_tensor_data)

    # if args.use_unc:
    #   train_tensor_data = replace_with_unc(tensor_data=train_tensor_data, tknzr=tknzr)
    #   dev_tensor_data = replace_with_unc(tensor_data=dev_tensor_data, tknzr=tknzr)
    #   test_tensor_data = replace_with_unc(tensor_data=test_tensor_data, tknzr=tknzr)

    #   write_tensor_data_to_file(
    #     exp_name=args.exp_name,
    #     file_name=f'{dset_name}_train.unc.pkl',
    #     tensor_data=train_tensor_data,
    #   )
    #   write_tensor_data_to_file(
    #     exp_name=args.exp_name,
    #     file_name=f'{dset_name}_dev.unc.pkl',
    #     tensor_data=dev_tensor_data,
    #   )
    #   write_tensor_data_to_file(
    #     exp_name=args.exp_name,
    #     file_name=f'{dset_name}_test.unc.pkl',
    #     tensor_data=test_tensor_data,
    #   )


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main(argv=sys.argv[1:])
