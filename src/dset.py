import json
import logging
import os
import pickle
from typing import Dict, List

import torch
import torch.utils.data

import src.vars

logger = logging.getLogger(__name__)


def read_txt_data(dset_name: str, pre_exp_name: str, split: str) -> List[str]:
  file_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, pre_exp_name, f'{dset_name}_{split}.txt')

  logger.info(f'Start loading dataset {file_path}')
  in_file = open(file_path, 'r', encoding='utf-8')
  sents = list(filter(bool, [line.strip() for line in in_file.readlines()]))
  in_file.close()
  logger.info(f'Finish loading dataset {file_path}')

  return sents


def read_tensor_data(
  dset_name: str,
  pre_exp_name: str,
  split: str,
  use_unc: bool,
) -> Dict[str, List[torch.Tensor]]:
  if use_unc:
    file_name = f'{dset_name}_{split}.unc.pkl'
  else:
    file_name = f'{dset_name}_{split}.pkl'

  file_path = os.path.join(src.vars.PREPROCESS_DATA_PATH, pre_exp_name, file_name)

  logger.info(f'Start loading dataset {file_path}')

  tensor_data = pickle.load(open(file_path, 'rb'))
  out_tensor_data = {
    'input_ids': [],
    'attention_mask': [],
    'token_type_ids': [],
    'answer': [],
  }
  for idx in range(len(tensor_data['input_ids'])):
    out_tensor_data['input_ids'].append(torch.LongTensor(tensor_data['input_ids'][idx]))
    out_tensor_data['attention_mask'].append(torch.LongTensor(tensor_data['attention_mask'][idx]))
    out_tensor_data['token_type_ids'].append(torch.LongTensor(tensor_data['token_type_ids'][idx]))
    out_tensor_data['answer'].append(torch.LongTensor(tensor_data['answer'][idx]))

  logger.info(f'Finish loading dataset {file_path}')

  return out_tensor_data


class TrainDset(torch.utils.data.Dataset):

  def __init__(self, pre_exp_name: str, use_unc: bool):
    self.attention_mask: List[torch.Tensor] = []
    self.input_ids: List[torch.Tensor] = []
    self.token_type_ids: List[torch.Tensor] = []
    self.answer: List[torch.Tensor] = []

    criterion_encode = json.load(
      open(
        os.path.join(src.vars.EXP_PATH, pre_exp_name, 'criterion_encode.json'),
        'r',
        encoding='utf-8',
      )
    )

    for dset_name in criterion_encode.keys():
      tensor_data = read_tensor_data(
        dset_name=dset_name,
        pre_exp_name=pre_exp_name,
        split='train',
        use_unc=False,
      )
      self.attention_mask.extend(tensor_data['attention_mask'])
      self.input_ids.extend(tensor_data['input_ids'])
      self.token_type_ids.extend(tensor_data['token_type_ids'])
      self.answer.extend(tensor_data['answer'])

      if not use_unc:
        continue

      tensor_data = read_tensor_data(
        dset_name=dset_name,
        pre_exp_name=pre_exp_name,
        split='train',
        use_unc=True,
      )
      self.attention_mask.extend(tensor_data['attention_mask'])
      self.input_ids.extend(tensor_data['input_ids'])
      self.token_type_ids.extend(tensor_data['token_type_ids'])
      self.answer.extend(tensor_data['answer'])

  def __len__(self) -> int:
    return len(self.input_ids)

  def __getitem__(self, idx: int):
    return (
      self.attention_mask[idx],
      self.input_ids[idx],
      self.token_type_ids[idx],
      self.answer[idx],
    )


class EvalDset(torch.utils.data.Dataset):

  def __init__(self, dset_name: str, pre_exp_name: str, split: str, use_unc: bool):
    assert split in ['train', 'dev', 'test']
    tensor_data = read_tensor_data(
      dset_name=dset_name,
      pre_exp_name=pre_exp_name,
      split=split,
      use_unc=use_unc,
    )

    self.attention_mask: List[torch.Tensor] = tensor_data['attention_mask']
    self.input_ids: List[torch.Tensor] = tensor_data['input_ids']
    self.token_type_ids: List[torch.Tensor] = tensor_data['token_type_ids']
    self.answer: List[torch.Tensor] = tensor_data['answer']
    self.sents: List[str] = read_txt_data(dset_name=dset_name, pre_exp_name=pre_exp_name, split=split)

  def __len__(self) -> int:
    return len(self.input_ids)

  def __getitem__(self, idx: int):
    return (
      self.attention_mask[idx],
      self.input_ids[idx],
      self.token_type_ids[idx],
      self.answer[idx],
      self.sents[idx],
    )
