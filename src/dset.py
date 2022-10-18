import logging
import os
import pickle
from typing import Dict, List

import torch
import torch.utils.data

import src.vars

logger = logging.getLogger(__name__)


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
  logger.info(f'Finish loading dataset {file_path}')

  return tensor_data


class Dset(torch.utils.data.Dataset):

  def __init__(self, dset_names: List[str], pre_exp_name: str, split: str, use_unc: bool):
    assert split in ['train', 'dev', 'test']
    self.attention_mask: List[torch.Tensor] = []
    self.input_ids: List[torch.Tensor] = []
    self.token_type_ids: List[torch.Tensor] = []
    self.answer: List[torch.Tensor] = []

    for dset_name in dset_names:
      tensor_data = read_tensor_data(
        dset_name=dset_name,
        pre_exp_name=pre_exp_name,
        split=split,
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
        split=split,
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
