import os
import re
from typing import List

import torch

import src.vars


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


def save_model(step: int, exp_name: str, model: torch.nn.Module) -> None:
  file_path = os.path.join(src.vars.EXP_PATH, exp_name, f'model-{step}.pt')
  torch.save(model, file_path)
