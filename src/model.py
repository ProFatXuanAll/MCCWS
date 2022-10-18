import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

import src.vars


class WithCriterion(nn.Module):

  def __init__(self, pre_exp_name: str, p_drop: float, dset_names: List[str]):
    super().__init__()

    model_name = os.path.join(src.vars.EXP_PATH, pre_exp_name)
    self.encoder = transformers.BertModel.from_pretrained(model_name)
    cfg = transformers.BertConfig.from_pretrained(model_name)

    self.n_tags = len(src.vars.TAG_SET)
    self.n_dsets = len(dset_names)

    self.bmes_linear = nn.Sequential(
      nn.Dropout(p=p_drop),
      nn.Linear(in_features=cfg.hidden_size, out_features=len(src.vars.TAG_SET)),
    )
    self.criterion_linear = nn.Sequential(
      nn.Dropout(p=p_drop),
      nn.Linear(in_features=cfg.hidden_size, out_features=len(dset_names)),
    )

    self.bmes_loss_fn = nn.CrossEntropyLoss(ignore_index=src.vars.TAG_SET['pad'])
    self.criterion_loss_fn = nn.CrossEntropyLoss()

  def forward(
    self,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    answer: Optional[torch.Tensor] = None,
  ):
    out = self.encoder(
      attention_mask=attention_mask,
      input_ids=input_ids,
      token_type_ids=token_type_ids,
    )
    # (B, S, H) -> (B, H) -> (B, n_dsets)
    criterion_logits = self.criterion_linear(out.last_hidden_state[:, 1, :])
    criterion_pred = F.softmax(criterion_logits, dim=1)

    # (B, S, H) -> (B, S-2, H) -> (B, S-2, n_tags)
    bmes_logits = self.bmes_linear(out.last_hidden_state[:, 2:, :])
    bmes_pred = F.softmax(bmes_logits, dim=2)

    criterion_loss = None
    bmes_loss = None
    if answer is not None:
      criterion_loss = self.criterion_loss_fn(criterion_logits, answer[:, 0])

      B = bmes_logits.size(0)
      S = bmes_logits.size(1)
      BS = B * S
      bmes_loss = self.bmes_loss_fn(bmes_logits.reshape(BS, self.n_tags), answer[:, 1:].reshape(BS))

    return criterion_pred, criterion_loss, bmes_pred, bmes_loss
