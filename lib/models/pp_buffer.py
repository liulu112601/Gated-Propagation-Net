import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from configs import AverageMeter
from datasets import cls_sampler
from copy import deepcopy

class pp_buffer(nn.Module):
  def __init__(self, dataset, fea_dim):
    super(pp_buffer, self).__init__()
    cls2idx = dataset.cls2idx
    sampler = cls_sampler(cls2idx)
    self.dataloader = torch.utils.data.DataLoader( dataset, batch_sampler=sampler, num_workers=16) 
    self.num_total_classes = len(cls2idx) 
    pp_running = torch.rand((self.num_total_classes, fea_dim)) 
    self.fea_dim = fea_dim
    self.register_buffer("pp_running", pp_running)

  def reset_buffer(self, emb_model):
    with torch.no_grad():
      for batch_idx, (img_idx, imgs, labels) in enumerate(self.dataloader):
        cls_pp_lst = []
        lab = labels[0]
        assert len(set(labels.tolist())) == 1
        for start in range(0, len(imgs), 200):
          end = min(len(imgs), start+200)
          pp = emb_model(imgs[start: end])
          cls_pp_lst.append(pp)
        cls_pp = torch.mean( torch.cat(cls_pp_lst, dim=0), dim=0 )
        self.pp_running[lab] = cls_pp
      torch.cuda.empty_cache()

  def init_test_buffer(self, train_pp_buffer):
    # all training classes include fine classes 
    with torch.no_grad():
      self.pp_running = deepcopy( train_pp_buffer.pp_running )
