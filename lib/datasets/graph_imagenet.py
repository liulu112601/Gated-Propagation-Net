from __future__ import print_function
from PIL import Image
import random, numpy as np
import os, torch, copy, math
import torch.utils.data as data
from pathlib import Path
from collections import defaultdict

def make_dir(new_dir):
  if not new_dir.exists():
    os.makedirs(new_dir)

class GraphImageNetDataset(data.Dataset):
  def __init__(self, dataset_dir, mode, transform):
    super(GraphImageNetDataset, self).__init__()
    print('Building dataset graph-tiered for [{}] ...'.format(mode))
    self.dataset_dir  = Path(dataset_dir)
    if not self.dataset_dir.exists():
      raise RuntimeError('Dataset not found. You can use download=True to download it')

    #self.idx2img = torch.load( self.dataset_dir / 'cache-raw-data-tiered.pth')[ "all_images" ]
    #self.idx2distance = torch.load( self.dataset_dir / "img_idx2distance.pth")
    self.info = torch.load( self.dataset_dir /  "IDX_{}_info.pth".format(mode) )
    self.idx2cls      = self.info['imgidx_cls']
    self.cls2idx      = self.info['cls_idxs']
    self.children     = self.info['graph_children'] 
    self.parents      = self.info['graph_parents'] 
    self.transform    = transform
    print('==> Dataset: Found {:} classes and {:} images for all levels'.format(len(self.cls2idx), len(self.idx2cls) ) )
   
  def __getitem__(self, idx):
    #idx = idx.item()
    pil_image = torch.load(self.dataset_dir.parents[0] / "IMGS" / "{}.pth".format(idx)).copy()
    pil_image = Image.fromarray( pil_image )
    if self.transform:
      pil_image = self.transform(pil_image)
    return torch.IntTensor([idx]), pil_image, self.idx2cls[idx]

  def __len__(self):
    return len(self.idx2cls)

class FewShotSnowballSampler(object):
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations, hierarchy_info, n_snow, mode):
    # TODO: consider train and test cases
    super(FewShotSnowballSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it  = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes
    self.n_snow = n_snow
    if len(hierarchy_info) == 2:
      self.parents, self.children = hierarchy_info; mode = "train"
    else:
      self.parents, self.children, self.all_train_classes, self.all_test_classes = hierarchy_info; mode = "test"
    assert parents.keys() == children.keys()
    self.mode = mode
     
  def __iter__(self):
    spc = self.sample_per_class
    cpi = self.classes_per_it
    for it in range(self.iterations):
      ok = False
      while not ok:
        prop_graph     = random.sample(self.cls2idx.keys().tolist(), 1) 
        prop_graph     = check(prop_graph)
        if len(prop_graph) > 0 : ok = True
      ok = False
      while not ok:
        for i in range(self.n_snow):
          new_snow = [self.children[cls]+self.parents[cls] for cls in prop_graph]
          prop_graph.extend(new_snow)
          prop_graph = list(set(prop_graph))
          prop_graph = check(prop_graph)
        if mode == "train":
          if len(prop_graph) >= cpi:
            few_shot_classes = random.sample(prop_graph, cpi)
            ok = True
        else:
          test_can = list( set(prop_graph).intersection(set(self.all_test_classes)) ) 
          if len(test_can) >= cpi:
            few_shot_classes = random.sample(test_can, cpi)
            ok = True
      few_shot_batch = []
      for i, c in enumerate(few_shot_classes):
        img_idxs = self.cls2idx[c]
        few_shot_batch.extend( random.sample(img_idxs, spc))
      yield few_shot_batch

  def check(self, in_lst):
    if 0 in in_lst:   in_lst.remove(0)
    if 219 in in_lst: in_lst.remove(219)
    if 304 in in_lst: in_lst.remove(304)
    return in_lst


class FewShotGlobalSampler(object):
  
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations):
    super(FewShotGlobalSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes

  def __iter__(self):
    '''
    yield a batch of indexes using random sampling
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      few_shot_batch = []
      ok = False
      while not ok: # exclude root fall11 fa11misc in the sampled classes
        batch_few_shot_classes = random.sample(self.few_shot_classes, cpi) 
        #if (0 in batch_few_shot_classes) or (219 in batch_few_shot_classes) or (304 in batch_few_shot_classes):
        if 0:
          ok = False
        else: ok = True 
      for i, c in enumerate(batch_few_shot_classes):
        img_idxs = self.cls2idx[c]
        few_shot_batch.extend( random.sample(img_idxs, spc))
      yield few_shot_batch

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class FewShotLocalSampler(object):
  
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations, local_nei):
    super(FewShotLocalSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes
    self.local_nei = local_nei 

  def __iter__(self):
    '''
    yield a batch of indexes using random sampling
    if the task has one emptu img list, then resample a task
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      ok = False
      while not ok:
        few_shot_batch = []
        batch_classes = random.sample(self.few_shot_classes, 1)
        for i in range(cpi-1):
          local_nei = [self.local_nei[cls] for cls in batch_classes]
          local_nei = [item for sublist in local_nei for item in sublist]
          [c] = random.sample(local_nei, 1) 
          batch_classes.append(c)
        #if not self.check_task(batch_classes):
        if len(batch_classes) != len(set(batch_classes)):
          continue; ok = False
        else: 
          ok = True
          for cls in batch_classes:
            img_idxs = self.cls2idx[cls] 
            few_shot_batch.extend( random.sample(img_idxs, spc))
      yield few_shot_batch
  
  def check_task(self, task):
    if (0 in task) or (219 in task) or (304 in task) or len(task) != len(set(task)): return False
    else: return True

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class FewShotMixSampler(object):
  
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations, local_nei):
    super(FewShotMixSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes
    self.local_nei = local_nei 

  def __iter__(self):
    '''
    yield a batch of indexes using random sampling
    if the task has one emptu img list, then resample a task
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      ok = False
      while not ok:
        few_shot_batch = []
        n_global = cpi//2
        batch_classes = random.sample(self.few_shot_classes, n_global)
        for i in range(cpi- n_global):
          local_nei = [self.local_nei[cls] for cls in batch_classes]
          local_nei = [item for sublist in local_nei for item in sublist]
          [c] = random.sample(local_nei, 1) 
          batch_classes.append(c)
        #if not self.check_task(batch_classes):
        if len(batch_classes) != len(set(batch_classes)):
          continue; ok = False
        else: 
          ok = True
          for cls in batch_classes:
            img_idxs = self.cls2idx[cls] 
            few_shot_batch.extend( random.sample(img_idxs, spc))
      yield few_shot_batch
  
  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class FewShotAuxSampler(object):
  
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations, local_nei, total_iter, bs, idx2cls):
    super(FewShotAuxSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes
    self.local_nei = local_nei 
    self.total_iter = total_iter
    self.idx_iter  = None
    self.bs = bs
    self.idx = list(idx2cls.keys())

  '''
  def __iter__(self):
    #yield a batch of indexes using random sampling
    #if the task has one emptu img list, then resample a task
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      
      self.prob_aux  = math.pow(0.9, 20 * self.idx_iter / self.total_iter)
      r_num = random.random()
      if r_num < self.prob_aux:
        few_shot_batch  =random.sample(self.idx, self.bs)
      else:
        batch_size = spc * cpi
        ok = False
        while not ok:
          few_shot_batch = []
          n_global = cpi//2
          batch_classes = random.sample(self.few_shot_classes, n_global)
          for i in range(cpi- n_global):
            local_nei = [self.local_nei[cls] for cls in batch_classes]
            local_nei = [item for sublist in local_nei for item in sublist]
            [c] = random.sample(local_nei, 1)
            batch_classes.append(c)
          #if not self.check_task(batch_classes):
          if len(batch_classes) != len(set(batch_classes)):
            continue; ok = False
          else: 
            ok = True
            for cls in batch_classes:
              img_idxs = self.cls2idx[cls]
              few_shot_batch.extend( random.sample(img_idxs, spc))
      self.idx_iter += 1
      yield few_shot_batch
  '''

  def __iter__(self):
    spc = self.sample_per_class
    cpi = self.classes_per_it
    for it in range(self.iterations):
      self.prob_aux  = math.pow(0.9, 20 * self.idx_iter / self.total_iter)
      r_num = random.random()
      if r_num < self.prob_aux:
        few_shot_batch  =random.sample(self.idx, self.bs)
      else:
        t_num = random.random()
        if t_num > 0.5: # local
          batch_size = spc * cpi
          ok = False
          while not ok:
            few_shot_batch = []
            batch_classes = random.sample(self.few_shot_classes, 1)
            for i in range(cpi-1):
              local_nei = [self.local_nei[cls] for cls in batch_classes]
              local_nei = [item for sublist in local_nei for item in sublist]
              [c] = random.sample(local_nei, 1) 
              batch_classes.append(c)
            #if not self.check_task(batch_classes):
            if len(batch_classes) != len(set(batch_classes)):
              continue; ok = False
            else: 
              ok = True
              for cls in batch_classes:
                img_idxs = self.cls2idx[cls] 
                few_shot_batch.extend( random.sample(img_idxs, spc))
        else: # global
          batch_size = spc * cpi
          few_shot_batch = []
          ok = False
          while not ok: # exclude root fall11 fa11misc in the sampled classes
            batch_few_shot_classes = random.sample(self.few_shot_classes, cpi) 
            #if (0 in batch_few_shot_classes) or (219 in batch_few_shot_classes) or (304 in batch_few_shot_classes):
            if 0:
              ok = False
            else: ok = True 
          for i, c in enumerate(batch_few_shot_classes):
            img_idxs = self.cls2idx[c]
            few_shot_batch.extend( random.sample(img_idxs, spc))
      self.idx_iter += 1
      yield few_shot_batch
          
        

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class VisSampler(object):
  
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations, fix_point):
    super(VisSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes
    self.fix_point = fix_point

  def __iter__(self):
    '''
    yield a batch of indexes using random sampling
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      few_shot_batch = []
      ok = False
      while not ok: # exclude root fall11 fa11misc in the sampled classes
        batch_few_shot_classes = random.sample(self.few_shot_classes, cpi) 
        if not self.fix_point in batch_few_shot_classes: batch_few_shot_classes[0] = self.fix_point
        #if (0 in batch_few_shot_classes) or (219 in batch_few_shot_classes) or (304 in batch_few_shot_classes):
        if 0:
          ok = False
        else: ok = True 
      for i, c in enumerate(batch_few_shot_classes):
        img_idxs = self.cls2idx[c]
        few_shot_batch.extend( random.sample(img_idxs, spc))
      yield few_shot_batch

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class VisLocalSampler(object):
  
  def __init__(self, few_shot_classes, cls2idx, classes_per_it, num_samples, iterations, local_nei, fix_point):
    super(VisLocalSampler, self).__init__()
    self.cls2idx = cls2idx
    self.classes_per_it = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations
    self.few_shot_classes = few_shot_classes
    self.local_nei = local_nei 
    self.fix_point = fix_point

  def __iter__(self):
    '''
    yield a batch of indexes using random sampling
    if the task has one emptu img list, then resample a task
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      batch_size = spc * cpi
      ok = False
      while not ok:
        few_shot_batch = []
        batch_classes = [self.fix_point] 
        for i in range(cpi-1):
          local_nei = [self.local_nei[cls] for cls in batch_classes]
          local_nei = [item for sublist in local_nei for item in sublist]
          [c] = random.sample(local_nei, 1)
          batch_classes.append(c)
        if len(batch_classes) != len(set(batch_classes)):
          continue; ok = False
        else: 
          ok = True
          for cls in batch_classes:
            img_idxs = self.cls2idx[cls] 
            few_shot_batch.extend( random.sample(img_idxs, spc))
      yield few_shot_batch
  
  def check_task(self, task):
    if (0 in task) or (219 in task) or (304 in task) or len(task) != len(set(task)): return False
    else: return True

  def __len__(self):
    '''
    returns the number of iterations (episodes) per epoch
    '''
    return self.iterations

class cls_sampler(object):

  def __init__(self, cls_idx_dict):
    super(cls_sampler, self).__init__()
    self.cls_idx_dict = cls_idx_dict

  def __iter__(self):
    for cls, idx_lst in self.cls_idx_dict.items():
      yield idx_lst 

  def __len__(self):
    return len(self.cls_idx_dict.keys())

class random_sampler(object):

  def __init__(self, idx2cls, batch_size):
    super(random_sampler, self).__init__()
    self.idx = list(idx2cls.keys())
    self.bs  = batch_size

  def __iter__(self):
    random.shuffle(self.idx)
    batch = []
    for i in self.idx:
      batch.append(i)
      if len(batch) == self.bs:
        yield batch
        batch = []
    if len(batch) > 0:
      yield batch

  def __len__(self):
    return len(self.idx)

