import os, sys, time, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from os import path as osp
import torch
import copy


def time_string():
  ISOTIMEFORMAT='%Y-%m-%d-%X'
  string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string


def check_graph_hier( parents, children, select_classes ):
  stop = 0
  for cls in parents.keys():
    if not cls in select_classes:
      continue
    if (parents[cls] == [] and children[cls] == []):
      stop = 1
      print("empty parents and children for cls {}".format(cls))
    if len( set(parents[cls]).intersection(set(children[cls])) ) > 0:
      stop = 1
      print( "overlap of parents and chidren for cls {}".format(cls) )

  assert parents.keys() == children.keys()

  for cls, par in parents.items():
    if not cls in select_classes:
      continue
    for p in par:
      if not cls in children[p]: stop = 1; print( "{} not in children[{}]".format(cls, p) )
  for cls, chil in children.items():
    if not cls in select_classes:
      continue
    for c in chil:
      if not cls in parents[c]: stop = 1; print( "{} not in parents[{}]".format(cls, c) ) 
  return stop


def check_neighbors (info, max_k):
  all_neighbors = [] 
  for k in range(1, max_k):
    all_neighbors.append( info['{}_neighbors'.format(k)] )
  nei_num = len(all_neighbors)
  for idx, nei in enumerate(all_neighbors):
    near_nei = all_neighbors[idx]
    for i in range(idx+1, nei_num):
      far_nei = all_neighbors[i] 
      assert near_nei.keys() == far_nei.keys()
      for cls, n_neis in near_nei.items():
        f_neis = far_nei[cls]
        assert len(f_neis) >= len(n_neis)
        assert set(f_neis).intersection(set(n_neis)) == set(n_neis)     


def prunning(select_classes, full_graph_classes, cls_to_parent, cls_to_child, info):
  # graph info; my dataset
  select_classes = list( set(select_classes))
  deselect_classes = list( set(full_graph_classes) -  set(select_classes) )
  assert set(deselect_classes).intersection(set(select_classes)) == set()
  print("construct real information for the whole graph done")
  graph_parents  = copy.deepcopy( cls_to_parent )
  graph_children = copy.deepcopy( cls_to_child )

  stop = check_graph_hier(graph_parents, graph_children, select_classes)
  if stop: import pdb; pdb.set_trace()

  # clean parents, children and the keys as the deselected classes
  for cls in sorted(deselect_classes):
    parents = copy.deepcopy( graph_parents[cls] )
    children = copy.deepcopy( graph_children[cls] )

    for p in parents:
      idx = graph_children[p].index(cls)
      graph_children[p].pop(idx)
      graph_children[p].extend(children)
    for c in children:
      idx = graph_parents[c].index(cls)
      graph_parents[c].pop(idx)
      graph_parents[c].extend(parents)
    stop = check_graph_hier(graph_parents, graph_children, select_classes)
    if stop: import pdb; pdb.set_trace()

  print("prunning step1 finish")
  select_graph_children = {}
  select_graph_parents  = {}
  for cls in select_classes:
    #print('original parents for cls {} : {}'.format(cls, graph_parents[cls]))
    #print('original children for cls {} : {}'.format(cls, graph_children[cls]))
    select_graph_children[cls] = copy.deepcopy( list( set( [c for c in graph_children[cls] if c in select_classes])))
    #print('select_graph_children for cls {} : {}'.format(cls, select_graph_children[cls]))
    select_graph_parents[cls]  = copy.deepcopy( list( set( [c for c in graph_parents[cls] if c in select_classes])))
    #print('select_graph_parents for cls {} : {}'.format(cls, select_graph_parents[cls]))
  stop = check_graph_hier(select_graph_parents, select_graph_children, select_classes)
  if stop: import pdb; pdb.set_trace()

  print("prunning step2 finish")
  info['graph_parents']  = select_graph_parents
  info['graph_children'] = select_graph_children

  return info 


def get_k_neighbours(info, max_k):
  parents = info['graph_parents']
  children = info['graph_children']
  for k in range(1, max_k):
    neighbors = defaultdict(list)
    classes = list( parents.keys() )
    assert parents.keys() == children.keys()
    for cls in classes:
      root_cls = [cls]
      for i in range(0, k):
        root_cls_tmp = []
        #print("root_cls : {} for nearest {} hop".format(root_cls, i))
        for c in root_cls:
          parents_lst  = parents[c]
          children_lst = children[c]
          neighbors[cls].extend(parents_lst+children_lst)
          root_cls_tmp.extend(parents_lst+children_lst)
        root_cls = list(set(root_cls_tmp))
      neighbors[cls] = list(set(neighbors[cls] ))
      #print("neighbors for cls {}: {}".format(cls, neighbors[cls]))
    info['{}_neighbors'.format(k)] = dict(neighbors)
  return info


def sample_img(cls_realcls, can_indexes, num_img_cls, wordid2idx, info, IDX_info):
  print("start sampling img ")
  cls_idxs = defaultdict(list)
  IDX_cls_idxs = defaultdict(list)
  imgidx_cls_realcls = {}
  IDX_imgidx_cls = {}
  classes = list( info['graph_parents'].keys() )
  assert 'root' in classes, 'Find root in classes : {:}'.format(classes)
  for cls in sorted(classes):
    realcls_lst = cls_realcls[cls]
    idx_lst = []
    for real in sorted(realcls_lst):
      img_idxs = random.sample( can_indexes[real], num_img_cls)
      idx_lst.extend(img_idxs)
      can_indexes[real] = sorted(list(set(can_indexes[real]).difference(set(img_idxs))))
      for idx in img_idxs:
        imgidx_cls_realcls[idx] = (cls, real)
        IDX_imgidx_cls[idx] = wordid2idx[cls]
    cls_idxs[cls] = list(set(idx_lst))
    IDX_cls_idxs[wordid2idx[cls]] = list(set(idx_lst))
  info['imgidx_cls_realcls'] = imgidx_cls_realcls
  info['cls_idxs'] = cls_idxs
  IDX_info['imgidx_cls'] = IDX_imgidx_cls
  IDX_info['cls_idxs'] = IDX_cls_idxs
  assert wordid2idx['root'] in IDX_info['cls_idxs'].keys()
  return info, IDX_info, can_indexes


def get_root_distance(info):
  # calculte the distance to "root" to decide which attention to use
  distance = {}
  parents  = info['graph_parents'] 
  classes  = sorted( list( parents.keys() ) )
  for cls in classes:
    if cls == 'root': distance[cls] = 0; continue
    dis = 0
    cur_classes = [cls]
    while not cur_classes == []:
      tmp = []
      for c in cur_classes:
        tmp.extend(parents[c])
      cur_classes = tmp
      dis += 1
      if "root" in cur_classes:
        break 
    distance[cls] = dis
  info['root_distance'] = distance
  return info


def idx_transformation(info, item_name, wordid2idx, IDX_info):
  idx_dict = {}
  wordid_dict = info[item_name]
  for key, item in wordid_dict.items():
    if isinstance(key, str): 
      key_idx  = wordid2idx[key]
    elif isinstance(key, int):
      key_idx  = key
    else: raise ValueError("invalid key {}".format(key))
    if isinstance(item, list):
      item_idx = [wordid2idx[i] for i in item]
    elif isinstance(item, int):
      item_idx = item
    else: raise ValueError("invalid item {}".format(item))
    idx_dict[key_idx] = item_idx
  IDX_info[item_name] = idx_dict
  return IDX_info


if __name__ == '__main__':
  assert len(sys.argv) == 2, 'invalid sys.argv : {:}'.format(sys.argv)
  dataset_save_dir = Path.home() / 'datasets' / sys.argv[1] 
  assert dataset_save_dir.exists(), '{:} does not exists'.format(dataset_save_dir)
  print ('dataset save dir : {:}'.format( dataset_save_dir ))
  # Prepare Raw-Data Cache File
  cache_file_name  = dataset_save_dir / 'cache-raw-data-tiered.pth'
  assert cache_file_name.exists()
  all_indexes = torch.load(cache_file_name)['all_indexes']

  # Prepare class information, collect leaf & coarse classes
  cache_file_name  = dataset_save_dir / 'cache-class2leaf-info.pth'
  assert cache_file_name.exists()
  print ('Try to load cache from {:}'.format( cache_file_name ))
  data = torch.load(cache_file_name)
  cls_realcls, cls_dis_info, cls_to_parent, cls_to_child = data['cls_realcls'], data['cls_dis_info'], data['cls_to_parent'], data['cls_to_child']

  cache_file_name  = dataset_save_dir / 'cache-train_test_class-info.pth'
  assert cache_file_name.exists()
  data = torch.load(cache_file_name)
  train_classes, test_classes = data['train_classes'], data['test_classes']
  
  # Set Random Seed
  torch.manual_seed(1111)
  random.seed(1111)

  wordid2idx = {}; wordid2idx['root'] = 0
  idx2wordid = {}; idx2wordid[0] = 'root'
  idx = 1
  for wordid in sorted(list(set(train_classes))):
    if wordid == 'root': continue
    wordid2idx[wordid] = idx
    idx2wordid[idx]    = wordid
    idx += 1
  add = len(wordid2idx)
  idx = 0
  for wordid in sorted(list(set(test_classes))):
    if wordid == 'root': continue
    wordid2idx[wordid]  = idx+add
    idx2wordid[idx+add] = wordid
    idx += 1

  torch.save(wordid2idx, dataset_save_dir / "wordid2idx.pth")
  torch.save(idx2wordid, dataset_save_dir / "idx2wordid.pth")

  train_info, test_info, all_info = {}, {}, {}
  full_graph_classes = sorted(list( cls_realcls.keys() ))
  ori_parent = copy.deepcopy(cls_to_parent)
  ori_child  = copy.deepcopy(cls_to_child)
  train_info = prunning(train_classes, full_graph_classes, cls_to_parent, cls_to_child, train_info)
  test_info  = prunning(test_classes, full_graph_classes, cls_to_parent, cls_to_child, test_info)
  all_info   = prunning(train_classes+test_classes, full_graph_classes, cls_to_parent, cls_to_child, all_info)

  if check_graph_hier(train_info['graph_parents'], train_info['graph_children'], train_classes): import pdb; pdb.set_trace()
  if check_graph_hier(test_info['graph_parents'] , test_info['graph_children'] , test_classes): import pdb; pdb.set_trace()
  if check_graph_hier(all_info['graph_parents']  , all_info['graph_children']  , train_classes+test_classes): import pdb; pdb.set_trace()

  max_k = 16
  train_info = get_k_neighbours(train_info, max_k)
  test_info  = get_k_neighbours(test_info, max_k)
  all_info   = get_k_neighbours(all_info, max_k)

  check_neighbors(train_info, max_k)
  check_neighbors(test_info, max_k)
  check_neighbors(all_info, max_k)

  train_info = get_root_distance(train_info)
  test_info  = get_root_distance(test_info)
  all_info   = get_root_distance(all_info)

  can_indexes = all_indexes.copy()
  num_img_cls = 20
  wordid2idx = torch.load(dataset_save_dir / "wordid2idx.pth")
  IDX_train_info, IDX_test_info, IDX_all_info = {}, {}, {}
  train_info, IDX_train_info, can = sample_img(cls_realcls, can_indexes, num_img_cls, wordid2idx, train_info, IDX_train_info)
  print("sample train img done")
  test_info, IDX_test_info, can   = sample_img(cls_realcls, can, num_img_cls, wordid2idx, test_info, IDX_test_info)
  print("sample test img done")
  
  torch.save(train_info, dataset_save_dir / "train_info.pth")
  torch.save(test_info , dataset_save_dir / "test_info.pth")
  torch.save(all_info  , dataset_save_dir / "all_info.pth")

  # transform wordid to idx for differnet mode
  for info, IDX_info in zip( [train_info, test_info, all_info], [IDX_train_info, IDX_test_info, IDX_all_info] ):
    IDX_info = idx_transformation(info, 'graph_children', wordid2idx, IDX_info)
    IDX_info = idx_transformation(info, 'graph_parents', wordid2idx, IDX_info)
    IDX_info = idx_transformation(info, 'root_distance', wordid2idx, IDX_info)
    for k in range(1, max_k):
      IDX_info = idx_transformation(info, '{}_neighbors'.format(k), wordid2idx, IDX_info)
  torch.save(IDX_train_info, dataset_save_dir / "IDX_train_info.pth")
  torch.save(IDX_test_info,  dataset_save_dir / "IDX_test_info.pth")
  torch.save(IDX_all_info,   dataset_save_dir / "IDX_all_info.pth")
  print( 'max test root distance', max( test_info['root_distance'].values() ))
  print( 'max train root distance', max( train_info['root_distance'].values() ))
  print( 'max all root distance', max( all_info['root_distance'].values() ))
