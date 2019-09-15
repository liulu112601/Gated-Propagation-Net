import os, sys, time, math, random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import cv2
import numpy as np
from os import path as osp
import pickle as pkl
import xml.etree.ElementTree as etree
import torch

random.seed(10)

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d-%X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

class ImageNetParser:
  def __init__(self, xml_path, is_remove_misc):
    self.nodes = {}
    xml_root = etree.parse(xml_path).getroot()
    if is_remove_misc: remove_misc(xml_root)
    self.populate_graph(xml_root)
    print ("Analysing graph...")
    if not self.is_tree():
      print ("Graph is not a Tree, nodes have multiple parents!!")

    # if not self.is_acyclic():
    #   print ("Graph has circular dependencies")
    print ("No circular dependencies found. Assuming DAG such that traversal is possible only from parent to child node")

  def __parse(self, node, node_id, parent_id):
    """DFS algorithm to populate the nodes"""
    # add node to self.nodes
    if node_id not in self.nodes:
      self.nodes[node_id] = {
        'id': node_id,
        'parent_ids': [] if parent_id is None else [parent_id],
        'child_ids': []
      }
    else:
      # looks like there are multiple parents
      self.nodes[node_id]['parent_ids'].append(parent_id)
    # recurse over children
    for child in node:
      if child.tag == 'synset':
        child_id = child.attrib['wnid']
        self.nodes[node_id]['child_ids'].append(child_id)
        self.__parse(child, child_id, node_id)

  def __traverse_along_path(self, node_id, follow='parent_ids', stop_at=None, dist=None, result=None):
    """
    returns a dictionary of all the ancestors(follow='parent_ids') or descendants(follow='child_ids)
    of the nodes in DAG along with its "shortest distance".
    """
    # initialisation of default argument because of Python's stupid single time initialisation of arguments
    if dist is None:
      dist = 0
    if result is None:
      result = {}

    self.check_if_valid_id(node_id)

    followed_nodes = self.nodes[node_id][follow]
    for fnode_id in followed_nodes:
      # the "min" essentially incorporates the shortest distance behavior in DAG
      result[fnode_id] = dist + 1 if fnode_id not in result else min(result[fnode_id], dist + 1)
      if node_id == stop_at:
        break
      else:
        self.__traverse_along_path(fnode_id, follow, stop_at, dist+1, result)

    return result

  def get_parent(self, node_id):
    parents = self.nodes[node_id]['parent_ids']
    return parents.copy()

  def get_child(self, node_id):
    childs = self.nodes[node_id]['child_ids']
    return childs.copy()

  def check_if_valid_id(self, node_id):
    if node_id not in self.nodes:
      raise KeyError("Invalid node_id: " + node_id)

  def populate_graph(self, xml_root):
    """
    Populates the internal data structure (self.nodes)
    :param xml_root: root node of the xml document
    """
    print ('Reading XML structure and populating the graph..')
    self.__parse(xml_root, 'root', None)

  def is_acyclic(self):
    """Check if the graph is cyclic"""
    for node_id in self.nodes:
      anc = self.get_ancestors(node_id)
      dec = self.get_descendants(node_id)
      # if the node is present in its ancestors or descendants, that indicates cyclic structure
      if len(list(set(anc) & set([node_id]))) > 0 or len(list(set(dec) & set([node_id]))) > 0:
        return False
    return True

  def is_tree(self):
    """Check if the graph is tree, that is each node has only one parent"""
    for node_id in self.nodes:
      if len(self.nodes[node_id]['parent_ids']) > 1:
        return False
    return True

  def get_ancestors(self, node_id):
    """returns list of wnids of all ancestor"""
    return self.__traverse_along_path(node_id, follow='parent_ids').keys()

  def get_descendants(self, node_id):
    """returns list of wnids of all descendants"""
    return self.__traverse_along_path(node_id, follow='child_ids').keys()

  def get_min_depth(self, node_id):
    """returns the depth from root of DAG
    Note: In case of multiple occurrence, minimum depth is returned
    """
    depths = self.__traverse_along_path(node_id, follow='parent_ids', stop_at='root')
    return depths['root'] if 'root' in depths else 0

  def get_max_depth(self, node):
    parents = self.get_parent( node )
    depth = 1
    while True:
      new_parents = []
      for p in parents:
        new_parents.extend( self.get_parent( p ) )
      parents = new_parents
      if parents == []: break
      depth += 1
    return depth

  def get_distance(self, node_id1, node_id2):
    # self.__traverse_along_path will take care of node_id1
    self.check_if_valid_id(node_id2)

    # Direct link means either node2 is in the ancestors of node1 or in the descendants of node1
    for follow_path in ['parent_ids', 'child_ids']:
      path = self.__traverse_along_path(node_id1, follow=follow_path, stop_at=node_id2)
      if node_id2 in path:
        return path[node_id2]
    raise ValueError('Lucy Ben Ben')


def get_str_wordid_dict(wordid_path, str_path):
  with open(osp.join(wordid_path, "synsets.txt"), 'r') as f:
    wordid_lst = f.readlines()
  with open(osp.join(str_path   , "class_names.txt")) as f:
    str_lst    = f.readlines()
  wordid_str_dict = dict()
  str_wordid_dict = dict()
  for index, item in enumerate(wordid_lst):
    wordid   = item[:-1]
    str_name = str_lst[index][:-1]
    wordid_str_dict[wordid] = str_name
    str_wordid_dict[str_name] = wordid
  return str_wordid_dict, wordid_str_dict 


def decompress(path, output):
  with open(output, 'rb') as f:
    u = pkl._Unpickler(f)
    u.encoding = 'latin1'
    array = u.load()
    #array = pkl.load(f)
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  print ('There are {:} images'.format( len(array) ))
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)


def read_cache(tiered_image_dir, mode):
  """Reads dataset from cached pkl file."""
  cache_path_images = osp.join(tiered_image_dir, '{:}_images.npz'.format(mode))
  cache_path_labels = osp.join(tiered_image_dir, '{:}_labels.pkl'.format(mode))

  # Decompress images.
  if not osp.exists(cache_path_images):
    png_pkl = cache_path_images[:-4] + '_png.pkl'
    assert osp.exists(png_pkl), '{:} does not exist'.format(png_pkl)
    decompress(cache_path_images, png_pkl)
  assert osp.exists(cache_path_labels) and osp.exists(cache_path_images)
  try:
    with open(cache_path_labels, "rb") as f:
      data = pkl.load(f, encoding='bytes')
      _label_specific = data[b"label_specific"]
      _label_general = data[b"label_general"]
      _label_specific_str = data[b"label_specific_str"]
      _label_specific_str = [str_name.decode("utf-8") for str_name in _label_specific_str]
      _label_general_str = data[b"label_general_str"]
  except:
    with open(cache_path_labels, "rb") as f:
      data = pkl.load(f)
      _label_specific = data["label_specific"]
      _label_general = data["label_general"]
      _label_specific_str = data["label_specific_str"]
      _label_general_str = data["label_general_str"]
  with np.load(cache_path_images, mmap_mode="r", encoding='latin1') as data:
    _images = data["images"]
  str_class_img_dict = defaultdict(list)
  for idx, label in enumerate(_label_specific):
    str_class = _label_specific_str[label]
    str_class_img_dict[str_class].append(_images[idx])
  str_class_img_dict = dict(str_class_img_dict)
  print ('images : {:}'.format(_images.shape))
  print ('label_specific : {:}, max={:}'.format(_label_specific.shape, _label_specific.max()))
  print ('label_general  : {:}, max={:}'.format(_label_general.shape , _label_general.max()))
  return _label_specific_str, str_class_img_dict


def str_img_to_wordid_img(img_str, str_wordid_dict):
  img_wordid  = {}
  for class_str, img in img_str.items():
    wordid = str_wordid_dict[class_str]
    img_wordid[wordid] = img
  return img_wordid


def prepare_data(dataset_root = Path.home() / 'datasets' / 'tiered-imagenet'):
  dataset_root = Path(str(dataset_root)).resolve()
  assert dataset_root.exists(), '{:} does not exists'.format( dataset_root )
  dataset_root = str(dataset_root)

  label_str_test, image_test   = read_cache(dataset_root, 'test')
  label_str_train, image_train = read_cache(dataset_root, 'train')
  label_str_val, image_val     = read_cache(dataset_root, 'val')
  str_wordid_dict, wordid_str_dict = get_str_wordid_dict(dataset_root, dataset_root)
  label_wordid_test   = [str_wordid_dict[str_name] for str_name in label_str_test]
  label_wordid_train  = [str_wordid_dict[str_name] for str_name in label_str_train]
  label_wordid_val    = [str_wordid_dict[str_name] for str_name in label_str_val]
  img_wordid_train    = str_img_to_wordid_img(image_train, str_wordid_dict) 
  img_wordid_val      = str_img_to_wordid_img(image_val, str_wordid_dict) 
  img_wordid_test     = str_img_to_wordid_img(image_test, str_wordid_dict) 
  # all things have been transformed to wordid and img pixels, no str img name
  all_classes = label_wordid_test + label_wordid_train + label_wordid_val 
  all_img_wordid = {}
  all_img_wordid.update(img_wordid_train)
  all_img_wordid.update(img_wordid_test)
  all_img_wordid.update(img_wordid_val)
  
  # build index
  all_indexes = defaultdict(list)
  all_images  = []
  for wordid, img_lst in all_img_wordid.items():
    for img in img_lst:
      all_images.append( img )
      all_indexes[wordid].append( len(all_images) - 1 )
  for wordid, indexes in all_indexes.items():
    print("[worid: {:}], {:} images, indexes range from {} to {}".format(wordid, len(indexes), min(indexes), max(indexes)))
  all_images = np.stack(all_images)
  return all_classes, dict(all_indexes), all_images



if __name__ == '__main__':
  assert len(sys.argv) == 4, 'invalid sys.argv : {:}'.format(sys.argv)
  this_dir         = Path(__file__).parent.resolve()
  print ('This dir : {:}'.format(this_dir))

  parser           = ImageNetParser(this_dir / 'structure_released.xml', False)
  dataset_save_dir = Path.home() / 'datasets' / sys.argv[1] 
  if not dataset_save_dir.exists(): dataset_save_dir.mkdir(exist_ok=True)
  print ('dataset save dir : {:}'.format( dataset_save_dir ))
  # Prepare Raw-Data Cache File
  cache_file_name  = dataset_save_dir / 'cache-raw-data-tiered.pth'
  if not cache_file_name.exists():
    os.system("cp ~/datasets/graph-tiered/cache-raw-data-tiered.pth {}".format(cache_file_name))
  print ('Try to load cache from {:}'.format( cache_file_name ))
  data = torch.load(cache_file_name)
  all_classes, all_indexes, all_images = data['all_classes'], data['all_indexes'], data['all_images']
  print("{:} tiered-imagenet has {:} classes".format(time_string(), len(all_classes)))

  # Prepare class information, collect leaf & coarse classes
  cache_file_name  = dataset_save_dir / 'cache-class2leaf-info.pth'
  if not cache_file_name.exists():
    os.system("cp ~/datasets/graph-tiered/cache-class2leaf-info.pth {}".format(cache_file_name))
  print ('Try to load cache from {:}'.format( cache_file_name ))
  data = torch.load(cache_file_name)
  cls_realcls, cls_dis_info   = data['cls_realcls'], data['cls_dis_info']
  cls_to_parent, cls_to_child = data['cls_to_parent'], data['cls_to_child']
  print('{:}\ncls_realcls has {:} keys'.format(cls_realcls, len(cls_realcls)))

  cache_file_name  = dataset_save_dir / 'cache-train_test_class-info.pth'
  min_dis_criterion = sys.argv[2]
  max_dis_criterion = sys.argv[3]
 
  if not cache_file_name.exists():
    os.system("cp ~/datasets/graph-tiered/cache-train_test_class-info.pth {}".format(cache_file_name))
  random.seed(111)
  print("load sampling cls results from {}".format(cache_file_name))
  data = torch.load(cache_file_name)
  train_classes = data['train_classes']
  train_ratio       = 0.8
  all_graph_classes = sorted( list(cls_realcls.keys()) )
  num_test_cls = len(all_graph_classes) * (1-train_ratio)
  print ('Create Train-Test Class with ratio : {:}'.format(train_ratio))
  test_classes, xiters = [], 0

  while True:
    add_to_train = False
    target_set = train_classes

    ok_classes = []
    for xcls in all_graph_classes:
      if len(target_set) == 0: xdis = 1e9
      else: xdis = min([ cls_dis_info[(xcls, _x)] for _x in target_set ])
      if int(min_dis_criterion) <= xdis <= int(max_dis_criterion):
        ok_classes.append(xcls)
    if len(ok_classes) == 0:
      print('Can not find suitable class, break after {:} iters!'.format(xiters))
      break
    
    select_cls = random.choice(ok_classes)
    all_graph_classes.pop( all_graph_classes.index(select_cls) )
    test_classes.append ( select_cls )
    xiters = xiters + 1
  for tecls in test_classes:
    assert tecls not in train_classes, '{:} in train classes'.format(tecls)
    dis = min([ cls_dis_info[(tecls, _x)] for _x in train_classes ])
    assert int(min_dis_criterion) <= dis  <= int(max_dis_criterion), '{:} not in distance range '.format(tecls)
  if not 'root' in train_classes: train_classes.append('root')
  if not 'root' in test_classes : test_classes.append('root')
  torch.save({'train_classes': train_classes,
              'test_classes' : test_classes}, cache_file_name)
  

  print("Num of train classes : {:}, num of test classes : {:}".format(len(train_classes), len(test_classes)))
  print("-" * 100)
