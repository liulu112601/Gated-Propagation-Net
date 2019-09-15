import torch
import torch.nn.functional as F
from configs import obtain_accuracy
from copy import deepcopy
import random

def euclidean_dist(x, y, transform=True):
  bs = x.shape[0]
  if transform:
    num_proto = y.shape[0]
    query_lst = []
    for i in range(bs):
      ext_query = x[i, :].repeat(num_proto, 1)
      query_lst.append(ext_query)
    x = torch.cat(query_lst, dim=0)
    y = y.repeat(bs, 1)
 
  return torch.pow(x - y, 2).sum(-1)

def update_acc(coef_lst, proto_lst, acc_lst, embs, meta_labels):
  n_query = len(embs)
  for coef, proto, acc in zip(coef_lst, proto_lst, acc_lst):
    if coef != 0:
      logits = - euclidean_dist(embs, proto, transform=True).view(n_query, len(proto))
      top_fs = obtain_accuracy(logits, meta_labels, (1,))
      acc.update(top_fs[0].item(), n_query)

def get_propagation_graph(target_classes, hierarchy_info, n_hop):
  parents, children = hierarchy_info
  assert parents.keys() == children.keys()
  propagation_graph = deepcopy( target_classes )
  current_cls       = deepcopy( target_classes )
  new_parents = {}
  new_children = {}
  for i in range(0, n_hop):
    tmp_current_cls = []
    for cls in current_cls:
      par = parents[cls]
      chi = children[cls]
      propagation_graph.extend(par)
      propagation_graph.extend(chi)
      tmp_current_cls.extend(par)
      tmp_current_cls.extend(chi)
    current_cls = deepcopy( list(set(tmp_current_cls)) )
  propagation_graph = list( set(propagation_graph))
  for cls in propagation_graph:  
    new_parents[cls]  = [c for c in parents[cls] if c in propagation_graph] 
    new_children[cls] = [c for c in children[cls] if c in propagation_graph]
  return list( set(propagation_graph)), new_parents, new_children

def get_part_propagation_graph(target_classes, hierarchy_info, n_hop, thre, ratio):
  parents, children = hierarchy_info
  assert parents.keys() == children.keys()
  propagation_graph = deepcopy( target_classes )
  current_cls       = deepcopy( target_classes )
  new_parents = {}
  new_children = {}
  for i in range(0, n_hop):
    tmp_current_cls = []
    for cls in current_cls:
      par = parents[cls]
      chi = children[cls]
      if len(par) > thre:
        par = random.sample(par, round(ratio*len(par)))
      if len(chi) > thre:
        chi = random.sample(chi, round(ratio*len(chi)))
      propagation_graph.extend(par)
      propagation_graph.extend(chi)
      tmp_current_cls.extend(par)
      tmp_current_cls.extend(chi)
    current_cls = deepcopy( list(set(tmp_current_cls)) )
  propagation_graph = list( set(propagation_graph))
  for cls in propagation_graph:  
    new_parents[cls]  = [c for c in parents[cls] if c in propagation_graph] 
    new_children[cls] = [c for c in children[cls] if c in propagation_graph]
  return list( set(propagation_graph)), new_parents, new_children

def get_real_propagation_graph(target_classes, hierarchy_info, n_hop, thre, ratio):
  if len(hierarchy_info) == 2:
    parents, children = hierarchy_info; mode = "train"
  else:
    parents, children, all_train_classes = hierarchy_info; mode = "test"
  assert parents.keys() == children.keys()
  propagation_graph = deepcopy( target_classes )
  current_cls       = deepcopy( target_classes )
  if mode == "test":
    parents, children = pruning(parents, children, target_classes+all_train_classes)
  new_parents  = {}
  new_children = {}
  for i in range(0, n_hop):
    tmp_current_cls = []
    for cls in current_cls:
      par = parents[cls]
      chi = children[cls]
      if len(par) > thre: par = random.sample(par, round(ratio*len(par)))
      if len(chi) > thre: chi = random.sample(chi, round(ratio*len(chi)))
      '''
      if 0 in par: par.remove(0)
      if 219 in par: par.remove(219)
      if 304 in par: par.remove(304)  
      if 0 in chi: chi.remove(0)
      if 219 in chi: chi.remove(219)
      if 304 in chi: chi.remove(304)  
      '''
      propagation_graph.extend(par)
      propagation_graph.extend(chi)
      tmp_current_cls.extend(par)
      tmp_current_cls.extend(chi)
    current_cls = deepcopy( list(set(tmp_current_cls)) )
  propagation_graph = list( set(propagation_graph))
  for cls in propagation_graph:  
    new_parents[cls]  = [c for c in parents[cls] if c in propagation_graph]
    new_children[cls] = [c for c in children[cls] if c in propagation_graph]
  label2metalabel = get_label2metalabel(target_classes, propagation_graph)
  forward_adj, backward_adj = get_adj(new_parents, new_children, label2metalabel)
  return label2metalabel, forward_adj, backward_adj

def pruning(cls_to_parents, cls_to_children, candidate_classes):
  graph_parents = deepcopy(cls_to_parents)
  graph_children = deepcopy(cls_to_children)
  assert graph_parents.keys() == graph_children.keys()
  deselect_classes = list( set(graph_parents.keys()).difference( set(candidate_classes)) )
  select_classes = list( set(graph_parents.keys()).intersection(set(candidate_classes)) )
  for cls in deselect_classes:
    parents  = deepcopy( graph_parents[cls] )
    children = deepcopy( graph_children[cls] )
    for p in parents:
      idx = graph_children[p].index(cls)
      graph_children[p].pop(idx)
      graph_children[p].extend(children)
    for c in children:
      idx = graph_parents[c].index(cls)
      graph_parents[c].pop(idx)
      graph_parents[c].extend(parents)
  new_parents = {}; new_children = {}
  for cls in select_classes:
    new_parents[cls]  = deepcopy( list( set( [c for c in graph_parents[cls] if c in select_classes] ) ) )
    new_children[cls] = deepcopy( list( set( [c for c in graph_children[cls] if c in select_classes] ) ) )
  return new_parents, new_children

def construct_propagation_graph(target_classes, hierarchy_info, test_pp, train_pp, n_nei, n_hop, thre, ratio):
  train_parents, train_children = deepcopy( hierarchy_info )
  assert train_parents.keys() == train_children.keys()
  propagation_graph = deepcopy( target_classes )
  current_cls = []
  new_parents = {}
  new_children = {}
  n_train, fea_dim = train_pp.shape
  n_test,  fea_dim = test_pp.shape
  test_pp  = test_pp.view(n_test, 1, fea_dim)
  train_pp = train_pp.view(1, n_train, fea_dim)
  similarity = - torch.pow(test_pp - train_pp, 2).sum(-1)
  for s, test_cls in zip(similarity, target_classes):
    top_sim, top_idx = torch.topk(s, n_nei)
    top_idx = [c.item() for c in top_idx]
    '''
    if 0 in top_idx: top_idx.remove(0); print("remove 0 in test")
    if 219 in top_idx: top_idx.remove(219); print("remove 219 in test")
    if 304 in top_idx: top_idx.remove(304); print("remove 304 in test")
    '''
    current_cls.extend(top_idx)
    for c in top_idx:
      train_parents[c].append( test_cls ) # link test classes and train graph by predicting children from the training classes
    new_parents[test_cls]  = []
    new_children[test_cls] = top_idx
    propagation_graph.extend(top_idx)
  for i in range(0, n_hop):
    tmp_current_cls = []
    for cls in current_cls:
      if not cls in target_classes:
        par = train_parents[cls]
        chi = train_children[cls]
        if len(par) > thre: par = random.sample(par, round(ratio*len(par)))
        if len(chi) > thre: chi = random.sample(chi, round(ratio*len(chi)))
        '''
        if 0 in par: par.remove(0)
        if 219 in par: par.remove(219)
        if 304 in par: par.remove(304)  
        if 0 in chi: chi.remove(0)
        if 219 in chi: chi.remove(219)
        if 304 in chi: chi.remove(304)  
        '''
        propagation_graph.extend(par)
        propagation_graph.extend(chi)
        tmp_current_cls.extend(par)
        tmp_current_cls.extend(chi)
    current_cls = deepcopy( list(set(tmp_current_cls)) )
  propagation_graph = list( set(propagation_graph))
  for cls in propagation_graph:  
    if not cls in target_classes:
      new_parents[cls]  = [c for c in train_parents[cls] if c in propagation_graph]
      new_children[cls] = [c for c in train_children[cls] if c in propagation_graph]
  label2metalabel = get_label2metalabel(target_classes, propagation_graph)
  forward_adj, backward_adj = get_adj(new_parents, new_children, label2metalabel)
  return label2metalabel, forward_adj, backward_adj

def get_max_spanning_tree(directed_adj, features):
  #directed_adj : a dag of the original graph, value in [0,1], for 1 in position i,j means node j points to node i
  #feature      : the feature of the nodes in the graph
  assert isinstance(directed_adj, torch.ByteTensor), 'The typs of directed_adj should be bool instead of {:}'.format(directed_adj.dtype)
  node_num, feat_dim = features.size()
  assert directed_adj.dim() == 2 and directed_adj.size(0) == node_num and directed_adj.size(1) == node_num, 'invalid shape of directed_adj : {:}'.format(directed_adj.shape)
  with torch.no_grad():
    # get similarity distance
    directed_adj, features = directed_adj.cpu(), features.cpu()
    q = features.view(1, node_num, feat_dim)
    k = features.view(node_num, 1, feat_dim)
    similarity = - torch.norm(q - k, p='fro', dim=2) # Find tree with the maximum similarity
    prime_edge = - similarity
    # compute
    new_adj = torch.zeros(node_num, node_num, dtype=torch.uint8)
    # init
    keep_dis = torch.zeros(node_num) + 1e9
    keep_dis[0] = - 1e9
    in_tree  = torch.zeros(node_num, dtype=torch.uint8)
    match_I  = torch.zeros(node_num, dtype=torch.int) - 1
    for i in range(node_num):
      add_node = -1
      for j in range(node_num):
        if in_tree[j] == 0 and (add_node==-1 or keep_dis[j] < keep_dis[add_node]):
          add_node = j
      assert add_node != -1, 'i={:}, can not find a node.'.format(i)
      #print ('i={:}, node={:}'.format(i, add_node))

      in_tree[add_node] = 1
      if i > 0:
        new_adj[match_I[add_node].item(), add_node] = 1
        new_adj[add_node, match_I[add_node].item()] = 1
        #print ('i={:}, node={:} - {:}'.format(i, add_node, match_I[add_node]))

      for j in range(node_num):
        if in_tree[j] == 0 and (directed_adj[j,add_node] or directed_adj[add_node,j]):
          keep_dis[j] = min(keep_dis[j], prime_edge[j,add_node])
          match_I[j] = add_node
  new_adj = new_adj * directed_adj
  return new_adj

def get_max_spanning_tree_kruskal(directed_adj, distance):
  assert isinstance(directed_adj, torch.ByteTensor) or isinstance(directed_adj, torch.cuda.ByteTensor), 'The typs of directed_adj should be bool instead of {:}'.format(directed_adj.dtype)
  node_num = distance.size(0)
  assert directed_adj.dim() == 2 and directed_adj.size(0) == node_num and directed_adj.size(1) == node_num, 'invalid shape of directed_adj : {:}'.format(directed_adj.shape)
  assert distance.dim() == 2 and distance.size(0) == node_num and distance.size(1) == node_num, 'invalid shape of distance : {:}'.format(distance)
  g = Graph(node_num)
  edge_nodes_lst = torch.nonzero(directed_adj).tolist()
  for edge_nodes in edge_nodes_lst:
    i, j = edge_nodes
    g.addEdge(i,j,distance[i,j])
  '''
  for i in range(node_num):
    for j in range(i+1, node_num):
      if directed_adj[i][j] + directed_adj[j][i] > 0 :
        g.addEdge(i,j,distance[i,j])
  print("time on adding edge 2 {}".format(time.time() - start))
  start = time.time()
  '''
  result = g.KruskalMST()  
  new_adj = torch.zeros_like(directed_adj)
  for (n1, n2, w) in result:
    new_adj[n1, n2] = 1
    new_adj[n2, n1] = 1
  new_adj = directed_adj * new_adj
  return new_adj


class Graph: 
  
  def __init__(self,vertices): 
    self.V= vertices # No. of vertices 
    self.graph = []  # default dictionary  
          
  # function to add an edge to graph 
  def addEdge(self,u,v,w): 
    self.graph.append([u,v,w]) 

  # A utility function to find set of an element i 
  def find(self, parent, i):
    if parent[i] == i: 
      return i 
    return self.find(parent, parent[i]) 
  
  # A function that does union of two sets of x and y 
  def union(self, parent, rank, x, y): 
    xroot = self.find(parent, x) 
    yroot = self.find(parent, y) 

    if rank[xroot] < rank[yroot]: 
      parent[xroot] = yroot 
    elif rank[xroot] > rank[yroot]: 
      parent[yroot] = xroot 

    else : 
      parent[yroot] = xroot 
      rank[xroot] += 1

  def KruskalMST(self): 

    result =[] #This will store the resultant MST 
    i = 0 # An index variable, used for sorted edges 
    e = 0 # An index variable, used for result[] 

    self.graph =  sorted(self.graph,key=lambda item: item[2]) 

    parent = list(range(self.V))
    rank   = [0] * self.V
    '''
    parent = [] ; rank = [] 
    # Create V subsets with single elements 
    for node in range(self.V): 
      parent.append(node) 
      rank.append(0) 
    '''
    # Number of edges to be taken is equal to V-1 
    while i < len(self.graph):

      u,v,w =  self.graph[i] 
      i = i + 1
      x = self.find(parent, u) 
      y = self.find(parent ,v) 

      if x != y: 
        e = e + 1     
        result.append([u,v,w]) 
        self.union(parent, rank, x, y)             
    return result
    

'''
def get_max_spanning_tree(directed_adj, feature):
  #directed_adj : a dag of the original graph, value in [0,1], for 1 in position i,j means node j points to node i 
  #feature      : the feature of the nodes in the graph
  INFINITY    = 1e6 
  # build the similarity matrix for between every two nodes
  (total_nodes, fea_dim) = feature.shape
  q = feature.repeat(total_nodes,1)
  k = feature.repeat(1,total_nodes).view(total_nodes*total_nodes, fea_dim)
  dis = - torch.pow(q - k, 2).sum(-1).view(total_nodes, total_nodes).cpu()
  assert (dis == torch.t(dis)).all()
  small_vec  = -INFINITY * torch.ones_like(dis)
  adj = torch.where((directed_adj + torch.t(directed_adj)) > 0, dis, small_vec)
  print("edge weight in Lucy's MST: {}".format(adj))
  # Below calculates the softmax weight
  #small_vec  = -INFINITY * torch.ones_like(dis)
  ## select the real linked nodes
  #similarity = torch.where(directed_adj > 0, dis, small_vec)
  #similarity = F.softmax(similarity, dim=1)
  ## since if no nodes point to a node i, the i-th line will be 1/num_line according to softmax, use a mask to set them to 0  
  #adj_num   = torch.sum(directed_adj, dim=1)
  #adj_mask  = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) )
  #similarity = similarity * adj_mask.view(-1,1)
  ## turn to undirected and use prim algorithm for mst
  #adj = similarity + torch.t(similarity)
  select_node = [0]
  for i in range(total_nodes):
    adj[i,i] = -INFINITY
  new_adj = torch.zeros_like(adj)
  mask_adj = torch.ones_like(adj) * (-INFINITY)
  while total_nodes > len(select_node):
    candidate_adj = mask_adj.clone()
    candidate_adj[select_node] = adj[select_node]
    max_idx  = torch.argmax(candidate_adj)
    max_line = max_idx / total_nodes
    max_col  = max_idx % total_nodes
    select_node.append(max_col.item())
    adj[select_node, max_col]  = -INFINITY
    adj[max_col, select_node]  = -INFINITY
    new_adj[max_line, max_col] = 1
    new_adj[max_col, max_line] = 1
  directed_new_adj   = new_adj * directed_adj
  return directed_new_adj 
'''

def get_adj(parents, children, label2metalabel):
  # For line i in the adj matrix, the 1 in this line indicates which class will propagate to class i
  # So the direction in the graph filled by 1 in (i, j) means j-->i
  assert len(parents) == len(children)
  num_cls = len(parents)
  with torch.no_grad():
    forward_adj  =  torch.zeros(num_cls, num_cls, dtype=torch.uint8)
    backward_adj =  torch.zeros(num_cls, num_cls, dtype=torch.uint8)
    for label, metalabel in label2metalabel.items():
      for p in parents[label]:
        meta_p = label2metalabel[p]
        forward_adj[metalabel][meta_p] = 1     
      for c in children[label]:
        meta_c = label2metalabel[c]
        backward_adj[metalabel][meta_c] = 1
  return forward_adj, backward_adj

def get_label2metalabel(target_classes, prop_graph):
  label2metalabel = {}
  for i, cls in enumerate(target_classes):
    label2metalabel[cls] = i
  add = 0
  for cls in prop_graph:
    if cls in target_classes:
      continue
    else:
      label2metalabel[cls] = len(target_classes) + add
      add += 1
  return label2metalabel 

def check_adj(forward_adj, backward_adj):
  adj = forward_adj + backward_adj
  adj = torch.sum(adj, dim=1)
  if (adj == 0).any(): raise TypeError("Invalid adj : {}".format(adj))   

'''
def get_propagation_adj(forward_adj, backward_adj):
  # n_hop * n_node * n_node
  forward_lst , backward_lst = [], [] 
  for idx, chi in enumerate(backward_adj): 
    forward = forward_adj.clone() * chi.view(-1,1) 
    backward = backward_adj.clone() * chi.clone()[idx].fill_(1)
    forward_lst.append(forward); backward_lst.append(backward)
  prop_forward_adj  = torch.stack(forward_lst, dim=0)
  prop_backward_adj = torch.stack(backward_lst, dim=0)
  return prop_forward_adj, prop_backward_adj
'''
