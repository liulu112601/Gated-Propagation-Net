import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.propagationLayer import PropagationLayer, PropagationMLPLayer, MultiheadPropagationLayer

class PropagationNet(nn.Module):
  def __init__(self, input_dim, keep_ratio, n_hop):
    super(PropagationNet, self).__init__()
    self.propagation = [PropagationLayer(input_dim, keep_ratio) for i in range(2*n_hop)]
    for i, prop in enumerate(self.propagation):
      self.add_module('propagation_{}'.format(i), prop)

  def forward(self, features, adj_lst):
    feature_lst = []
    for prop_model, adj in zip(self.propagation, adj_lst):
      features = prop_model(features, adj)
      
    '''
      feature_lst.append(features)
    for i, fea in enumerate(feature_lst):
      for j in range(i+1, len(feature_lst)):
        if (feature_lst[i] == feature_lst[j]).all():
          print("feature in propagation step {} equals to step {}".format(i, j))
        else:
          print("feature in propagation step {} not equals to step {}".format(i, j))
    '''
    return features 


class PropagationNetSimRatio(nn.Module):
  def __init__(self, input_dim, keep_ratio, n_hop):
    super(PropagationNetSimRatio, self).__init__()
    self.propagation = [PropagationLayer(input_dim, keep_ratio) for i in range(2*n_hop)]
    for i, prop in enumerate(self.propagation):
      self.add_module('propagation_{}'.format(i), prop)

  def forward(self, features, adj_lst):
    feature_lst = []
    ori_features = features.clone()
    for prop_model, adj in zip(self.propagation, adj_lst):
      features = prop_model(features, adj, ori_features)
      
    '''
      feature_lst.append(features)
    for i, fea in enumerate(feature_lst):
      for j in range(i+1, len(feature_lst)):
        if (feature_lst[i] == feature_lst[j]).all():
          print("feature in propagation step {} equals to step {}".format(i, j))
        else:
          print("feature in propagation step {} not equals to step {}".format(i, j))
    '''
    return features 


class MultiheadPropagationNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_heads, n_hop, keep_ratio, dropout):
    super(MultiheadPropagationNet, self).__init__()
    self.dropout = dropout
    self.multihead_attention = []
    self.out_attention = []
    for i in range(n_hop*2):
      multihead_att = [MultiheadPropagationLayer(input_dim, hidden_dim, keep_ratio) for _ in range(n_heads)]
      self.multihead_attention.append( multihead_att )
      out_att       = MultiheadPropagationLayer(hidden_dim*n_heads, input_dim, keep_ratio)
      self.out_attention.append( out_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)
      self.add_module('out_attention_{}'.format(i), out_att)

  def forward(self, features, adj_lst):
    for multihead_att, out_att, adj in zip(self.multihead_attention, self.out_attention, adj_lst):
        x = F.dropout(features, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in multihead_att], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        features = F.elu(out_att(x, adj)) 
       
    return features

class MultiheadCosinePropagationNet(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads):
    super(MultiheadCosinePropagationNet, self).__init__()
    self.multihead_attention = []
    self.out_attention = []
    for i in range(n_hop*2):
      multihead_att =  [ PropagationLayer(input_dim, keep_ratio, hid_dim) for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)

  def forward(self, features, adj_lst):
    for multihead_att, adj in zip(self.multihead_attention, adj_lst):
      x = torch.stack([att(features, adj) for att in multihead_att], dim=2)
      features = torch.mean(x, dim=2)
    return features


class MultiheadCosinePropagationNetSimRatio(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads, temp, mst_att=None):
    super(MultiheadCosinePropagationNetSimRatio, self).__init__()
    self.mst_att = mst_att
    self.multihead_attention = []
    self.out_attention = []
    self.temp = temp
    for i in range(n_hop*2):
      multihead_att = [ PropagationLayer(input_dim, keep_ratio, hid_dim, temp, "cosine_sim") for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)

  def forward(self, features, adj_lst):
    ori_features = features.clone()
    for multihead_att, adj in zip(self.multihead_attention, adj_lst):
      x = torch.stack([att(features, adj, ori_features) for att in multihead_att], dim=2)
      features = torch.mean(x, dim=2)
    return features

class MultiheadCosinePropagationNetAttGate(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads, temp, mst_att=None):
    super(MultiheadCosinePropagationNetAttGate, self).__init__()
    self.mst_att = mst_att
    self.multihead_attention = []
    self.out_attention = []
    self.temp = temp
    for i in range(n_hop*2):
      multihead_att = [ PropagationLayer(input_dim, keep_ratio, hid_dim, temp, "cosine_att") for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)

  def forward(self, features, adj_lst):
    ori_features = features.clone()
    for multihead_att, adj in zip(self.multihead_attention, adj_lst):
      x = torch.stack([att(features, adj, ori_features) for att in multihead_att], dim=2)
      features = torch.mean(x, dim=2)
    return features

class MultiheadCosinePropagationNetSimSigmoidRatio(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads, temp, mst_att=None):
    super(MultiheadCosinePropagationNetSimSigmoidRatio, self).__init__()
    self.mst_att = mst_att
    self.multihead_attention = []
    self.out_attention = []
    self.temp = temp
    for i in range(n_hop*2):
      multihead_att = [ PropagationLayer(input_dim, keep_ratio, hid_dim, temp, "cosine_sigmoid") for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)

  def forward(self, features, adj_lst):
    ori_features = features.clone()
    for multihead_att, adj in zip(self.multihead_attention, adj_lst):
      x = torch.stack([att(features, adj, ori_features) for att in multihead_att], dim=2)
      features = torch.mean(x, dim=2)
    return features

class MultiheadCosinePropagationNetSimSoftmaxRatio(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads, temp, mst_att=None):
    super(MultiheadCosinePropagationNetSimSoftmaxRatio, self).__init__()
    self.mst_att = mst_att
    self.multihead_attention = []
    self.out_attention = []
    self.temp = temp
    for i in range(n_hop*2):
      multihead_att = [ PropagationLayer(input_dim, keep_ratio, hid_dim, temp, "cosine_softmax") for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)

  def forward(self, features, adj_lst):
    ori_features = features.clone()
    for multihead_att, adj in zip(self.multihead_attention, adj_lst):
      x = torch.stack([att(features, adj, ori_features) for att in multihead_att], dim=2)
      features = torch.mean(x, dim=2)
    return features

class MultiheadMLPPropagationNetSimSigmoidRatio(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads, temp, mst_att=None):
    super(MultiheadMLPPropagationNetSimSigmoidRatio, self).__init__()
    self.mst_att = mst_att
    self.multihead_attention = []
    self.out_attention = []
    self.temp = temp
    for i in range(n_hop*2):
      multihead_att = [ PropagationMLPLayer(input_dim, keep_ratio, hid_dim, temp, "cosine_sigmoid") for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)

  def forward(self, features, adj_lst):
    ori_features = features.clone()
    for multihead_att, adj in zip(self.multihead_attention, adj_lst):
      x = torch.stack([att(features, adj, ori_features) for att in multihead_att], dim=2)
      features = torch.mean(x, dim=2)
    return features

class PropagationNetParamFree(nn.Module):
  def __init__(self):
    super(PropagationNetParamFree, self).__init__()
   
  def forward(self, features, adj_lst):
    (n_node, fea_dim) = features.size()
    for adj in adj_lst:
      raw_weight = torch.ones(n_node, n_node).cuda()
      zero_vec   = -9e15*torch.ones_like(raw_weight)
      attention  = torch.where(adj > 0, raw_weight, zero_vec)
      attention  = F.softmax(attention, dim=1)
      prop_fea   = torch.matmul(attention, features)
      adj_num    = torch.sum(adj, dim=1)
      adj_mask   = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) ).float()
      features   = adj_mask.view(n_node,1) * prop_fea + (1-adj_mask).view(n_node, 1) * features
    return features 


'''
class MultiheadCosinePropagationNet(nn.Module):
  def __init__(self, input_dim, hid_dim, keep_ratio, n_hop, n_heads):
    super(MultiheadCosinePropagationNet, self).__init__()
    self.multihead_attention = []
    self.out_attention = []
    for i in range(n_hop*2):
      multihead_att =  [ PropagationLayer(input_dim, keep_ratio, hid_dim) for _ in range(n_heads)]   
      self.multihead_attention.append( multihead_att )
      out_att       = nn.Linear(n_heads * input_dim, input_dim)
      self.out_attention.append( out_att )
      for j, att in enumerate(multihead_att):
        self.add_module('multihead_attention_{}hop_{}head'.format(i, j), att)
      self.add_module('out_attention_{}'.format(i), out_att)

  def forward(self, features, adj_lst):
    for multihead_att, out_att, adj in zip(self.multihead_attention, self.out_attention, adj_lst):
      x = torch.cat([att(features, adj) for att in multihead_att], dim=1)
      features = out_att(x)
    return features
'''
