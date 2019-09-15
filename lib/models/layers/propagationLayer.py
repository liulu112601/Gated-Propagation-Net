import torch
import torch.nn as nn
import torch.nn.functional as F

def propagate(model, features, adj):
  (n_node, fea_dim) = features.size()
  q = model.linear_q( features.repeat(n_node,1) )
  k = model.linear_k( features.repeat(1, n_node).view(n_node*n_node,fea_dim) ) 
  raw_att = model.cosine(q,k)
  raw_att = raw_att.view(n_node, n_node)
  zero_vec  = -9e15*torch.ones_like(raw_att)
  attention = torch.where(adj > 0, raw_att, zero_vec)
  attention = attention * model.temp # in order to magnify the difference of the attention score
  attention = F.softmax(attention, dim=1)
  prop_fea  = torch.matmul(attention, features)
  adj_num   = torch.sum(adj, dim=1)
  adj_mask  = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) ).float()
  prop_fea  = adj_mask.view(n_node,1) * prop_fea + (1-adj_mask).view(n_node, 1) * features
  return prop_fea 

def propagate_mlp(model, features, adj):
  (n_node, fea_dim) = features.size()
  q = features.repeat(n_node,1)
  k = features.repeat(1, n_node).view(n_node*n_node,fea_dim)
  raw_att = model.w( model.w_1(k) + model.w_2(q))
  raw_att = raw_att.view(n_node, n_node)
  zero_vec  = -9e15*torch.ones_like(raw_att)
  attention = torch.where(adj > 0, raw_att, zero_vec)
  attention = attention * model.temp
  attention = F.softmax(attention, dim=1)
  prop_fea  = torch.matmul(attention, features)
  adj_num   = torch.sum(adj, dim=1)
  adj_mask  = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) ).float()
  prop_fea  = adj_mask.view(n_node,1) * prop_fea + (1-adj_mask).view(n_node, 1) * features
  return prop_fea  

def cosine_sim_gate(features, prop_fea, ori_features):
  fea_sim  = F.cosine_similarity(features, ori_features, dim=-1)
  prop_sim = F.cosine_similarity(prop_fea, ori_features, dim=-1)
  norm_sim = fea_sim + prop_sim
  fea_sim  = (fea_sim / norm_sim).view(-1, 1)
  prop_sim = (prop_sim / norm_sim).view(-1, 1)
  return prop_sim, fea_sim

def cosine_att_gate(gate_model,  features, prop_fea, ori_features):
  # two scores decide how much the propagation is: 1. prop feature score 2. current node feature score
  gate_linear_q, gate_linear_k, cosine = gate_model
  (n_node, fea_dim) = features.size()
  q_f = gate_linear_q( features )
  q_p = gate_linear_q( prop_fea )
  k   = gate_linear_k( ori_features )
  raw_q_f = cosine(q_f, k).view(-1,1)
  raw_q_p = cosine(q_p, k).view(-1,1)
  attention = F.softmax( torch.cat((raw_q_f, raw_q_p), dim=1), dim=1 )
  # TODO: the attention score are similar
  return attention[:, 1].view(-1,1), attention[:, 0].view(-1,1)
    
def cosine_sigmoid_gate(features, prop_fea, ori_features):
  fea_sim  = F.cosine_similarity(features, ori_features, dim=-1)
  prop_sim = F.cosine_similarity(prop_fea, ori_features, dim=-1)
  norm_sim = fea_sim + prop_sim
  fea_sim  = (fea_sim / norm_sim).view(-1, 1)
  prop_sim = (prop_sim / norm_sim).view(-1, 1)
  #fea_sim  = torch.sigmoid(fea_sim)
  #prop_sim = 1 - fea_sim
  #prop_sim = torch.sigmoid(prop_sim)
  #fea_sim  = 1 - prop_sim 
  #prop_sim = torch.sigmoid(prop_sim)
  #fea_sim  = torch.sigmoid(fea_sim)
  prop_sim = torch.sigmoid(prop_sim)
  fea_sim  = torch.sigmoid(fea_sim)
  
  sim = torch.cat([fea_sim, prop_sim], dim=1)
  sim = F.softmax(sim, dim=1)
  fea_sim  = sim[:, 0].view(-1,1)
  prop_sim = sim[:, 1].view(-1,1)

  return prop_sim, fea_sim 

def cosine_softmax_gate(features, prop_fea, ori_features, temp):
  fea_sim  = F.cosine_similarity(features, ori_features, dim=-1).view(-1,1)
  prop_sim = F.cosine_similarity(prop_fea, ori_features, dim=-1).view(-1,1)
  sim = torch.cat([fea_sim, prop_sim], dim=1)
  sim = F.softmax(temp * sim, dim=1)
  fea_sim  = sim[:, 0].view(-1,1)
  prop_sim = sim[:, 1].view(-1,1)
  return prop_sim, fea_sim


class PropagationLayer(nn.Module):
  def __init__(self, input_dim, keep_ratio, hid_dim=128, temp=1, gate=None):
    super(PropagationLayer, self).__init__()
    self.linear_q = nn.Linear(input_dim, hid_dim, bias=False)
    self.linear_k = nn.Linear(input_dim, hid_dim, bias=False)
    self.cosine   = nn.CosineSimilarity(dim=-1, eps=1e-4)
    self.keep_ratio = keep_ratio
    self.temp       = temp
    self.gate       = gate
    if self.gate == "cosine_att":
      self.gate_linear_q = nn.Linear(input_dim, hid_dim, bias=False)
      self.gate_linear_k = nn.Linear(input_dim, hid_dim, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

  def forward(self, features, adj, ori_features=None):
    prop_fea = propagate(self, features, adj)
    if   self.gate == "cosine_sim": 
      prop_weight, fea_weight = cosine_sim_gate(features, prop_fea, ori_features)
    elif self.gate == "cosine_att":
      gate_model = self.gate_linear_q, self.gate_linear_k, self.cosine
      prop_weight, fea_weight = cosine_att_gate(gate_model, features, prop_fea, ori_features)
    elif self.gate == "cosine_sigmoid":
      prop_weight, fea_weight = cosine_sigmoid_gate(features, prop_fea, ori_features)
    elif self.gate == "cosine_softmax":
      prop_weight, fea_weight = cosine_softmax_gate(features, prop_fea, ori_features, self.temp)
    else:
      prop_weight = (1- self.keep_ratio)
      fea_weight  = self.keep_ratio
    prop_fea = prop_weight * prop_fea + fea_weight * features
    return prop_fea

class PropagationMLPLayer(nn.Module):
  def __init__(self, input_dim, keep_ratio, hid_dim=128, temp=1, gate=None):
    super(PropagationMLPLayer, self).__init__()
    self.w_1 = nn.Linear(input_dim, hid_dim, bias=True)
    self.w_2 = nn.Linear(input_dim, hid_dim, bias=False)
    self.w   = nn.Linear(hid_dim, 1, bias=True)
    self.non_linear = nn.Tanh()
    self.keep_ratio = keep_ratio
    self.temp       = temp
    self.gate       = gate
    if self.gate == "cosine_att":
      self.gate_linear_q = nn.Linear(input_dim, hid_dim, bias=False)
      self.gate_linear_k = nn.Linear(input_dim, hid_dim, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

  def forward(self, features, adj, ori_features=None):
    prop_fea = propagate_mlp(self, features, adj)
    if   self.gate == "cosine_sim": 
      prop_weight, fea_weight = cosine_sim_gate(features, prop_fea, ori_features)
    elif self.gate == "cosine_att":
      gate_model = self.gate_linear_q, self.gate_linear_k, self.cosine
      prop_weight, fea_weight = cosine_att_gate(gate_model, features, prop_fea, ori_features)
    elif self.gate == "cosine_sigmoid":
      prop_weight, fea_weight = cosine_sigmoid_gate(features, prop_fea, ori_features)
    else:
      prop_weight = (1- self.keep_ratio)
      fea_weight  = self.keep_ratio
    prop_fea = prop_weight * prop_fea + fea_weight * features
    return prop_fea

  '''
  # The following is the implementation for inidividual path for every propagation map fot every node
  def forward(self, features, adj):
    (bs, n_node, fea_dim) = features.size()
    q = features.repeat(1,n_node,1)
    k = features.repeat(1,1, n_node).view(bs, n_node*n_node,fea_dim)
    raw_att = self.cosine(q,k)
    raw_att = raw_att.view(bs, n_node, n_node)
    zero_vec  = -9e15*torch.ones_like(raw_att)
    attention = torch.where(adj.repeat(bs,1,1) > 0, raw_att, zero_vec)
    attention = F.softmax(attention, dim=2)
    prop_fea  = torch.matmul(attention, features)
    adj_num   = torch.sum(adj, dim=1)
    adj_mask  = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) ).float()
    prop_fea  = adj_mask.view(1, n_node,1) * prop_fea + (1-adj_mask).view(1, n_node, 1) * features
    return prop_fea

    # non-parallel implementation
    prop_fea_lst = []
    for i in range(bs):
      feature   = features[i]
      q         = self.linear_q( feature.repeat(n_node,1) )
      k         = self.linear_k( feature.repeat(1,n_node).view(n_node*n_node, fea_dim) )
      raw_att   = self.cosine(q,k)
      raw_att   = raw_att.view(n_node, n_node)
      zero_vec  = -9e15*torch.ones_like(raw_att)
      attention = torch.where(adj > 0, raw_att, zero_vec)
      attention = F.softmax(attention, dim=1)
      prop_fea  = torch.matmul(attention, feature)
      adj_num   = torch.sum(adj, dim=1)
      adj_mask  = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) ).float()
      prop_fea  = adj_mask.view(-1,1) * prop_fea + (1-adj_mask).view(-1,1) * feature
      prop_fea_lst.append(prop_fea)
    prop_fea = torch.stack(prop_fea_lst, dim=0)
    print(prop_fea)
    import pdb; pdb.set_trace()
    return prop_fea
  '''

class MultiheadPropagationLayer(nn.Module):
  def __init__(self, in_features, out_features, keep_ratio, dropout=0.6, alpha=0.2, concat=True):
    super(MultiheadPropagationLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.out_features = out_features
    self.keep_ratio = keep_ratio
    self.alpha = alpha
    self.concat = concat

    self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
    nn.init.xavier_uniform_(self.W.data, gain=1.414)
    self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
    nn.init.xavier_uniform_(self.a.data, gain=1.414)
    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, input_feature, adj):
    h = torch.mm(input_feature, self.W)
    N = h.size(0)

    a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
    e = torch.matmul(a_input, self.a).squeeze(2)
    e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

    zero_vec  = -9e15*torch.ones_like(e)
    attention = torch.where(adj > 0, e, zero_vec)
    attention = F.softmax(attention, dim=1)
    attention = F.dropout(attention, self.dropout, training=self.training)
    h_prime   = torch.matmul(attention, h)
    adj_num   = torch.sum(adj, dim=1)
    adj_mask  = torch.where( adj_num>0, torch.ones_like(adj_num), torch.zeros_like(adj_num) ).float()
    n_node    = len(adj_num)
    h_prime   = adj_mask.view(n_node,1) * h_prime + (1-adj_mask).view(n_node, 1) * h
    h_prime   = (1- self.keep_ratio) * h_prime + self.keep_ratio * h

    if self.concat:
        return F.elu(h_prime)
        return h_prime
    else:
        return h_prime

