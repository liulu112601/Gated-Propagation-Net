import time
import random
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
from configs import AverageMeter, obtain_accuracy, time_string
from models import euclidean_dist, update_acc, construct_propagation_graph, get_max_spanning_tree_kruskal, get_adj

def graph_nohier_test(mymodel, train_hierarchy_info, criterion, logger, dataloader, epoch, args, n_support, pp_running, n_hop):
  args = deepcopy(args)
  losses, acc1 = AverageMeter(), AverageMeter()
  acc_base, acc_prop = AverageMeter(), AverageMeter()
  data_time, batch_time, end  = AverageMeter(), AverageMeter(), time.time()
  num_device = len(mymodel.emb_model.device_ids)
  
  mymodel.emb_model.eval(); mymodel.propagation_net.eval()
  if mymodel.mst_net: mymodel.mst_net.eval()
  metaval_accuracies = []

  for batch_idx, (_, imgs, labels) in enumerate(dataloader):
    assert len(set(labels.tolist())) == args.classes_per_it_tr
    embs  = mymodel.emb_model(imgs)
    n_train_classes = len(train_hierarchy_info[0])
    target_classes  = list( set(labels.tolist()))
    test_pp_list = []
    grouped_s_idxs, grouped_q_idxs = [], []
    for cls in target_classes:
      all_idxs = (labels == cls).nonzero().view(-1).tolist()
      s_idx = all_idxs[:n_support]; q_idx = all_idxs[n_support:]
      grouped_s_idxs.append(torch.IntTensor(s_idx)); grouped_q_idxs.append(torch.IntTensor(q_idx))
      test_pp_list.append(torch.mean( embs[s_idx], dim=0))
    test_pp = torch.stack(test_pp_list, dim=0)
    # get nodes
    label2metalabel, forward_adj, backward_adj = construct_propagation_graph(target_classes, train_hierarchy_info, test_pp, pp_running[:n_train_classes], args.n_chi, args.n_hop, 5, 0.5)
    base_proto = pp_running[list( label2metalabel.keys() )] # input a list indices will create a clone
    for cls, pp in zip(target_classes, test_pp_list):
      base_proto[ label2metalabel[cls] ] = pp 
  
    if "mst" in args.training_strategy:
      features = base_proto
      forward_adj = forward_adj.cuda(); backward_adj = backward_adj.cuda()
      if mymodel.mst_net: 
        distance_forward  = mymodel.mst_net(features, forward_adj)
        distance_backward = mymodel.mst_net(features, backward_adj)
      else:
        node_num, feat_dim = features.size()
        q = features.view(1, node_num, feat_dim)
        k = features.view(node_num, 1, feat_dim)
        distance_forward = distance_backward = torch.norm(q - k, p='fro', dim=2)

      #print("[before mst]-foward_adj is {}".format(forward_adj))
      #print("[before mst]-backward_adj is {}".format(backward_adj))
      # get edges
      forward_adj  = get_max_spanning_tree_kruskal(forward_adj, distance_forward)
      backward_adj = get_max_spanning_tree_kruskal(backward_adj, distance_backward)
      #print("[after mst]-foward_adj is {}".format(forward_adj))
      #print("[after mst]-backward_adj is {}".format(backward_adj))
    # propagation 
    if "single-round" in args.training_strategy:
      adj_lst = [forward_adj for i in range(args.n_hop)] + [backward_adj for i in range(args.n_hop)]
    elif "multi-round" in args.training_strategy:
      adj_lst = []
      for i in range(args.n_hop):
        adj_lst.append(forward_adj)
        adj_lst.append(backward_adj)
    elif "allaround" in args.training_strategy:
      all_adj = forward_adj + backward_adj 
      adj_lst = [all_adj for i in range(args.n_hop)] 
    elif "only-forward" in args.training_strategy:
      adj_lst = [forward_adj for i in range(args.n_hop)]
    elif "only-backward" in args.training_strategy:
      adj_lst = [backward_adj for i in range(args.n_hop)]
    else: raise ValueError("invalid training_strategy for adj : {}".format(args.training_strategy))
    prop_proto  = mymodel.propagation_net(base_proto, adj_lst*num_device)
    target_prop_proto  = prop_proto[:len(target_classes)]
    target_base_proto  = base_proto[:len(target_classes)]
    if args.coef_base == -1 and args.coef_prop == -1:
      if epoch == -1:
        final_proto = target_prop_proto
    else:
      coef_norm = args.coef_base + args.coef_prop
      final_proto = target_base_proto * args.coef_base / coef_norm + target_prop_proto * args.coef_prop / coef_norm
    query_idxs  = torch.cat(grouped_q_idxs, dim=0).tolist()
    query_meta_labels  =  torch.LongTensor( [label2metalabel[labels[i].item()] for i in query_idxs] ).cuda(non_blocking=True)
    logits      = - euclidean_dist(embs[query_idxs], final_proto, transform=True).view(len(query_idxs), len(target_classes))
    loss        = criterion(logits, query_meta_labels)
    losses.update(loss.item(), len(query_idxs))

    top_fs       = obtain_accuracy(logits, query_meta_labels, (1,))
    acc1.update(top_fs[0].item(), len(query_idxs))
    update_acc([args.coef_base, args.coef_prop], [target_base_proto, target_prop_proto], [acc_base, acc_prop], embs[query_idxs], query_meta_labels)
    metaval_accuracies.append(top_fs[0].item())

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    if batch_idx + 1 == len(dataloader):
      metaval_accuracies = np.array(metaval_accuracies)
      stds = np.std(metaval_accuracies, 0)
      ci95 = 1.96*stds/np.sqrt(batch_idx + 1)
      logger.print("ci95 is : {:}".format(ci95))
      Tstring = 'TIME[{data_time.val:.2f} ({data_time.avg:.2f}) {batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(data_time=data_time, batch_time=batch_time)
      Sstring = '{:} {:} [Epoch={:03d}/{:03d}] [{:03d}/{:03d}]'.format(time_string(), "test", epoch, args.epochs, batch_idx, len(dataloader))
      Astring = 'loss=({:.3f}, {:.3f}), acc@1=({:.1f}, {:.1f}), acc@base=({:.1f}, {:.1f}), acc@prop=({:.1f}, {:.1f})'.format(losses.val, losses.avg, acc1.val, acc1.avg, acc_base.val, acc_base.avg, acc_prop.val, acc_prop.avg)
      Cstring = 'p_base_weigth : {:.4f}; p_prop_weight : {:.4f} '.format(args.coef_base, args.coef_prop)
      logger.print('{:} {:} {:} \n'.format(Sstring, Tstring, Astring))
  return losses, acc1, acc_base, acc_prop
