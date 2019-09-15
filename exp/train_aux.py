import os, sys, time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from copy import deepcopy
from collections import defaultdict, OrderedDict

lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from datasets import GraphImageNetDataset, FewShotGlobalSampler, FewShotLocalSampler, FewShotMixSampler, FewShotAuxSampler, cls_sampler 
from configs import get_parser, Logger, time_string, convert_secs2time, AverageMeter, obtain_accuracy
from training_strategy import graphtrain, graph_nohier_test
import models 

class MyModel(object):
  def __init__(self):
    self.emb_model       = None
    self.propagation_net = None
    self.mst_net         = None

  def parallel_submodels(self):
    if self.emb_model: self.emb_model = torch.nn.DataParallel(self.emb_model)
    if self.propagation_net: self.propagation_net = torch.nn.DataParallel(self.propagation_net)
    if self.mst_net: self.mst_net = torch.nn.DataParallel(self.mst_net)
  
  def cuda_submodels(self):
    if self.emb_model: self.emb_model = self.emb_model.cuda()
    if self.propagation_net: self.propagation_net = self.propagation_net.cuda()
    if self.mst_net: self.mst_net = self.mst_net.cuda()

  def get_all_params(self):
    params_all = []
    for submodel in [self.emb_model, self.propagation_net, self.mst_net]:
      if submodel: params_all += [p for p in submodel.parameters()]
    return params_all

  def get_prop_params(self):
    params_prop = []
    for submodel in [self.propagation_net, self.mst_net]:
      if submodel: params_prop += [p for p in submodel.parameters()]
    return params_prop

def train_model(lr_scheduler, mymodel, pp_buffer, criterion, optimizer, logger, dataloader, hierarchy_info, epoch, args, mode, n_support, n_hop):
  training_strategy = deepcopy(args.training_strategy)
  losses, acc1, acc_base, acc_prop = graphtrain(lr_scheduler, mymodel, criterion, optimizer, logger, dataloader, hierarchy_info, epoch, args, mode, n_support, pp_buffer, n_hop)
  return losses, acc1, acc_base, acc_prop



def run(args, mymodel, logger, criterion, optimizer, lr_scheduler, train_dataloader_, test_dataloader, hierarchy_info_train, hierarchy_info_test_close, hierarchy_info_test_far, train_pp_buffer):
  args = deepcopy(args)
  start_time = time.time()
  epoch_time = AverageMeter()
  best_acc_global_close, best_acc_nohier_global_close, best_acc_global_far, best_acc_nohier_global_far, arch = 0, 0, 0, 0, args.arch
  best_acc_local_close, best_acc_nohier_local_close, best_acc_local_far, best_acc_nohier_local_far = 0, 0, 0, 0
  model_best_path = '{:}/model_{:}_best.pth'.format(str(logger.baseline_classifier_dir), arch)
  model_best_nohier_path = '{:}/model_no_hier_{:}_best.pth'.format(str(logger.baseline_classifier_dir), arch)
  model_lst_path  = '{:}/model_{:}_lst.pth'.format(str(logger.baseline_classifier_dir), arch)

  train_dataloader_global, train_dataloader_local, train_dataloader_mix, train_dataloader_aux = train_dataloader_
  test_dataloader_global_close, test_dataloader_global_far, test_dataloader_local_close, test_dataloader_local_far = test_dataloader
  train_parents, train_children = hierarchy_info_train
  test_parents_close, _, all_train_classes_close = hierarchy_info_test_close
  test_parents_far, _, all_train_classes_far = hierarchy_info_test_far
  optimizer_all, optimizer_prop = optimizer
  lr_scheduler_all, lr_scheduler_prop = lr_scheduler 

  name_lst                = ["global-close", "global-far", "local-close", "local-far"]
  test_dataloader_lst     = [test_dataloader_global_close, test_dataloader_global_far, test_dataloader_local_close, test_dataloader_local_far]
  hierarchy_info_test_lst = [hierarchy_info_test_close, hierarchy_info_test_far, hierarchy_info_test_close, hierarchy_info_test_far]
  test_parents_lst        = [test_parents_close, test_parents_far, test_parents_close, test_parents_far]
  best_acc_lst            = [best_acc_global_close, best_acc_global_far, best_acc_local_close, best_acc_local_far]
  best_acc_nohier_lst     = [best_acc_nohier_global_close, best_acc_nohier_global_far, best_acc_nohier_local_close, best_acc_nohier_local_far]

  if os.path.isfile(model_lst_path):
    checkpoint  = torch.load(model_lst_path)
    start_epoch = checkpoint['epoch'] + 1
    mymodel.emb_model.load_state_dict(checkpoint['emb_state_dict'])
    mymodel.propagation_net.load_state_dict(checkpoint['propagation_net_state_dict'])
    if mymodel.mst_net: mymodel.mst_net.load_state_dict(checkpoint['mst_net_state_dict'])
    train_pp_buffer.load_state_dict(checkpoint['train_pp_buffer'])
    optimizer_all.load_state_dict(checkpoint['optimizer_all'])
    optimizer_prop.load_state_dict(checkpoint['optimizer_prop'])
    lr_scheduler_all.load_state_dict(checkpoint['scheduler_all'])
    lr_scheduler_prop.load_state_dict(checkpoint['scheduler_prop'])
    logger.print ('load checkpoint from {:}'.format(model_lst_path))
  else:
    start_epoch = 0

  for iepoch in range(start_epoch, args.epochs):

    time_str = convert_secs2time(epoch_time.val * (args.epochs- iepoch), True)
    logger.print ('Train {:04d} / {:04d} Epoch, [LR-ALL={:6.4f} ~ {:6.4f}, LR-PROP={:6.4f} ~ {:6.4f}], {:}'.format(iepoch, args.epochs, min(lr_scheduler_all.get_lr()), max(lr_scheduler_all.get_lr()), min(lr_scheduler_prop.get_lr()), max(lr_scheduler_prop.get_lr()), time_str))
    
    if iepoch >= args.start_decay_epoch:
      lr_sch = [lr_scheduler_all, lr_scheduler_prop]
    else:
      lr_sch = None
    '''
    if iepoch <= args.fix_cnn_epoch:
      logger.print("prop optimizer")
      optimizer = optimizer_prop
    else:
      logger.print("all optimizer")
      optimizer = optimizer_all
    '''
    optimizer = optimizer_all
    idx_iter = args.iterations * iepoch
    train_dataloader_aux.batch_sampler.idx_iter = idx_iter
    train_dataloader = train_dataloader_aux
        
    train_loss, acc1, acc_base, acc_prop = train_model(lr_sch, mymodel, train_pp_buffer.pp_running, criterion, optimizer, logger, train_dataloader, hierarchy_info_train, iepoch, args, 'train', args.num_support_tr, args.n_hop)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

    info = {'epoch'           : iepoch,
            'args'            : deepcopy(args),
            'finish'          : iepoch+1==args.epochs,
            'emb_state_dict'  : mymodel.emb_model.state_dict(),
            'propagation_net_state_dict': mymodel.propagation_net.state_dict(),
            'train_pp_buffer' : train_pp_buffer.state_dict(),
            'optimizer_all'   : optimizer_all.state_dict(),
            'optimizer_prop'  : optimizer_prop.state_dict(),
            'scheduler_all'   : lr_scheduler_all.state_dict(),
            'scheduler_prop'  : lr_scheduler_prop.state_dict(),
            }
    if mymodel.mst_net: info['mst_net_state_dict'] = mymodel.mst_net.state_dict()
    try:
      torch.save(info, model_lst_path)
      logger.print(' -->> joint-arch :: save into {:}\n'.format(model_lst_path))
    except PermmisionError:
      print("unsucceful write log") 


    with torch.no_grad():
      if iepoch % args.test_interval == 0 or iepoch == args.epochs -1 :
        for name, test_dataloader, hierarchy_info_test, test_parents, best_acc, best_acc_nohier in zip(name_lst, test_dataloader_lst, hierarchy_info_test_lst, test_parents_lst, best_acc_lst, best_acc_nohier_lst):
          logger.print('---------init_test_pp for {}-------------'.format(name))
          with torch.no_grad():
            # write from train_pp to test_pp
            train_copy = train_pp_buffer.pp_running.clone()
            test_only_pp = torch.rand(max(test_parents)-max(train_parents), train_copy.shape[-1]).cuda()
            test_pp = torch.cat((train_copy, test_only_pp), dim=0)

            test_loss, test_acc1, test_acc_base, test_acc_prop = train_model(None, mymodel, test_pp, criterion, None, logger, test_dataloader, hierarchy_info_test, -1, args, 'test', args.num_support_val, args.n_hop)
            logger.print ('[{}-HIERARCHY]-Epoch: {:04d} / {:04d} || Train-Loss: {:.4f} Train-Acc: {:.3f} || Test-Loss: {:.4f} Test-Acc1: {:.3f}, Test-Acc_base: {:.3f}, Test-Acc_prop: {:.3f}\n'.format(name, iepoch, args.epochs, train_loss.avg, acc1.avg, test_loss.avg, test_acc1.avg, test_acc_base.avg, test_acc_prop.avg))
            if test_acc1.avg >= best_acc:
              try:
                torch.save(info, model_best_path)
              except PermissionError: pass
              best_acc = test_acc1.avg

            test_loss, test_acc1, test_acc_base, test_acc_prop  = graph_nohier_test(mymodel, hierarchy_info_train, criterion, logger, test_dataloader, -1, args, args.num_support_val, test_pp, args.n_hop)
            logger.print ('[{}-NO-HIERARCHY]-Epoch: {:04d} / {:04d} || Train-Loss: {:.4f} Train-Acc: {:.3f} || Test-Loss: {:.4f} Test-Acc1: {:.3f}, Test-Acc_base: {:.3f}, Test-Acc_prop: {:.3f}\n'.format(name, iepoch, args.epochs, train_loss.avg, acc1.avg, test_loss.avg, test_acc1.avg, test_acc_base.avg, test_acc_prop.avg))
            if test_acc1.avg >= best_acc_nohier:
              try:
                torch.save(info, model_best_nohier_path)
              except PermissionError: pass
              best_acc_nohier = test_acc1.avg

    if iepoch % args.reset_interval == 0:
      logger.print("reset training buffers, the running mean prototypes are updated according to the concurrent CNN weight ....")
      train_pp_buffer.reset_buffer(mymodel.emb_model)

  with torch.no_grad():
    best_checkpoint = torch.load(model_best_path)
    mymodel.emb_model.load_state_dict(best_checkpoint['emb_state_dict'])
    mymodel.propagation_net.load_state_dict(best_checkpoint['propagation_net_state_dict'])
    if mymodel.mst_net: mymodel.mst_net.load_state_dict(best_checkpoint['mst_net_state_dict'])
    train_pp_buffer.load_state_dict( best_checkpoint['train_pp_buffer'] )
    for name, test_dataloader, hierarchy_info_test, test_parents, best_acc, best_acc_nohier in zip(name_lst, test_dataloader_lst, hierarchy_info_test_lst, test_parents_lst, best_acc_lst, best_acc_nohier_lst):
      logger.print("-----------init_test_pp for best {}, the best result in record : [HIERARCHY]{} ----------".format(name, best_acc))
      train_copy = train_pp_buffer.pp_running.clone()
      test_only_pp = torch.rand(max(test_parents)-max(train_parents), train_copy.shape[-1]).cuda()
      test_pp = torch.cat((train_copy, test_only_pp), dim=0)
      for i in range(11):
        args.coef_base = 0.1 * i; args.coef_prop = 1- args.coef_base
        logger.print ('*[{}-TEST-Best-base{}-prop{}]*'.format(name, args.coef_base, args.coef_prop))
        test_loss, test_acc1, test_acc_base, test_acc_prop = train_model(None, mymodel, test_pp, criterion, None, logger, test_dataloader, hierarchy_info_test, -1, args, 'test', args.num_support_val, args.n_hop)

    best_checkpoint = torch.load(model_best_nohier_path)
    mymodel.emb_model.load_state_dict(best_checkpoint['emb_state_dict'])
    mymodel.propagation_net.load_state_dict(best_checkpoint['propagation_net_state_dict'])
    if mymodel.mst_net: mymodel.mst_net.load_state_dict(best_checkpoint['mst_net_state_dict'])
    train_pp_buffer.load_state_dict( best_checkpoint['train_pp_buffer'] )
    for name, test_dataloader, hierarchy_info_test, test_parents, best_acc, best_acc_nohier in zip(name_lst, test_dataloader_lst, hierarchy_info_test_lst, test_parents_lst, best_acc_lst, best_acc_nohier_lst):
      logger.print("-----------init_test_pp for best nohier {}, the best result in record is  [NO-H]{} ----------".format(name, best_acc))
      train_copy = train_pp_buffer.pp_running.clone()
      test_only_pp = torch.rand(max(test_parents)-max(train_parents), train_copy.shape[-1]).cuda()
      test_pp = torch.cat((train_copy, test_only_pp), dim=0)
      for i in range(11):
        args.coef_base = 0.1 * i; args.coef_prop = 1- args.coef_base
        logger.print('*[{}-TEST-Best-Nohier-base-{}-prop-{}]*'.format(name, args.coef_base, args.coef_prop))
        test_loss, test_acc1, test_acc_base, test_acc_prop  = graph_nohier_test(mymodel, hierarchy_info_train, criterion, logger, test_dataloader, -1, args, args.num_support_val, test_pp, args.n_hop)


def main():
  args = get_parser()
  # create logger
  if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
  logger = Logger(args.log_dir, args.manual_seed)
  logger.print ("args :\n{:}".format(args))

  assert torch.cuda.is_available(), 'You must have at least one GPU'

  # set random seed
  torch.backends.cudnn.benchmark = True
  np.random.seed(args.manual_seed)
  torch.manual_seed(args.manual_seed)
  torch.cuda.manual_seed(args.manual_seed)

  # create dataloader
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_transform  = transforms.Compose([transforms.Resize(150), transforms.RandomResizedCrop(112), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
  [dataset_close, dataset_far] = args.dataset_root.split(",")
  train_dataset    = GraphImageNetDataset(dataset_close, 'train', train_transform)
  test_transform   = transforms.Compose([transforms.Resize(150), transforms.CenterCrop(112), transforms.ToTensor(), normalize])
  test_dataset_close = GraphImageNetDataset(dataset_close, 'test', test_transform)
  test_dataset_far   = GraphImageNetDataset(dataset_far, 'test', test_transform)

  train_sampler_global   = FewShotGlobalSampler(list(train_dataset.cls2idx.keys()), train_dataset.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, args.iterations )
  train_sampler_local    = FewShotLocalSampler(list(train_dataset.cls2idx.keys()), train_dataset.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, args.iterations, train_dataset.info['{}_neighbors'.format(args.k_nei)])
  train_sampler_mix      = FewShotMixSampler(list(train_dataset.cls2idx.keys()), train_dataset.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, args.iterations, train_dataset.info['{}_neighbors'.format(args.k_nei)])
  train_sampler_aux      = FewShotAuxSampler(list(train_dataset.cls2idx.keys()), train_dataset.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, args.iterations, train_dataset.info['{}_neighbors'.format(args.k_nei)], args.iterations * args.epochs, args.batch_size, train_dataset.idx2cls)

  test_sampler_global_close = FewShotGlobalSampler(list(test_dataset_close.cls2idx.keys()), test_dataset_close.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, 600 )
  test_sampler_global_far   = FewShotGlobalSampler(list(test_dataset_far.cls2idx.keys()), test_dataset_far.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, 600 )
  test_sampler_local_close  = FewShotLocalSampler(list(test_dataset_close.cls2idx.keys()), test_dataset_close.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, 600, test_dataset_close.info['{}_neighbors'.format(args.k_nei)] )
  test_sampler_local_far    = FewShotLocalSampler(list(test_dataset_far.cls2idx.keys()), test_dataset_far.cls2idx, args.classes_per_it_tr, args.num_support_tr + args.num_query_tr, 600, test_dataset_far.info['{}_neighbors'.format(args.k_nei)] )

  train_dataloader_global = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler_global, num_workers=args.workers)
  train_dataloader_local  = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler_local, num_workers=args.workers)
  train_dataloader_mix    = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler_mix, num_workers=args.workers)
  train_dataloader_aux    = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler_aux, num_workers=args.workers)
  train_dataloader_ = [train_dataloader_global, train_dataloader_local, train_dataloader_mix, train_dataloader_aux]

  test_dataloader_global_close  = torch.utils.data.DataLoader(test_dataset_close, batch_sampler=test_sampler_global_close, num_workers=args.workers)
  test_dataloader_global_far    = torch.utils.data.DataLoader(test_dataset_far, batch_sampler=test_sampler_global_far, num_workers=args.workers)
  test_dataloader_local_close   = torch.utils.data.DataLoader(test_dataset_close, batch_sampler=test_sampler_local_close, num_workers=args.workers)
  test_dataloader_local_far     = torch.utils.data.DataLoader(test_dataset_far, batch_sampler=test_sampler_local_far, num_workers=args.workers)
  test_dataloader = [test_dataloader_global_close, test_dataloader_global_far, test_dataloader_local_close, test_dataloader_local_far]

  # children include both training and testing classes children info
  hierarchy_info_train   = [train_dataset.parents, train_dataset.children]  
  all_info_close = torch.load( os.path.join( dataset_close, "IDX_all_info.pth") )
  hierarchy_info_test_close = [all_info_close['graph_parents'], all_info_close['graph_children'], list(train_dataset.cls2idx.keys())]
  all_info_far = torch.load( os.path.join( dataset_far, "IDX_all_info.pth") )
  hierarchy_info_test_far = [all_info_far['graph_parents'], all_info_far['graph_children'], list(train_dataset.cls2idx.keys())]
  # create model
  mymodel = MyModel()

  mymodel.emb_model = models.EmbeddingModels[args.arch]()
  if args.arch == "resnet18":
    fea_dim = 512
  elif args.arch == "resnet08":
    fea_dim = 576 
  elif args.arch == "resnet08_classifier":
    fea_dim = 576
    mymodel.emb_model.use_classifier = False
  elif args.arch == "simpleCnn":
    fea_dim = 3136 
  else: raise ValueError("invalid arch {}".format(args.arch))

  if args.prop_model == "cosine":
    mymodel.propagation_net = models.PropagationNet(fea_dim, args.keep_ratio, args.n_hop)
  elif args.prop_model == "cosinesim":
    mymodel.propagation_net = models.PropagationNetSimRatio(fea_dim, args.keep_ratio, args.n_hop)
  elif args.prop_model == "multihead_cosine":
    mymodel.propagation_net = models.MultiheadCosinePropagationNet(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads)
  elif args.prop_model == "multihead_cosinesim":
    mymodel.propagation_net = models.MultiheadCosinePropagationNetSimRatio(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp)
  elif args.prop_model == "multihead_cosinesim_sigmoid":
    mymodel.propagation_net = models.MultiheadCosinePropagationNetSimSigmoidRatio(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp)
  elif args.prop_model == "multihead_cosinesim_softmax":
    mymodel.propagation_net = models.MultiheadCosinePropagationNetSimSoftmaxRatio(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp)
  elif args.prop_model == "multihead_mlpsim_sigmoid":
    mymodel.propagation_net = models.MultiheadMLPPropagationNetSimSigmoidRatio(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp) 
  elif args.prop_model == "multihead_mlp":
    mymodel.propagation_net = models.MultiheadPropagationNet(fea_dim, fea_dim, args.n_heads, args.n_hop,  args.keep_ratio, 0.6)
  elif args.prop_model == "multihead_cosinesim_mstatt":
    mymodel.propagation_net = models.MultiheadCosinePropagationNetSimRatio(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp)
    mymodel.mst_net = models.Attention_cosine(fea_dim)
  elif args.prop_model == "mstatt":
    mymodel.propagation_net = models.PropagationNetParamFree()
    mymodel.mst_net = models.Attention_cosine(fea_dim)
  elif args.prop_model == "multihead_attgate":
    mymodel.propagation_net = models.MultiheadCosinePropagationNetAttGate(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp)
  elif args.prop_model == "multihead_attgate_mstatt":
    mymodel.propagation_net = models.MultiheadCosinePropagationNetAttGate(fea_dim, 128, args.keep_ratio, args.n_hop, args.n_heads, temp=args.temp)
    mymodel.mst_net = models.Attention_cosine(fea_dim)
    
  else: raise ValueError("invalid prop_model : {}".format(args.prop_model))
  
  mymodel.parallel_submodels()
  mymodel.cuda_submodels()

  train_pp_buffer = models.pp_buffer(train_dataset, fea_dim).cuda()
  logger.print ("emb_model:::\n{:}".format(mymodel.emb_model))
  logger.print ("For hop {:}, propagation_net:::\n{:}".format(args.n_hop, mymodel.propagation_net))
  logger.print ("mst_net:::\n{:}".format(mymodel.mst_net))
  logger.print ("train_pp_buffer:::\n{:}".format(train_pp_buffer))
  logger.print ("pp_running:::\n{:}".format(train_pp_buffer.pp_running))
  criterion = nn.CrossEntropyLoss().cuda()

  params_all = mymodel.get_all_params()
  optimizer_all     = torch.optim.Adam(params_all, lr=args.lr, weight_decay=args.weight_decay)
  lr_scheduler_all  = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_all, gamma=args.lr_gamma, step_size=args.lr_step)
  params_prop = mymodel.get_prop_params()
  optimizer_prop    = torch.optim.Adam(params_prop, lr=args.lr, weight_decay=args.weight_decay)
  lr_scheduler_prop = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_prop, gamma=args.lr_gamma, step_size=args.lr_step)
  optimizer    = [optimizer_all, optimizer_prop]
  lr_scheduler = [lr_scheduler_all, lr_scheduler_prop]

  run(args, mymodel, logger, criterion, optimizer, lr_scheduler, train_dataloader_, test_dataloader, hierarchy_info_train, hierarchy_info_test_close, hierarchy_info_test_far, train_pp_buffer)


if __name__ == '__main__':
  main()
