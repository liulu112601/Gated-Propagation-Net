import random, argparse
import math

base = math.exp(12)

def get_parser():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset_root',       type=str,                           help='path to dataset')
    parser.add_argument('--iterations',         type=int,                           help='few-shot sampler: number of episodes per epoch, default=100')
    parser.add_argument('--epochs',             type=int,                           help='number of epochs to train for')
    parser.add_argument('--start_decay_epoch',  type=int,                           help='number of epochs to train for')
    parser.add_argument('--fix_cnn_epoch',      type=int,                           help='epochs to fix cnn models')
    parser.add_argument('--local_epoch',        type=str,                           help='epochs for local sampler: alter / int / None')
    parser.add_argument('--workers',            type=int,                            help='number of workers for dataloader')
    parser.add_argument('--manual_seed',        type=int,                           help='input for the manual seeds initializations')
    parser.add_argument('--fix_point',          type=int,                           help='input for the manual seeds initializations')
    # base arch
    parser.add_argument('--temp',               type=float,                   help='')
    parser.add_argument('--arch',               type=str,                           help='the architecture of base model')
    parser.add_argument('--prop_model',         type=str,                           help='the architecture of propagation model')
    parser.add_argument('--coef_base',          type=float,                   help='')
    parser.add_argument('--coef_prop',          type=float,                   help='')
    parser.add_argument('--keep_ratio',         type=float,                   help='the keep ratio of the original feature in propagation')
    parser.add_argument('--n_heads',            type=int,                            help='number of heads in multi-head attention')
    parser.add_argument('--n_hop',              type=int,                            help='number of hops')
    parser.add_argument('--k_nei',              type=int,                            help='k neighbors for data generation')
    parser.add_argument('--n_chi',              type=int,                            help='n closest neighbors as children for no hierarchy test')
    parser.add_argument('--sample',             type=str,                            help='global / local')
    parser.add_argument('--training_strategy',  type=str,                           help='graph_train')
    parser.add_argument('--pretrain_path',      type=str,                           help='pretrainedcnn model')
    # log
    parser.add_argument('--log_dir',            type=str,                            help='where to store logs')
    parser.add_argument('--log_interval',       type=int,                            help='number of batches to print log')
    parser.add_argument('--test_interval',      type=int,                           help='number of batches to print log')
    parser.add_argument('--reset_interval',     type=int,                           help='number of batches to print log')
    parser.add_argument('--lr',                 type=float,                         help='learning rate for the coarse classifier')
    parser.add_argument('--lr_step',         type=int,   help='StepLR learning rate scheduler step, default=20')
    parser.add_argument('--lr_gamma',    type=float,                         help='StepLR learning rate scheduler gamma, default=0.5')
    parser.add_argument('--momentum',           type=float, default=0.9,            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay',       type=float, default=1e-4,           help='SGD weight decay (default: 1e-4)')
    parser.add_argument('--batch_size',       type=int,   default=128,             help='batch-size for coarse classifier')
    # training fine
    parser.add_argument('--classes_per_it_tr',  type=int,                           help='number of random classes per episode for training, default=60')
    parser.add_argument('--num_support_tr',     type=int,                           help='number of samples per class to use as support for training, default=5')
    parser.add_argument('--num_query_tr',       type=int,                           help='number of samples per class to use as query for training, default=5')
    parser.add_argument('--classes_per_it_val', type=int,                           help='number of random classes per episode for validation, default=5')
    parser.add_argument('--num_support_val',    type=int,                           help='number of samples per class to use as support for validation, default=5') 
    parser.add_argument('--num_query_val',      type=int,                           help='number of samples per class to use as query for validation, default=15')

    args = parser.parse_args()
    if args.manual_seed is None:
      args.manual_seed = random.randint(1, 10000)
    return args
