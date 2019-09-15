#!/bin/bash
#bash scripts/train_aux.sh 0 5 1 multihead_cosinesim_softmax mst_allaround 

if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 5 parameters for the GPUs, N, K, prop_model, prop_strategy"
  exit 1 
fi
gpus=$1
lr=1e-3
batch_size=128
weight_decay=1e-5
classes_per_it_tr=$2
num_support_tr=$3
num_query_tr=$3
classes_per_it_val=$2
num_support_val=$3
num_query_val=$3
arch=resnet08_classifier
coef_base=-1
coef_prop=-1
k_nei=5
n_chi=2
training_strategy=$5
n_hop=2
reset_interval=3
keep_ratio=0.5
pretrain_path=None
prop_model=$4
fix_cnn_epoch=None
local_epoch=aux
n_heads=5
data_close=1-4
data_far=5-10
temp=1


CUDA_VISIBLE_DEVICES=${gpus} python ./exp/train_aux.py \
	       --prop_model ${prop_model} --n_heads ${n_heads} \
	       --dataset_root ${HOME}/datasets/graph-tiered-1-4/,${HOME}/datasets/graph-tiered-5-10/ --workers 16 --pretrain_path ${pretrain_path} \
	       --log_dir ./logs/PRETRAIN${data_close}-${data_far}${pretrain}${fix_cnn_epoch}-${local_epoch}-${classes_per_it_val}way${num_support_val}shot_${training_strategy}_base${coef_base}_prop${coef_prop}_${prop_model}_${k_nei}_n_chi${n_chi}_hop${n_hop}_keep${keep_ratio}_reset${reset_interval}/ --log_interval 20 --test_interval 500 \
               --arch ${arch} --coef_base ${coef_base} --coef_prop ${coef_prop} --k_nei ${k_nei} --n_chi ${n_chi} --training_strategy ${training_strategy} --n_hop ${n_hop} --reset_interval ${reset_interval} --keep_ratio ${keep_ratio} \
	       --temp ${temp} --epochs 3500 --start_decay_epoch 200 --local_epoch ${local_epoch} --iterations 100 --lr ${lr} --lr_step 10000 --lr_gamma 0.9 --weight_decay ${weight_decay} --batch_size ${batch_size} \
	       --classes_per_it_tr ${classes_per_it_tr} --num_support_tr ${num_support_tr} --num_query_tr ${num_query_tr} --classes_per_it_val ${classes_per_it_val} --num_support_val ${num_support_val}  --num_query_val ${num_query_val}
