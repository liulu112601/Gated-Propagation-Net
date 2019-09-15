# Learning to Propagate for Graph Meta-Learning 

## Requirements
- Python       >=3.6  
- PyTorch      >=1.0  
- torchvision  >=0.2.2  
- tqdm         >=4.31.1  
- opencv       >=3.4.2  
- numpy        >=1.16.4  

## Project Structure
```
.
├──exp [Training GPN]  
│  └──train_aux.py  
├──lib [Library for configuration, dataset, mode and training strategy]  
│  └──configs  
│     ├──__init__.py [Import useful Classes and Functions in configs]  
│     ├──args.py [Parser for command-line options, arguments and sub-commands]  
│     ├──logger.py [Logger for recording experiments results]  
│     └──utils.py [A mix of useful utility Functions]  
├──datasets [Dataset and sampler]  
│  ├──__init__.py [Import useful Classes and Functions in datasets]  
│  └──graph_imagenet.py [Dataset Class and different sampling methods]  
├──models [Model Selection]
│  ├──__init__.py [Import useful models]
│  ├──cifarResnet.py [CNN]
│  ├──initialization.py [Parameter initialization]
│  ├──propagationNet.py [Propagation model]
│  ├──pp_buffer.py [Prototype buffer]
│  ├──utils.py [Model related utility functions]
│  └──layers [One step propagation layer]  
│     └──propagationLayer.py [Different propagation mechanism]  
├──training_strategy [Strategy selection]  
│  ├──__init__.py [Import useful strategies]
│  ├──graph_nohier_test.py [No hierarchy test setting]
│  └──graphtrain.py [Normal setting]
├──sample-tiered-imagenet [Dataset extraction]  
└──scripts [Scripts for running]  
```

## Dataset Extraction 

1. Download images: Please download `tiered-imagenet.tar` from [here](https://github.com/renmengye/few-shot-ssl-public#tieredimagenet), and extract it into `${HOME}/datasets/`. Therefore, there should be a directory : `${HOME}/datasets/tiered-imagenet/`.

2. Download class graph: Please download WordNet structure `structure_released.xml` from [here](http://www.image-net.org/api/xml/structure_released.xml), and put it into `${ROOT}/sample-tiered-imagenet/`

3. Compute some statistics for the classes, images and the graph. Sample a training set of classes based on these information:
`python sample-tiered-imagenet/build_data_dag.py graph-tiered`
We fixed the random seed as the one we used for the paper. There should be 773 training classes sampled and a directory stores the statistics about the graph: `${HOME}/datasets/graph-tiered/`

4. Sample a test set of classes with the minimal hops`${MIN-DIS}` and the maximal hops `${MAX-DIS}` to the training set:
`python sample-tiered-imagenet/build_classes_dif_distance.py graph-tiered-${MIN-DIS}-${MAX-DIS} ${MIN-DIS} ${MAX-DIS}`
For tieredImageNet-Close: `${MIN-DIS}=1` `${MAX-DIS}=4`. For tieredImageNet-Far: `${MIN-DIS}=5` `${MAX-DIS}=10`.
There should be 315 test classes sampled for tieredImageNet-Close and 26 test classes sampled for tieredImageNet-Far. The following directory should be created:`${HOME}/datasets/graph-tiered-${MIN-DIS}-${MAX-DIS}`.

5. Sample the images per class: `python sample-tiered-imagenet/pre_data_for_model.py graph-tiered`

6. Save images for PyTorch: `python sample-tiered-imagenet/save_imgs.py graph-tiered`, `python sample-tiered-imagenet/save_imgs_twoinone.py graph-tiered-1-4 graph-tiered-5-10 IMGS` 

## Parameters for training Gated Propagation Network (GPN)

We introduce selected parameters here. Please refer `lib/configs/args.py` for all parameters.

- `--classes_per_it_tr`  : Number of the ways(classes) for training tasks
- `--num_support_tr`     : Size of support(training) set for training tasks
- `--num_query_tr`       : Size of query(validation) set for training tasks
- `--classes_per_it_val` : Number of the ways(classes) for testing tasks
- `--num_support_val`    : Size of support(training) set for test tasks
- `--num_query_val`      : Size of query(validation) set for test tasks
- `--arch`               : Architecture of the backbone CNN
- `--prop_model`         : Architecture of the propagation model
- `--training_strategy`  : Training strategy
- `--sample`             : Subgraph sampling strategy
- `--n_hop`              : Number of propagation steps
- `--n_heads`            : Number of heads for attention-based propagation

## Experiments of GPN
Usage (train GPN and evaluate the trained GPN every `$test_interval$` epochs):
```
bash scripts/train_aux.sh ${GPU} ${N-WAY} ${K-SHOT} ${Propagation-Model} ${Propagation-Strategy}
```
An example script for 5-way-1shot experiment is `bash scripts/train_aux.sh 0 5 1 multihead_cosinesim_softmax mst_allaround`.

## Citation
If you find this project helpful, please consider to cite the following paper:
```
@inproceedings{liu2019GPN,
  title={Learning to Propagate for Graph Meta-Learning},
  author={Liu, Lu and Zhou, Tianyi and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle={ Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019}
}
```
