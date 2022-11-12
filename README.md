<div align="center">

# Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation

[![Conference](https://img.shields.io/badge/CVPR-2022-blue)](https://openaccess.thecvf.com/content/CVPR2022/papers/Phan_Class_Similarity_Weighted_Knowledge_Distillation_for_Continual_Semantic_Segmentation_CVPR_2022_paper.pdf)
[![Youtube](https://img.shields.io/badge/Youtube-link-red)](https://www.youtube.com/watch?v=QIV9gQq5VdE&t=14s)

</div>


![Vizualization on VOC 15-1](images/visualization_results.png)


This repository contains all of our code. It is a modified version of
[Cermelli et al.'s repository](https://github.com/fcdl94/MiB).


# Requirements

You need to install the following libraries:
- Python (3.6)
- Pytorch (1.8.1+cu102)
- torchvision (0.9.1+cu102)
- tensorboardX (1.8)
- apex (0.1)
- matplotlib (3.3.1)
- numpy (1.17.2)
- [inplace-abn](https://github.com/mapillary/inplace_abn) (1.0.7)

Note also that apex seems to only work with some CUDA versions, therefore try to install Pytorch (and torchvision) with
the 10.2 CUDA version. You'll probably need anaconda instead of pip in that case, sorry! Do:

```
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
cd apex
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


# Dataset

Two scripts are available to download ADE20k and Pascal-VOC 2012, please see in the `data` folder.
For Cityscapes, you need to do it yourself, because you have to ask "permission" to the holders; but be
reassured, it's only a formality, you can get the link in a few days by mail.

![Performance on VOC and ADE](images/results.png)


# How to perform training
The most important file is run.py, that is in charge to start the training or test procedure.
To run it, simpy use the following command:

> python -m torch.distributed.launch --nproc_per_node=\<num_GPUs\> run.py --data_root \<data_folder\> --name \<exp_name\> .. other args ..

The default is to use a pretraining for the backbone used, that is searched in the pretrained folder of the project.
We used the pretrained model released by the authors of In-place ABN (as said in the paper), that can be found here:
 [link](https://github.com/mapillary/inplace_abn#training-on-imagenet-1k).

Since the pretrained are made on multiple-gpus, they contain a prefix "module." in each key of the network. Please, be sure to remove them to be compatible with this code (simply rename them using key = key\[7:\]) (if you're working on single gpu).
If you don't want to use pretrained, please use --no-pretrained.

There are many options (you can see them all by using --help option), but we arranged the code to being straightforward to test the reported methods.
Leaving all the default parameters, you can replicate the experiments by setting the following options.
- please specify the data folder using: --data_root \<data_root\>
- dataset: --dataset voc (Pascal-VOC 2012) | ade (ADE20K)
- task: --task \<task\>, where tasks are
    - 15-5, 15-5s, 19-1 (VOC), 100-50, 100-10, 50, 100-50b, 100-10b, 50b (ADE, b indicates the order)
- step (each step is run separately): --step \<N\>, where N is the step number, starting from 0
- (only for Pascal-VOC) disjoint is default setup, to enable overlapped: --overlapped
- learning rate: --lr 0.01 (for step 0) | 0.001 (for step > 0)
- batch size: --batch_size \<24/num_GPUs\>
- epochs: --epochs 30 (Pascal-VOC 2012) | 60 (ADE20K)
- method: --method \<method name\>, where names are
    - FT, LWF, LWF-MC, ILT, EWC, RW, PI, MIB, REMINDER

For all details please follow the information provided using the help option.

#### Example commands

LwF on the 100-50 setting of ADE20K, step 0:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 12 --dataset ade --name LWF --task 100-50 --step 0 --lr 0.01 --epochs 60 --method LWF

MIB on the 50b setting of ADE20K, step 2:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 12 --dataset ade --name MIB --task 100-50 --step 2 --lr 0.001 --epochs 60 --method MIB

PLOP on 15-1 overlapped setting of VOC, step 1:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 16 --dataset voc --name PLOP --task 15-5s --overlapped --step 1 --lr 0.001 --epochs 30 --method PLOP

REMINDER on 15-1 overlapped setting of REMINDER, step 1:
> python -m torch.distributed.launch --nproc_per_node=2 run.py --data_root data --batch_size 16 --dataset voc --name REMINDER --task 15-5s --overlapped --step 1 --lr 0.001 --epochs 30 --method REMINDER

Once you trained the model, you can see the result on tensorboard (we perform the test after the whole training)
 or you can test it by using the same script and parameters but using the command
> --test

that will skip all the training procedure and test the model on test data.

Or more simply you can use one of the provided script that will launch every step of a continual training.

For example, do

````
bash scripts/voc/reminder_15-1.sh
````

Note that you will need to modify those scripts to include the path to your data folder.

## Update on Table 5 in our CVPR paper
We find out more a more optimal set of hyper-parameters when doing ablation study on distillation loss. The updated version of Table 5 is shown below 

|   Distillation loss  | 0-15 | 16-20 |  all |
|-----|:------:|:-------:|:------:|
| CSW-KD   | 68.30 | 27.23 | 58.52 |
| Normal KD| 59.88 | 25.43 | 51.68 |
| UNKD | 65.09 | 21.53 | 54.72 |

```
@inproceedings{phan2022reminder,
  title={Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation},
  authors={Phan, Minh Hieu and Ta, The-Anh and Phung, Son Lam and Tran-Thanh, Long and Bouzerdoum, Abdesselam},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
