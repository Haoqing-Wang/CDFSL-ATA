# Cross-Domain Few-Shot Classification via Adversarial Task Augmentation
PyTorch implementation of:
<br>
[**Cross-Domain Few-Shot Classification via Adversarial Task Augmentation**](https://www.ijcai.org/proceedings/2021/0149.pdf)
<br>

Haoqing Wang, [Zhi-hong Deng](http://www.cis.pku.edu.cn/jzyg/szdw/dzh.htm)

IJCAI 2021

## Abstract

Few-shot classification aims to recognize unseen classes with few labeled samples from each class. Many meta-learning models for few-shot classification elaborately design various task-shared inductive bias (meta-knowledge) to solve such tasks, and achieve impressive performance. However, when there exists the domain shift between the training tasks and the test tasks, the obtained inductive bias fails to generalize across domains, which degrades the performance of the meta-learning models. In this work, we aim to improve the robustness of the inductive bias through task augmentation. Concretely, we consider the worst-case problem around the source task distribution, and propose the adversarial task augmentation method which can generate the inductive bias-adaptive 'challenging' tasks. Our method can be used as a simple plug-and-play module for various meta-learning models, and improve their cross-domain generalization capability. We conduct extensive experiments under the cross-domain setting, using nine few-shot classification datasets: mini-ImageNet, CUB, Cars, Places, Plantae, CropDiseases, EuroSAT, ISIC and ChestX. Experimental results show that our method can effectively improve the few-shot classification performance of the meta-learning models under domain shift, and outperforms the existing works.

## Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{ijcai2021-149,
  title     = {Cross-Domain Few-Shot Classification via Adversarial Task Augmentation},
  author    = {Wang, Haoqing and Deng, Zhi-Hong},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {1075--1081},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/149},
  url       = {https://doi.org/10.24963/ijcai.2021/149},
}
```

## Dependencies
* Python >= 3.5
* Pytorch >= 1.2.0 and torchvision (https://pytorch.org/)

## Datasets
We use miniImageNet as the single source domain, and use CUB, Cars, Places, Plantae, CropDiseases, EuroSAT, ISIC and ChestX as the target domains.

For miniImageNet, CUB, Cars, Places and Plantae, download and process them seperately with the following commands.
- Set `DATASET_NAME` to: `miniImagenet`, `cub`, `cars`, `places` or `plantae`.
```
cd filelists
python process.py DATASET_NAME
cd ..
```

For CropDiseases, EuroSAT, ISIC and ChestX, download them from

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018

* **CropDiseases**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data
    
    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

and put them under their respective paths, e.g., 'filelists/CropDiseases', 'filelists/EuroSAT', 'filelists/ISIC', 'filelists/chestX', then process them with following commands.
- Set `DATASET_NAME` to: `CropDiseases`, `EuroSAT`, `ISIC` or `chestX`.
```
cd filelists/DATASET_NAME
python write_DATASET_NAME_filelist.py
cd ..
```

## Pre-training
We adopt `baseline` pre-training from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) for all models.
- Download the pre-trained feature encoders from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot).
- Or train your own pre-trained feature encoder.
```
python pretrain.py --dataset miniImagenet --name Pretrain --train_aug
```

## Training

1.Train meta-learning models

Set `method` to `MatchingNet`, `RelationNet`, `ProtoNet`, `GNN` or `TPN`. For MatchingNet, RelationNet and TPN models, we set the training shot be 5 for both 1s and 5s evaluation.
```
python train.py --model ResNet10 --method GNN --n_shot 5 --name GNN_5s --train_aug
python train.py --model ResNet10 --method TPN --n_shot 5 --name TPN --train_aug
```

2.Train meta-learning models with feature-wise transformations.

Set `method` to `MatchingNet`, `RelationNet`, `ProtoNet`, `GNN` or `TPN`.
```
python train_FT.py --model ResNet10 --method GNN --n_shot 5 --name GNN_FWT_5s --train_aug
python train_FT.py --model ResNet10 --method TPN --n_shot 5 --name TPN_FWT --train_aug
```

3.Explanation-guided train meta-learning models.

Set `method` to `RelationNetLRP` or `GNNLRP`.
```
python train.py --model ResNet10 --method GNNLRP --n_shot 5 --name GNN_LRP_5s --train_aug
python train.py --model ResNet10 --method RelationNetLRP --n_shot 5 --name RelationNet_LRP --train_aug
```

4.Train meta-learning models with Adversarial Task Augmentation.

Set `method` to `MatchingNet`, `RelationNet`, `ProtoNet`, `GNN` or `TPN`.
```
python train_ATA.py --model ResNet10 --method GNN --max_lr 80. --T_max 5 --prob 0.5 --n_shot 5 --name GNN_ATA_5s --train_aug
python train_ATA.py --model ResNet10 --method TPN --max_lr 20. --T_max 5 --prob 0.6 --n_shot 5 --name TPN_ATA --train_aug
```
To get the results of the iteration goal without the regularization term, with the sample-wise Euclidean distance regularization term and with the maximum mean discrepancy (MMD) distance regularization term, run
```
python train_NR.py --model ResNet10 --method GNN --max_lr 80. --T_max 5 --n_shot 5 --name GNN_NR_5s --train_aug
python train_Euclid.py --model ResNet10 --method GNN --max_lr 40. --T_max 5 --lamb 1. --n_shot 5 --name GNN_Euclid_5s --train_aug
python train_MMD.py --model ResNet10 --method GNN --max_lr 80. --T_max 5 --lamb 1. --n_shot 5 --name GNN_MMD_5s --train_aug
```

## Evaluation and Fine-tuning

1.Test the trained model on the unseen domains.

- Specify the target dataset with `--dataset`: `cub`, `cars`, `places`, `plantae`, `CropDiseases`, `EuroSAT`, `ISIC` or `chestX`.
- Specify the saved model you want to evaluate with `--name`.
```
python test.py --dataset cub --n_shot 5 --model ResNet10 --method GNN --name GNN_ATA_5s
python test.py --dataset cub --n_shot 5 --model ResNet10 --method GNN --name GNN_LRP_5s
```

2.Fine-tuning with linear classifier.
To get the results of traditional pre-training and fine-tuning, run
```
python finetune.py --dataset cub --n_shot 5 --finetune_epoch 50 --model ResNet10 --name Pretrain
```

3.Fine-tuning the meta-learning models.
```
python finetune_ml.py --dataset cub --method GNN --n_shot 5 --finetune_epoch 50 --model ResNet10 --name GNN_ATA_5s
```

## Note
- This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot), [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot), [cdfsl-benchmark](https://github.com/IBM/cdfsl-benchmark), [few-shot-lrp-guided](https://github.com/SunJiamei/few-shot-lrp-guided) and [TPN-pytorch](https://github.com/csyanbin/TPN-pytorch).
- The dataset, model, and code are for non-commercial research purposes only.
- You only need a GPU with 11G memory for training and fine-tuning all models.
