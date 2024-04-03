# Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images
Pytorch implementation for the multiple instance learning model described in the paper [Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images](https://arxiv.org/abs/2303.06873) (_CVPR 2023, selected as a highlight_).
![](vis_ibmil.png)

## Installation
a. Create a conda virtual environment and activate it.

```shell
conda create -n ibmil python=3.7 -y
conda activate ibmil
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Install other third-party libraries.

- Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).  
[Tutorial 1](https://openslide.org/) and [Tutorial 2 (Windows)](https://www.youtube.com/watch?v=0i75hfLlPsw).  




- Install faiss for clustering
  ```shell
  conda install faiss-gpu cudatoolkit=10.0 -c pytorch #For confounder generation, assuming CUDA=10.0
  ```

- For other packages, please refer to [dsmil](https://github.com/binli123/dsmil-wsi/blob/master/env.yml), [TransMIL](https://github.com/hrzhang1123/DTFD-MIL) and [DTFD-MIL](https://github.com/hrzhang1123/DTFD-MIL).

## Stage 1: Data pre-processing and computing features
Please refer to [dsmil](https://github.com/binli123/dsmil-wsi/) for these steps.
- Data pre-processing: Download the raw WSI data and Prepare the patches.
- Computing features: Train the  feature extractor and using the pre-trained feature extractor for instance-level features. Note that the default feature extractor is ResNet, which can be replaced by other networks, e.g., ViT and CTransPath. Download the MoCo v3 pretrained ViT and SRCL pretrained CTransPath from https://github.com/Xiyue-Wang/TransPath.
- The pre-computed features are released at [Baidu cloud](https://pan.baidu.com/share/init?surl=_b6XWs7LiHQfIAKVUOXqNg&pwd=2i9p).

## Stage 2: Training aggregator and generating confounder
The aggregator is firstly trained with bag-level labels end to end.

- For abmil and dsmil:
  ```
  python train_tcga.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model [abmil/dsmil]
  ```
- For TransMIL:
  ```
  python train_tcga_transmil.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model transmil
  ```
- For DTFD-MIL:
  ```
  python train_tcga_DTFD.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model DTFD
  ```
Confounder is then generated with pre-trained aggregator.

- For abmil, dsmil and TransMIL:
  ```
  python clustering.py --num_classes [according to your dataset] --dataset [C16/tcga] --feats_size [size of pre-computed features] --model [abmil/transmil/dsmil] --load_path [path of pre-trained aggregator]
  ```
- For DTFD-MIL:
  ```
  python clustering_DTFD.py --num_classes [according to your dataset] --dataset [C16/tcga] --feats_size [size of pre-computed features] --model DTFD --load_path [path of pre-trained aggregator]
  ```
An example with feature extractor of ImageNet-pretrained ResNet-18, MIL model of abmil, dataset of Camelyon16, load_path of `pretrained_weights/agg.pth`:
 ```
python train_tcga.py --num_classes 1 --dataset Camelyon16_Img_nor --agg no --feats_size 512 --model abmil
python clustering.py --num_classes 1 --dataset Camelyon16_Img_nor --feats_size 512 --model abmil --load_path pretrained_weights/agg.pth
```
## Stage 3: Interventional training
The proposed interventional training for MIL models.
- For abmil and dsmil:
  ```
  python train_tcga.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features]  --model [abmil/dsmil] --c_path [path of the generated confounders] (Interventional training is activated if `--c_path` is specified.)
  ```
- For TransMIL:
  ```
  python train_tcga_transmil.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model transmil --c_path [path of the generated confounders] (Interventional training is activated if `--c_path` is specified.)
  ```
- For DTFD-MIL:
  ```
  python train_tcga_DTFD.py --num_classes [according to your dataset] --dataset [C16/tcga] --agg no --feats_size [size of pre-computed features] --model DTFD --c_path [path of the generated confounders] (Interventional training is activated if `--c_path` is specified.)
  ```
An example with feature extractor of ImageNet-pretrained ResNet-18, MIL model of abmil, dataset of Camelyon16,  c_path of `datasets_deconf/Camelyon16_Img_nor/train_bag_cls_agnostic_feats_proto_8.npy`:
 ```
python train_tcga.py --num_classes 1 --dataset Camelyon16_Img_nor --agg no --feats_size 512   --model abmil --c_path datasets_deconf/Camelyon16_Img_nor/train_bag_cls_agnostic_feats_proto_8.npy
```

## Citing IBMIL
```
@inproceedings{lin2023interventional,
  title={Interventional bag multi-instance learning on whole-slide pathological images},
  author={Lin, Tiancheng and Yu, Zhimiao and Hu, Hongyu and Xu, Yi and Chen, Chang-Wen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19830--19839},
  year={2023}
}
```
