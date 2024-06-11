# Linguistic-Aware Patch Slimming Framework for Fine-grained Cross-Modal Alignment

The official codes for our paper "[Linguistic-Aware Patch Slimming Framework for Fine-grained Cross-Modal Alignment](https://openaccess.thecvf.com/content/CVPR2024/html/Fu_Linguistic-Aware_Patch_Slimming_Framework_for_Fine-grained_Cross-Modal_Alignment_CVPR_2024_paper.html)", which is accepted by the  IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2024.
We referred to the implementations of [VSE++](https://github.com/fartashf/vsepp), [SCAN](https://github.com/kuanghuei/SCAN), [GPO](https://github.com/woodfrog/vse_infty), and [HREM](https://github.com/CrossmodalGroup/HREM) to build up the repository. 


## Introduction
Cross-modal alignment aims to build a bridge connecting vision and language. 
It is an important multi-modal task that efficiently learns the semantic similarities between images and texts.
Traditional fine-grained alignment methods heavily rely on pre-trained object detectors to extract region features for subsequent region-word alignment, thereby incurring substantial computational costs for region detection and error propagation issues for two-stage training.

<div align=center>
<img src="imgs/fig1-1.jpg" width="80%">
</div>

In this paper, we focus on the mainstream vision transformer, incorporating patch features for patch-word alignment, while addressing the resultant issue of visual patch redundancy and patch ambiguity for semantic alignment.
We propose a novel Linguistic-Aware Patch Slimming (LAPS) framework for fine-grained alignment, 
which explicitly identifies redundant visual patches with language supervision and rectifies their semantic and spatial information to facilitate more effective and consistent patch-word alignment.
Extensive experiments on various evaluation benchmarks and model backbones show LAPS outperforms the state-of-the-art fine-grained alignment methods.

<div align=center>
<img src="imgs/fig1-2.jpg" width="100%">
</div>


## Preparation

### Environments
We recommended the following dependencies:
- python >= 3.8
- torch >= 1.12.0
- torchvision >= 0.13.0
- transformers >=4.32.0
- opencv-python
- tensorboard


### Datasets

We have prepared the caption files for two datasets in  `data/` folder, hence you just need to download the images of the datasets. 
The Flickr30K (f30k) images can be downloaded in [flickr30k-images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). The MSCOCO (coco) images can be downloaded in [train2014](http://images.cocodataset.org/zips/train2014.zip), and [val2014](http://images.cocodataset.org/zips/val2014.zip).
We hope that the final data are organized as follows:


```
data
├── coco  # coco captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── testall_ids.txt
│   ├── testall_caps.txt
│   └── id_mapping.json
│
├── f30k  # f30k captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── test_ids.txt
│   ├── test_caps.txt
│   └── id_mapping.json
│
├── flickr30k-images # f30k images
│
├── coco-images # coco images
│   ├── train2014
│   └── val2014
```


### Model Weights

Our framework needs to get the pre-trained weights for [BERT-base](https://huggingface.co/bert-base-uncased), [ViT-base](https://huggingface.co/google/vit-base-patch16-224-in21k), and [Swin-base](https://huggingface.co/microsoft/swin-base-patch4-window7-224) models. 
You also can choose the weights downloaded by [transformers](https://github.com/huggingface/transformers) automatically (the weights will be downloaded at  `~/.cache`).


## Training
First, we set up the **arguments**, detailed information about the arguments is shown in ```arguments.py```.

- `--dataset`: the chosen datasets, e.g., `f30k` and `coco`.
- `--data_path`: the root path of datasets, e.g., `data/`.
- `--multi_gpu`: whether to use the multiple GPUs (DDP) to train the models.
- `--gpu-id`, the chosen GPU number, e.g., 0-7.
- `--logger_name`, the path of logger files, e.g., `runs/f30k_test` or `runs/coco_test`


Then, we run the ```train.py``` for model training. 
The models need about 20,000 GPU-Memory (one 3090 GPU) when batch size = 64 and about 40,000 GPU-Memory (one A40 GPU) when batch size = 108.
You need to modify the batch size according to the hardware conditions, and we also support the **multiple GPUs** training. 
Besides, considering the GPU-memory limitation, we don't integrate the Gumbel-softmax sampling for the patch selection in the repository. 
The performances are not affected much but GPU-memory is reduced a lot (see more details in the paper).

```
## single GPU

### vit + f30k 
python train.py --dataset f30k --gpu-id 0 --logger_name runs/f30k_vit --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

### swin + f30k
python train.py --dataset f30k --gpu-id 0 --logger_name runs/f30k_swin --batch_size 64 --vit_type swin  --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6

### vit + coco 
python train.py --dataset coco --gpu-id 0 --logger_name runs/coco_vit --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

### swin + coco
python train.py --dataset coco --gpu-id 0 --logger_name runs/coco_swin --batch_size 64 --vit_type swin  --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6


## multiple GPUs

### vit + f30k
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py --dataset f30k --multi_gpu 1 --logger_name runs/f30k_vit --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

### swin + f30k
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 train.py --dataset f30k --multi_gpu 1 --logger_name runs/f30k_swin --batch_size 64 --vit_type swin --embed_size 1024 --sparse_ratio 0.8 --aggr_ratio 0.6


### vit + coco
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py --dataset coco --multi_gpu 1 --logger_name runs/coco_vit --batch_size 64 --vit_type vit --embed_size 512 --sparse_ratio 0.5 --aggr_ratio 0.4

### swin + coco
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node=3 train.py --dataset coco --multi_gpu 1 --logger_name runs/coco_swin --batch_size 72 --vit_type swin --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py --dataset coco --multi_gpu 1 --logger_name runs/coco_swin --batch_size 64 --vit_type swin --embed_size 512 --sparse_ratio 0.8 --aggr_ratio 0.6
```

## Evaluation
Run ```eval.py``` to evaluate the trained models on f30k or coco datasets, and you need to specify the model paths.

```
python eval.py --dataset f30k --data_path data/ --gpu-id 0
python eval.py --dataset coco --data_path data/ --gpu-id 1
```


## Performances
The following tables show the reproducing results of cross-modal retrieval on **MSCOCO** and **Flickr30K** datasets. 
We provide the training logs, checkpoints, performances, and hyper-parameters.

|Datasets| Visual encoders |I2T R@1|I2T R@5|T2I R@1|T2I R@5|Model checkpoint|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Flickr30K |ViT | 75.8 | 93.8 | 62.5  |87.5 |[Link](https://drive.google.com/drive/folders/1m3Y9TMkas2efSbeDV_ESci6uwMGd3MUY?usp=sharing)|
|Flickr30K |Swin | 84.5 | 97.7 | 72.3 | 92.7 |[Link](https://drive.google.com/drive/folders/1jd4i2sGEtbYjkNwjas51NTHmiOmq7Zaz?usp=sharing)|
|MSCOCO-1K |ViT | 78.6 | 96.0 | 65.5 | 91.4 |[Link](https://drive.google.com/drive/folders/1C0XZ7FAoq47huy6gItcIV4XqudRVfe3n?usp=sharing)|
|MSCOCO-1K |Swin | 83.9 | 97.9 | 51.2 | 79.3 |[Link](https://drive.google.com/drive/folders/1JvSVOjIQhofGveR2m8xPGiwhuFutyi15?usp=sharing)|
|MSCOCO-5K |ViT | 56.1 | 83.9 | 71.9 | 93.7 |[Link](https://drive.google.com/drive/folders/1C0XZ7FAoq47huy6gItcIV4XqudRVfe3n?usp=sharing)|
|MSCOCO-5K |Swin | 65.1 | 90.2 | 51.2 | 79.3 |[Link](https://drive.google.com/drive/folders/1JvSVOjIQhofGveR2m8xPGiwhuFutyi15?usp=sharing)|


## Reference

```
@InProceedings{fu2024linguistic,
    author    = {Fu, Zheren and Zhang, Lei and Xia, Hou and Mao, Zhendong},
    title     = {Linguistic-Aware Patch Slimming Framework for Fine-grained Cross-Modal Alignment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {26307-26316}
}
```

