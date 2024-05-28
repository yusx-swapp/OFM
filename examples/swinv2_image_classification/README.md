# Optimize Swin Transformer via OSF 

## Experiment Goal
In this experiment, we will:

- [x] 1. Train Swin Transformer models (Swin and Swinv2) using OSF

- [x] 2. Extract subnets from Swin models (> 30\% model size reduction), and zero-shot evaluation on image classification task


## Hands-on Tutorial
We provide Jupyter Notebook Tutorial for you to validate our results step-by-step: **[Swin Example](swin_img_classification.ipynb)**

## Available Supernet checkpoints

We pushed our trained supernets to the Huggingface model hub, you can find the checkpoints in the following links:

- [ ] [Super-Swinv2-base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-swin-base-patch4-window7-cifar10)
- [ ] [Super-Swinv2-base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-swinv2-base-patch4-window7-cifar100/tree/main)
- [ ] [Super-CLIP-base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-clip-base-patch32-cifar10)
- [ ] [Super-CLIP-base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-clip-base-patch32-cifar100)
- [ ] [Super-Mamba-1.4B](https://huggingface.co/yusx-swapp/ofm-mamba-1.4b-lambda-hf)
- [ ] [Super-ViT-Base for ImageNet](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-imagenet)
- [ ] [Super-ViT-Base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar100)
- [ ] [Super-ViT-Base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar10)


## Run the Experiments

### Installation
Refer to the detailed [installation](../../README.md) guide.

```bash
conda create -n ofm python=3.10
conda activate ofm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


### Start Training


```bash
cd OFM/
```
Training on single node:
```bash
python scripts/train_img_classification.py \
--model swinv2 \
--save_dir ckpts/swinv2-cifar10-single \
--dataset cifar10 \ # you can change to cifar100 and ImageNet
--num_shards 16 \
--lr 2e-5 \
--batch_size 64 \
--log_interval 100 \
--huggingface_token "Your_token_here (optional)"  \ #Mandatory for ImageNet and push your ckpt to Huggingface model hub
--elastic_config scripts/swin_elastic_space.json 
```

Training on multiple GPUs and wish to launch distributed training, you can use the following command:
```bash
torchrun --nproc_per_node='your numer of gpus' --nnodes=1 scripts/train_img_classification.py 
--model swinv2 \
--save_dir ckpts/swinv2-cifar10-multi \
--dataset cifar10 \ # you can change to cifar100 and ImageNet
--num_shards 16 \
--lr 2e-5 \
--batch_size 64 \
--log_interval 100 \
--huggingface_token "Your_token_here (optional)"  \ #Mandatory for ImageNet and push your ckpt to Huggingface model hub
--elastic_config scripts/swin_elastic_space.json \
```

<!--
## Results

We have some simple meta results shown on the tutorial: **[post_training_deployment.ipynb](./post_training_deployment.ipynb)**

| ![Performance vs Params](./figures/RoBERTa_performance_vs_params.png) | ![ViT Performance vs Params](./figures/vit_performance_vs_params.png) |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: |
|                   Fig.1 - Scalable RoBERTa on SST-2                   |                    Fig.2 - Scalable ViT on CIFAR10                    |

Figure 1 shows the trained RoBERTa on SST-2 dataset, we sample resource-aware scaled submodel in different size, and evaluate without further training, all submodels get the same level of accuracy.

Similarlly, in Figure 2, we show the trained scalable ViT's performance on CIFAR-10, notebally, with half of the parameter scaled out, submodels with 45M parameters (75% FLOPs reduction) achieves 94.5% accuracy without further training.

In summry, Foundation Models trained by RaFFM are scalable, which can enables heterogeneous model deployment post-federated learning without further training. -->

## **Reference**
:raised_hands: Thanks for the great work from the authors of Swin Transformer and Swin Transformer v2, we appreciate your efforts and contributions to the community.:raised_hands:
```

@inproceedings{liu2022video,
  title={Video swin transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={3202--3211},
  year={2022}
}

@inproceedings{liu2022swin,
  title={Swin transformer v2: Scaling up capacity and resolution},
  author={Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and Ning, Jia and Cao, Yue and Zhang, Zheng and Dong, Li and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12009--12019},
  year={2022}
}

```