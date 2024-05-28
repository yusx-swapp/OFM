# Optimize CLIP via OSF 

## Experiment Goal
In this experiment, we will:

- [x] 1. Concert pre-trained CLIP model to supernet use OSM with contrastive loss

- [x] 2. Extract CLIP subnets (> 30\% model size reduction), and evaluate on image classification task


## Hands-on Tutorial
We provide Jupyter Notebook Tutorial for you to validate our results step-by-step, in our tutorial we will show you how to extract subnets and evaluate on image classification task: **[CLIP Tutorial](CLIP_img_classification.ipynb)**

## Aviailable Supernet checkpoints

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
python scripts/train_clip_img_classification.py \
--model clip \
--save_dir ckpts/clip-cifar10 \
--dataset cifar10 \
--num_shards 16 \
--lr 2e-5 \
--batch_size 64 \
--log_interval 100 \
--huggingface_token "Your_token_here (optional)"  \ #Mandatory for ImageNet and push your ckpt to Huggingface model hub
--elastic_config scripts/clip_elastic_space.json

```

Training on multiple GPUs and wish to lunch distributed training, you can use the following command:
```bash
torchrun --nproc_per_node='your numer of gpus' --nnodes=1 scripts/train_clip_img_classification.py \
--model clip \
--save_dir ckpts/clip-cifar10 \
--dataset cifar10 \
--num_shards 16 \
--lr 2e-5 \
--batch_size 64 \
--log_interval 100 \
--huggingface_token "Your_token_here (optional)"  \ #Mandatory for ImageNet and push your ckpt to Huggingface model hub
--elastic_config scripts/clip_elastic_space.json
```


## **Reference**
:raised_hands: Thanks for the great work from the authors of CLIP, we appreciate your efforts and contributions to the community.:raised_hands:
```
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}

```