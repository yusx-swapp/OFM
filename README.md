# Optimized Supernet Formation: Transforming Pretrained Models for Efficient On-device Inference

This is the official implementation for the paper:

_Optimized Supernet Formation: Transforming Pretrained Models for Efficient On-device Inference_

## Updates

- [x] [11/07/2023] High-level API for edge
- [x] [12/04/2023] APIs for Segment Anything (SAM)
- [x] [02/01/2024] ViT-base supernet checkpoints pushed to huggingface model hub
- [x] [02/01/2024] Hands-on tutorial for quickly converting a given pre-trained model to supernet
- [x] [03/28/2024] Update examples for Mamba, SAM, Swin, and CLIP. Released checkpoints.
## Installation

First, create a conda environment, then install PyTorch.

```bash
conda create -n ofm python=3.10
conda activate ofm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Next, install the OFM package

```bash
cd OFM/
pip install .
```

## Supernets checkpoints

OSF modeling a given pre-trained model as a supernet, with an efficient parallel finetuning process, OSM can transform the target pre-trained model to a supernet, and can be quickly specialized to a wide range of resource constraints (> $10^{12}$) with zero-shot.

To validate the results we reported in our paper, we provide several trained supernet checkpoints. These checkpoints are been pushed to the anonymous Huggingface model hub Repos, which you can find in the following links.

### Checkpoints Links

We pushed our trained supernet to the Huggingface model hub, you can find the checkpoints in the following links:

- [ ] [Super-Swinv2-base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-swin-base-patch4-window7-cifar10)
- [ ] [Super-Swinv2-base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-swinv2-base-patch4-window7-cifar100/tree/main)
- [ ] [Super-CLIP-base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-clip-base-patch32-cifar10)
- [ ] [Super-CLIP-base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-clip-base-patch32-cifar100)
- [ ] [Super-Mamba-1.4B](https://huggingface.co/yusx-swapp/ofm-mamba-1.4b-lambda-hf)
- [ ] [Super-ViT-Base for ImageNet](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-imagenet)
- [ ] [Super-ViT-Base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar100)
- [ ] [Super-ViT-Base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar10)

**_You don't need to download the ckpt files, you can use Huggingface Model Card to load the ckpts files directly.
We will show you how to do that in the following section._**

### Checkpoints Usage

**We provide detailed instructions and hands-on tutorial for you to validate our zero-shot downsized models:**

- [Example on quickly evaluate ViT supernet with high-performance subnets](./examples/post_training_deployment/vit_zero_shot_specialization_turorial.ipynb)

Besides, we also provide a high-level API for you to quickly generate sunets for your supernet with **2 lines of codes**, as shown in the following example:

```python
from transformers import AutoModelForImageClassification
from ofm import OFM

# Generate downsized models
ckpt_path = "ckpts_repo_name" # Copy the huggingface model hub repo name from above link

model = AutoModelForImageClassification.from_pretrained(
    ckpt_path,
    num_labels=10,
    ignore_mismatched_sizes=True,
)

supernet = OFM(model.to("cpu"))
print("Original FM number of parameters:",supernet.total_params)

#Randomly sample a downsized FM
ds_model, params, config = supernet.random_resource_aware_model()
print("subnetwork params",params)
```

## Train your own supernet (Single Node)

### Scripts for converting ViT to a supernet

OFM with its mini-shard training strategy can convert a pre-trained model to a supernet in a fast and efficient way. For instance, you can train a super-ViT on CIFAR-100 with the following command:

```bash
python3 scripts/train_img_classification.py --model vit \
--save_dir ckpts/cifar100  \
--dataset cifar100 \
--num_shards 30 \
--lr 1e-5 \
--batch_size 224 \
--elastic_config scripts/elastic_space.json \
--spp \
--log_interval 100
```

To check the results, you can:

- Check the output information from the terminal console
- Use tensorboard: `tensorboard --logdir log/vit`

### Training on ImageNet

Before you start, you have to be granted access to the ImageNet dataset. You can request and download the dataset from [here](https://huggingface.co/datasets/imagenet-1k).

Set the arguments ` --huggingface_token` to your huggingface token, which should have been granted access to the ImageNet dataset.

```bash
python3 scripts/train_img_classification.py --model vit \
--save_dir 'your_dir'  \
--dataset imagenet-1k \
--num_shards 500 \
--lr 2e-5 \
--batch_size 152 \
--log_interval 500 \
--huggingface_token "your-token-here" \
--elastic_config scripts/elastic_space.json
```

### Distributed Training (Multiple Nodes)

If you have multiple GPUs, you can use the following command to train the super-FM with distributed training:

```bash
torchrun --nproc_per_node='your numer of gpus' --nnodes=1 scripts/dist_train_img_classification.py --model vit \
--save_dir 'your_dir'  \
--dataset imagenet-1k \
--num_shards 500 \
--lr 2e-5 \
--batch_size 152 \
--log_interval 500 \
--huggingface_token "your-token-here" \
--elastic_config scripts/elastic_space.json
```

**[Note]**: More APIs and scripts will be posted, please check the [**Updates**](#updates).

## Supported Foundation Models (02/01/2024)

- [x] ViT
- [x] BERT
- [x] RoBERTa
- [x] DistilBERT
- [x] Flan-T5
- [x] SAM
- [x] Mamba SSM
- [x] LLaMA-7B (deprecated after commit ea6815b7162494667edb9dcd32f554346f07401b)

## Contact

anonymous

<!-- ## TODO

- [x] ViT pre-trained ckpts
- [x] ViT FL simulation scripts
- [x] Tensorboard logger
- [x] Elastic space APIs for system-heteo
- [x] Load ckpt high-level APIs
- [x] Simulation scripts on GLUE
- [x] ViT CIFAR-100 ckpts
- [x] High level API for real edge-FL
- [x] API for segment anything (SAM)
- [x] Evaluate Scripts for resource-aware models
- [ ] BERT-large, FLAN-T5 ckpts
- [ ] Simulation scripts on SQUAD
- [ ] ONNX and TensorRT APIs for edge
- [ ] Tiny fedlib -->

## Citation

If you find our work is helpful, please kindly support our efforts by citing our paper:

```

under review

```

## Acknowledgement

The experiments of this work is sponsored by **[anonymous institution]** and **[anonymous institution]**.

```

```
