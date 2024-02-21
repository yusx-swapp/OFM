# One Foundation Model Fits All (OFM): Single-stage Foundation Model Training with Zero-shot Deployment

This is the official implementation for the paper:

_One Foundation Model Fits All: Single-stage Foundation Model Training with Zero-shot Deployment_

## Updates

- [x] [11/07/2023] High-level API for edge
- [x] [12/04/2023] APIs for Segment Anything (SAM)
- [x] [02/01/2024] ViT-base supernet checkpoints pushed to huggingface model hub
- [x] [02/01/2024] Hands on tutorial for quickly specialize FM with zero-shot

## Installation

First, create a conda environment, then install pytorch.

```bash
conda create -n ofm python=3.10
conda activate ofm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Next, install OFM package

```bash
cd OFM/
pip install .
```

## Super-FMs checkpoints

OFM modeling a given foundation model (FM) as a supernet, with single-stage FM training, the FM can be quickly specialized to a wide range of resource constaints (> $10^{12}$) with zero-shot.

We validate the results we reported in our paper, we provide several trained supernet checkpoints. These checkpoints are been pushed to the anonymous Huggingface model hub Repos, you can find in following links.

### Checkpoints Links

We pushed our trained super-FMs to the Huggingface model hub, you can find the checkpoints in the following links:

- [Super-ViT-Base for ImageNet](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-imagenet)
- [Super-ViT-Base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar100)
- [Super-ViT-Base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar10)

**_You dont need download the ckpt files, you can use Huggingface Model Card to load the ckpt files directly.
We will show you how to do that in the following section._**

### Checkpoints Usage

**We provide detailed instructions and hands on tutorial for you to validate our zero-shot downsized models:**

- [Example on quickly specialize ViT with zero-shot downsized models](./examples/post_training_deployment/vit_zero_shot_specialization_turorial.ipynb)

Besides, we also provide a high-level API for you to quickly generate zero-shot models for your own supernet with **2 lines of codes**, as shown in the following example:

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

## Train your own super-FMs (Single Node)

### Scripts for train Super-ViT

OFM with its mini-shard training strategy can train a super-FM in a fast speed. You can train a super-ViT on CIFAR-100 with the following command:

```bash
python3 scripts/train_vit.py --model vit \
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

Set the arguments ` --huggingface_token` to your huggingface token, which should have be granted access to the ImageNet dataset.

```bash
python3 scripts/train_vit.py --model vit \
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
torchrun --nproc_per_node='your numer of gpus' --nnodes=1 scripts/dist_train_vit.py --model vit \
--save_dir 'your_dir'  \
--dataset imagenet-1k \
--num_shards 500 \
--lr 2e-5 \
--batch_size 152 \
--log_interval 500 \
--huggingface_token "your-token-here" \
--elastic_config scripts/elastic_space.json
```

**[Note]**: More APIs and scripts will post, please check the [**Updates**](#updates).

## Supported Foundation Models (02/01/2024)

- [x] ViT
- [x] BERT
- [x] RoBERTa
- [x] DistilBERT
- [x] Flan-T5
- [x] SAM
- [x] LLaMA-7B

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
