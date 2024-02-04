# Zero-shot foundation model specialization with OFM

## Experiment Goal

In this experiment we show:

**OFM Super-FMs trained is highly scalable, and can be used to a large number of generate high-performance zero-shot downsized models (> $10^{12}$) without further training**

Most importantly, such downsized FMs are **high-performance**, which can be directly deployed to edge devices without further training.

## Reproduce the Experiments

### Installation

Refer the detailed [installation guide](../../README.md).

```bash
conda create -n ofm python=3.10
conda activate ofm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### The provided Super-FMs checkpoints

We pushed our trained super-FMs to the Huggingface model hub, you can find the checkpoints in the following links:

- [Super-ViT-Base for ImageNet](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-imagenet)
- [Super-ViT-Base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar100)
- [Super-ViT-Base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar10)

### Run the Experiments

We provide a Jupyter Notebook Tutorial **[vit_zero_shot_specialization_turorial.ipynb](./vit_zero_shot_specialization_turorial.ipynb)** with detailed instruction and high level APIs to reproduce our experiments.

<!--
## Results

We have some simple meta results shown on the tutorial: **[post_training_deployment.ipynb](./post_training_deployment.ipynb)**

| ![Performance vs Params](./figures/RoBERTa_performance_vs_params.png) | ![ViT Performance vs Params](./figures/vit_performance_vs_params.png) |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: |
|                   Fig.1 - Scalable RoBERTa on SST-2                   |                    Fig.2 - Scalable ViT on CIFAR10                    |

Figure 1 shows the trained RoBERTa on SST-2 dataset, we sample resource-aware scaled submodel in different size, and evaluate without further training, all submodels get the same level of accuracy.

Similarlly, in Figure 2, we show the trained scalable ViT's performance on CIFAR-10, notebally, with half of the parameter scaled out, submodels with 45M parameters (75% FLOPs reduction) achieves 94.5% accuracy without further training.

In summry, Foundation Models trained by RaFFM are scalable, which can enables heterogeneous model deployment post-federated learning without further training. -->
