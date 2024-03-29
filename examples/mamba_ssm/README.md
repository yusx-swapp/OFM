# Optimize Swin Transformer via OFM 

## Experiment Goal

In this experiment we will:

- [x] 1. Train selective state space model Mamba using OFM

- [x] 2. Extract downsized Mamba models (> 800M model parameter reduction), and evaluation on Lambda dataset.

## Dependencies
Before start, you need to install the **[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)**. We appreciate their efforts and contributions to the community.:raised_hands:    
To install the **lm-eval** package from the github repository, run:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
More detailed instructions, please see their repo **[Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)**.
## Hands on Tutorial
We provides Jupyter Notebook Tutorial for you to validate our results step-by-step: **[Mamba Example](mamba_lm_harness.ipynb)**

## Aviailable Super-FMs checkpoints

We pushed our trained super-FMs to the Huggingface model hub, you can find the checkpoints in the following links:

- [ ] [Super-Swin-base for CIFAR-10]()
- [ ] [Super-Swin-base for CIFAR-100]()
- [ ] [Super-CLIP-base for CIFAR-10]()
- [ ] [Super-CLIP-base for CIFAR-100]()
- [ ] [Super-Mamba-370m]()
- [ ] [Super-Mamba-1.4B]()
- [ ] [Super-ViT-Base for ImageNet](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-imagenet)
- [ ] [Super-ViT-Base for CIFAR-100](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar100)
- [ ] [Super-ViT-Base for CIFAR-10](https://huggingface.co/yusx-swapp/ofm-vit-base-patch16-224-cifar10)


## Run the Experiments

### Installation

Refer the detailed [installation guide](../../README.md).

```bash
conda create -n ofm python=3.10
conda activate ofm
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```


### Validate the results

See tutorial: **[Mamba Example](mamba_lm_harness.ipynb)**



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

@misc{gu2023mamba,
      title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces}, 
      author={Albert Gu and Tri Dao},
      year={2023},
      eprint={2312.00752},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}

```