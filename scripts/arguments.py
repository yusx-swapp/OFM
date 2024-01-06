import argparse


def arguments():
    parser = argparse.ArgumentParser(
        description="Resource Adaptive Foundation Model Fine-tuning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["resnet", "vit", "vit-large", "distilbert", "bert-base", "bert-large"],
        help="Model architecture to use (resnet or vit)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ckpts/",
        help="dir save the model",
    )

    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="dir save the model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=[
            "cifar100",
            "imagenet-1k",
            "flowers102",
            "Caltech101",
            "cifar10",
            "Food101",
            "sst2",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=None,
        help="split k-shot local data",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=100,
        help="Number of mini shards data",
    )

    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=5,
        help="Step size for the learning rate scheduler",
    )
    parser.add_argument(
        "--num_local_epochs",
        type=int,
        default=1,
        help="Number of local epochs for each client in a federated learning setting",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of communication rounds for federated learning",
    )
    parser.add_argument(
        "--spp", action="store_true", help="salient parameter prioritization"
    )
    parser.add_argument(
        "--batch_size", type=int, help="per device batch size", default=64
    )

    # PEFT arguments
    parser.add_argument(
        "--peft", action="store_true", help="parameter efficient finetuning"
    )

    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default=None,
        help="pre-trained adapter ckpt dir",
    )

    parser.add_argument(
        "--elastic_config",
        type=str,
        default=None,
        help="Elastic space file path",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Convergence patience for early stopping",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval",
    )

    parser.add_argument(
        "--grad_beta",
        type=float,
        default=0.5,
        help="Gradient accumulation beta",
    )

    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="Huggingface token for private model",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        # default="~/.cache/huggingface/datasets",
        default="/work/LAS/jannesar-lab/sixing/.cache",
        help="Cache directory for datasets",
    )

    args = parser.parse_args()
    return args
