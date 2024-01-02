#!/bin/bash
#SBATCH --time=3-1:00:00  # max job runtime
#SBATCH --cpus-per-task=8  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=gpu  # partition(s)
#SBATCH --gres=gpu:4
#SBATCH --exclude=crysis,gpu03,frost-2,frost-3,frost-4,frost-5,frost-6  # there's no real way to specify multiple gpus types, so exclude the nodes that contain unacceptable gpus
#SBATCH --mem=128G  # max memory
#SBATCH -J "train_vit_cifar100"  # job name
#SBATCH --mail-user=yusx@iastate.edu  # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=job.%J.out 
#SBATCH --error=job.%J.err 

cd /work/LAS/jannesar-lab/yusx/RAFM

module load ml-gpu

ml-gpu python3 scripts/train_vit.py --model vit --save_dir ckpts/vit-base  --dataset cifar100 --num_shards 20 --batch_size 128 --elastic_config scripts/elastic_space.json