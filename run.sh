export HF_HOME="/home/giabao/cultural-meme-detector/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=4 ddp_torch.py
