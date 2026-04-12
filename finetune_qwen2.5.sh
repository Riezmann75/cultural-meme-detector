export HF_HOME="/home/giabao/cultural-meme-detector/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch train_qwen_meme_classifier.py
