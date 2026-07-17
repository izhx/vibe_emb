# vibe_emb

## ENV
<!-- 在哈深机器是暂时用 viet 环境. 在带/mnt/的计算集群用 /mnt/share/envs/embt 环境 -->

```shell
conda create -n viet python=3.12
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install "transformers==5.8.1" "accelerate==1.13.0" "datasets==4.8.5" "peft==0.19.1" "mteb==2.14.9" "filelock<3.21.0"
# lower filelock version to fix behavior on other filesystem.
# install flash_attn from wheels
# pip install ~/files/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

## vllm 要装太多东西，单独开环境吧
# pip install vllm==0.19.1
# 0.13 for torch 2.9
```
