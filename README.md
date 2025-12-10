# vlmAttack
Here we perform evaluation of adversarial robustness of Vision Language Models.


Clone the repository with: 


`git clone https://github.com/ChethanKodase/vlmAttack.git`


Create a conda environment :

`conda create -n vlmAttack python=3.10 -y`

Activate :

`conda activate vlmAttack`

Run :

`export PYTHONNOUSERSITE=1`

Install torch and torchvision :

`python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`


Install other packages :

```
python -m pip install \
  "transformers>=4.45.0" \
  accelerate \
  huggingface_hub \
  pillow \
  sentencepiece \
  tiktoken \
  einops \
  "protobuf<5"

```

Install hugging face hub:

`pip install huggingface_hub`


Login with HF token: 

`hf auth login` 


Make a directory inside the repo:

`mkdir Qwen2.5-VL-7B-Instruct`

Paste the address of the above directory in the local_dir in the below command

```

python - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="/home/luser/vlmAttack/Qwen2.5-VL-7B-Instruct",
    local_dir_use_symlinks=False,
)
print("Download complete.")
EOF


```


Make directories: 

`mkdir outputsStorage`


`mkdir outputsStorage/convergence`


Run : 

`python vlm_attack.py`