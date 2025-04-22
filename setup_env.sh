# environment setup: A6000, cuda=12.2, nv-driver=535.216.03
conda create --name qwen-ocr python=3.12
conda activate qwen-ocr
pip install git+https://github.com/huggingface/transformers.git accelerate
pip install qwen-vl-utils[decord]==0.0.8
pip install datasets pillow matplotlib requests flash-attn torchvision peft
pip install scikit-image evaluate anls RapidFuzz
pip install huggingface_hub
# Download pretrained weights for Qwen2.5-VL-3B-Instruct
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./Qwen2.5-VL-3B-Instruct-Pretrained

# Assume NodeJS, pdflatex, gs are installed

# Training Pipeline
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install wandb # Setup your WandDB account
# Update dataset info to include our datasets
cp ../dataset_info.json data/dataset_info.json