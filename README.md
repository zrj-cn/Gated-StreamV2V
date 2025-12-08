# Gated-StreamV2V

## Quick Start

### Preparation
```bash
# get the code 
git clone https://github.com/zrj-cn/Gated-StreamV2V.git
cd Gated-StreamV2V

# environment
conda env create -f environment.yml
pip install opencv-python==4.9.0.80
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu121
```

⚠️ For required LoRA and other components, please refer to [StreamV2V's README.md](./README-old.md).
### Modify paths in the code
Search for strings containing "zrj" in the files and replace them with your own paths. 

### Run the code
⭐️ Single-video testing

```bash
cd vid2vid
# similarity gate streamv2v
python main.py --input ./demo_selfie/tennis.mp4 --prompt "Ukiyo-e Art - a man holding a tennis racket on a tennis court" --cache_interval 1 --cached_attn_style "similarity"

# confidence gate streamv2v
python main.py --input ./demo_selfie/tennis.mp4 --prompt "Ukiyo-e Art - a man holding a tennis racket on a tennis court" --cache_interval 1 --cached_attn_style "confidence"

# original streamv2v
python main.py --input ./demo_selfie/tennis.mp4 --prompt "Ukiyo-e Art - a man holding a tennis racket on a tennis court" --cache_interval 1 --cached_attn_style "origin"
```

🌟 Batch-video evaluate
```bash
# Make sure you are in the project root directory
cd ..
bash run.sh
```

## Code Description
Gating implementation based on similarity, see [similarity_gate_attention.py](src/streamv2v/models/similarity_gate_attention.py)  
Gating implementation based on alignment confidence, see [confidence_gate_attention.py](src/streamv2v/models/confidence_gate_attention.py)


## Acknowledgements
Our project is built upon [StreamV2V](https://github.com/Jeff-LiangF/streamv2v). We thank the contributors of StreamV2V.
