"""Download cube checkpoint from HuggingFace and save as _object.ckpt for eval.py."""

import sys
sys.path.insert(0, '/users/md4047/projects/le-wm')

import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from huggingface_hub import hf_hub_download
from pathlib import Path

from jepa import JEPA
from module import ARPredictor, Embedder, MLP

print("Downloading weights.pt from HuggingFace (quentinll/lewm-cube)...")
weights_path = hf_hub_download(repo_id='quentinll/lewm-cube', filename='weights.pt')
print(f"Downloaded to: {weights_path}")

# Build model matching config.json from HuggingFace:
#   encoder: vit-tiny, patch_size=14, image_size=224
#   predictor: ARPredictor, num_frames=3, dim=192, depth=6, heads=16
#   action_encoder: Embedder, input_dim=25, emb_dim=192
#   projector/pred_proj: MLP(192->2048->192) with BatchNorm1d
print("Building JEPA model from config...")

encoder = spt.backbone.utils.vit_hf(
    'tiny',
    patch_size=14,
    image_size=224,
    pretrained=False,
    use_mask_token=False,
)

predictor = ARPredictor(
    num_frames=3,
    input_dim=192,
    hidden_dim=192,
    output_dim=192,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.0,
)

action_encoder = Embedder(input_dim=25, emb_dim=192)

projector = MLP(
    input_dim=192,
    output_dim=192,
    hidden_dim=2048,
    norm_fn=torch.nn.BatchNorm1d,
)

pred_proj = MLP(
    input_dim=192,
    output_dim=192,
    hidden_dim=2048,
    norm_fn=torch.nn.BatchNorm1d,
)

model = JEPA(
    encoder=encoder,
    predictor=predictor,
    action_encoder=action_encoder,
    projector=projector,
    pred_proj=pred_proj,
)

print("Loading weights...")
weights = torch.load(weights_path, map_location='cpu', weights_only=False)
model.load_state_dict(weights)
model.eval()
print("Weights loaded successfully.")

cache_dir = Path(swm.data.utils.get_cache_dir())
cube_dir = cache_dir / 'cube'
cube_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = cube_dir / 'lewm_object.ckpt'

torch.save(model, ckpt_path)
print(f"Checkpoint saved to: {ckpt_path}")
