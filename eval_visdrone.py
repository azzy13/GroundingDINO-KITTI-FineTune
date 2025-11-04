#!/usr/bin/env python3
"""Minimal VisDrone detection test - uses your existing evaluation code"""

import argparse
import json
import torch
from pathlib import Path

from util.slconfig import SLConfig
from datasets import build_dataset, get_coco_api_from_dataset
from torch.utils.data import DataLoader
import util.misc as utils
from engine import evaluate
from groundingdino.util.utils import clean_state_dict


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--coco_anno', required=True)
parser.add_argument('--image_root', required=True)
parser.add_argument('--output_dir', default='./output')
args = parser.parse_args()

# Load config
cfg = SLConfig.fromfile(args.config_file)
for k, v in cfg._cfg_dict.to_dict().items():
    if not hasattr(args, k):
        setattr(args, k, v)

# Set device
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Build model
from models.registry import MODULE_BUILD_FUNCS
build_func = MODULE_BUILD_FUNCS.get(args.modelname)
model, criterion, postprocessors = build_func(args)

# Load checkpoint
checkpoint = torch.load(args.checkpoint, map_location='cpu')
model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
model.to(args.device)
model.eval()
args.fix_size = False 

# Build dataset
dataset = build_dataset('val', args, {'dataset_mode': 'coco', 'anno': args.coco_anno, 'root': args.image_root})
dataloader = DataLoader(dataset, 1, sampler=torch.utils.data.SequentialSampler(dataset),
                       collate_fn=utils.collate_fn, num_workers=4)
base_ds = get_coco_api_from_dataset(dataset)

# Set flags
args.use_coco_eval = False
args.amp = False
args.debug = False
args.save_results = False
args.label_list = ['car','pedestrian']

Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Run evaluation (your existing function)
stats, coco_eval = evaluate(model, criterion, postprocessors, dataloader, 
                           base_ds, args.device, args.output_dir, False, args, None)

# Save results
with open(Path(args.output_dir) / 'results.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nResults saved to {args.output_dir}/results.json")