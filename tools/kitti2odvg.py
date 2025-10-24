#!/usr/bin/env python3
"""
KITTI to ODVG Format Converter - Clean Implementation
Converts KITTI tracking/detection labels to ODVG format for GroundingDINO fine-tuning.

Author: Built from scratch to avoid the mess
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

class KITTIToODVG:
    """Converter for KITTI -> ODVG format"""
    
    # KITTI class name mapping (lowercase for consistency)
    CLASS_MAP = {
        'Car': 'car',
        'Van': 'car',
        'Truck': 'truck',
        'Pedestrian': 'pedestrian',
        'Person_sitting': 'pedestrian',
        'Cyclist': 'cyclist',
        'Tram': 'tram',
        'Misc': 'misc',
        'DontCare': None  # Skip
    }
    
    def __init__(self, image_root, label_root, output_file, dataset_type='tracking'):
        """
        Args:
            image_root: Path to images (e.g., 'training/image_02')
            label_root: Path to labels (e.g., 'training/label_02')
            output_file: Output ODVG jsonl file
            dataset_type: 'tracking' or 'detection'
        """
        self.image_root = Path(image_root)
        self.label_root = Path(label_root)
        self.output_file = output_file
        self.dataset_type = dataset_type
        
        # Stats
        self.stats = {
            'total_images': 0,
            'total_instances': 0,
            'classes': defaultdict(int),
            'skipped_dontcare': 0,
            'empty_images': 0
        }
    
    def parse_tracking_label(self, label_file):
        """
        Parse KITTI tracking label file
        Format: frame trackID type truncated occluded alpha bbox_2d dim location rotation_y score
        """
        frames = defaultdict(list)
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:
                    continue
                
                frame_id = int(parts[0])
                obj_type = parts[2]
                
                # Skip DontCare and unknown classes
                if obj_type not in self.CLASS_MAP or self.CLASS_MAP[obj_type] is None:
                    self.stats['skipped_dontcare'] += 1
                    continue
                
                # Extract bbox: left, top, right, bottom (indices 6-9)
                # KITTI format: x_min, y_min, x_max, y_max in pixels
                bbox = [
                    float(parts[6]),  # left (x_min)
                    float(parts[7]),  # top (y_min)
                    float(parts[8]),  # right (x_max)
                    float(parts[9])   # bottom (y_max)
                ]
                
                # Sanity check: ensure bbox is valid
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue  # Invalid bbox
                
                phrase = self.CLASS_MAP[obj_type]
                self.stats['classes'][phrase] += 1
                
                frames[frame_id].append({
                    'bbox': bbox,
                    'phrase': phrase
                })
        
        return frames
    
    def parse_detection_label(self, label_file):
        """
        Parse KITTI detection label file (single frame per file)
        Format: type truncated occluded alpha bbox_2d dim location rotation_y
        """
        regions = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                
                obj_type = parts[0]
                
                # Skip DontCare
                if obj_type not in self.CLASS_MAP or self.CLASS_MAP[obj_type] is None:
                    self.stats['skipped_dontcare'] += 1
                    continue
                
                # Extract bbox (indices 4-7 for detection)
                bbox = [
                    float(parts[4]),  # left
                    float(parts[5]),  # top
                    float(parts[6]),  # right
                    float(parts[7])   # bottom
                ]
                
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue
                
                phrase = self.CLASS_MAP[obj_type]
                self.stats['classes'][phrase] += 1
                
                regions.append({
                    'bbox': bbox,
                    'phrase': phrase
                })
        
        return regions
    
    def get_image_dimensions(self, image_path):
        """Get image dimensions from actual file"""
        try:
            with Image.open(image_path) as img:
                return img.size[1], img.size[0]  # height, width
        except Exception as e:
            print(f"Warning: Could not read {image_path}: {e}")
            return 375, 1242  # Default KITTI dimensions
    
    def convert_tracking(self):
        """Convert KITTI tracking dataset"""
        print(f"Converting KITTI Tracking dataset...")
        print(f"Image root: {self.image_root}")
        print(f"Label root: {self.label_root}")
        
        # Get all label files
        label_files = sorted(self.label_root.glob('*.txt'))
        print(f"Found {len(label_files)} sequences")
        
        with open(self.output_file, 'w') as out_f:
            for label_file in tqdm(label_files, desc="Processing sequences"):
                seq_id = label_file.stem
                
                # Parse all frames in this sequence
                frames = self.parse_tracking_label(label_file)
                
                # Write each frame as separate ODVG entry
                for frame_id in sorted(frames.keys()):
                    regions = frames[frame_id]
                    
                    if not regions:
                        self.stats['empty_images'] += 1
                        continue  # Skip empty frames
                    
                    # Construct image path
                    image_filename = f"{seq_id}/{frame_id:06d}.png"
                    image_path = self.image_root / seq_id / f"{frame_id:06d}.png"
                    
                    # Get actual image dimensions
                    height, width = self.get_image_dimensions(image_path)
                    
                    # Create ODVG entry
                    entry = {
                        "filename": image_filename,
                        "height": height,
                        "width": width,
                        "grounding": {
                            "regions": regions
                        }
                    }
                    
                    out_f.write(json.dumps(entry) + '\n')
                    self.stats['total_images'] += 1
                    self.stats['total_instances'] += len(regions)
    
    def convert_detection(self):
        """Convert KITTI detection dataset"""
        print(f"Converting KITTI Detection dataset...")
        print(f"Image root: {self.image_root}")
        print(f"Label root: {self.label_root}")
        
        label_files = sorted(self.label_root.glob('*.txt'))
        print(f"Found {len(label_files)} images")
        
        with open(self.output_file, 'w') as out_f:
            for label_file in tqdm(label_files, desc="Processing images"):
                image_id = label_file.stem
                
                # Parse single frame
                regions = self.parse_detection_label(label_file)
                
                if not regions:
                    self.stats['empty_images'] += 1
                    continue
                
                # Image path
                image_filename = f"{image_id}.png"
                image_path = self.image_root / image_filename
                
                # Get dimensions
                height, width = self.get_image_dimensions(image_path)
                
                # Create ODVG entry
                entry = {
                    "filename": image_filename,
                    "height": height,
                    "width": width,
                    "grounding": {
                        "regions": regions
                    }
                }
                
                out_f.write(json.dumps(entry) + '\n')
                self.stats['total_images'] += 1
                self.stats['total_instances'] += len(regions)
    
    def convert(self):
        """Run conversion"""
        if self.dataset_type == 'tracking':
            self.convert_tracking()
        else:
            self.convert_detection()
        
        self.print_stats()
    
    def print_stats(self):
        """Print conversion statistics"""
        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")
        print("="*60)
        print(f"Output file: {self.output_file}")
        print(f"Total images: {self.stats['total_images']}")
        print(f"Total instances: {self.stats['total_instances']}")
        print(f"Empty images skipped: {self.stats['empty_images']}")
        print(f"DontCare instances skipped: {self.stats['skipped_dontcare']}")
        print(f"\nClass distribution:")
        for cls, count in sorted(self.stats['classes'].items()):
            print(f"  {cls}: {count}")
        print("="*60)


def validate_odvg(odvg_file, num_samples=5):
    """Validate ODVG file format"""
    print(f"\n{'='*60}")
    print(f"VALIDATING: {odvg_file}")
    print("="*60)
    
    with open(odvg_file, 'r') as f:
        lines = f.readlines()
        print(f"Total entries: {len(lines)}\n")
        
        for i, line in enumerate(lines[:num_samples]):
            entry = json.loads(line)
            print(f"Sample {i+1}:")
            print(f"  Filename: {entry['filename']}")
            print(f"  Dimensions: {entry['width']}x{entry['height']}")
            print(f"  Regions: {len(entry['grounding']['regions'])}")
            
            for j, region in enumerate(entry['grounding']['regions'][:3]):
                bbox = region['bbox']
                print(f"    Region {j+1}: {region['phrase']}")
                print(f"      BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
                # Validation checks
                if not (0 < bbox[0] < entry['width'] and 0 < bbox[2] < entry['width']):
                    print(f"      ⚠️  WARNING: X coords outside image width!")
                if not (0 < bbox[1] < entry['height'] and 0 < bbox[3] < entry['height']):
                    print(f"      ⚠️  WARNING: Y coords outside image height!")
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    print(f"      ⚠️  ERROR: Invalid bbox (min >= max)!")
            
            print()
    
    print("✓ Validation complete!\n")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert KITTI to ODVG format')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Path to image directory (e.g., data/kitti_tracking/training/image_02)')
    parser.add_argument('--label_root', type=str, required=True,
                        help='Path to label directory (e.g., data/kitti_tracking/training/label_02)')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for ODVG file (default: current directory)')
    parser.add_argument('--output_name', type=str, default='kitti_tracking_train_odvg.jsonl',
                        help='Output filename (default: kitti_tracking_train_odvg.jsonl)')
    parser.add_argument('--dataset_type', type=str, default='tracking', choices=['tracking', 'detection'],
                        help='Dataset type: tracking or detection (default: tracking)')
    parser.add_argument('--validate_samples', type=int, default=5,
                        help='Number of samples to validate (default: 5)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / args.output_name
    
    print(f"\nConfiguration:")
    print(f"  Image root: {args.image_root}")
    print(f"  Label root: {args.label_root}")
    print(f"  Output file: {output_file}")
    print(f"  Dataset type: {args.dataset_type}\n")
    
    # Run conversion
    converter = KITTIToODVG(
        image_root=args.image_root,
        label_root=args.label_root,
        output_file=str(output_file),
        dataset_type=args.dataset_type
    )
    converter.convert()
    
    # Validate output
    validate_odvg(str(output_file), num_samples=args.validate_samples)
    
    print("\n🎯 Ready for training!")
    print(f"Use this file in your config: {output_file}")