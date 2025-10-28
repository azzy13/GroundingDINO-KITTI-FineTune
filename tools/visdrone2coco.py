import argparse
import os
from tqdm import tqdm
import json
from PIL import Image

'''
{
    "info": {
        "description": "COCO 2017 Dataset","url": "http://cocodataset.org","version": "1.0","year": 2017,"contributor": "COCO Consortium","date_created": "2017/09/01"
    },
    "licenses": [
        {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"}
    ],
    "images": [
        {"id": 242287, "license": 4, "coco_url": "http://images.cocodataset.org/val2017/xxxxxxxxxxxx.jpg", "flickr_url": "http://farm3.staticflickr.com/2626/xxxxxxxxxxxx.jpg", "width": 426, "height": 640, "file_name": "xxxxxxxxx.jpg", "date_captured": "2013-11-15 02:41:42"},
        {"id": 245915, "license": 4, "coco_url": "http://images.cocodataset.org/val2017/nnnnnnnnnnnn.jpg", "flickr_url": "http://farm1.staticflickr.com/88/xxxxxxxxxxxx.jpg", "width": 640, "height": 480, "file_name": "nnnnnnnnnn.jpg", "date_captured": "2013-11-18 02:53:27"}
    ],
    "annotations": [
        {"id": 125686, "category_id": 0, "iscrowd": 0, "segmentation": [[164.81, 417.51,......167.55, 410.64]], "image_id": 242287, "area": 42061.80340000001, "bbox": [19.23, 383.18, 314.5, 244.46]},
        {"id": 1409619, "category_id": 0, "iscrowd": 0, "segmentation": [[376.81, 238.8,........382.74, 241.17]], "image_id": 245915, "area": 3556.2197000000015, "bbox": [399, 251, 155, 101]},
        {"id": 1410165, "category_id": 1, "iscrowd": 0, "segmentation": [[486.34, 239.01,..........495.95, 244.39]], "image_id": 245915, "area": 1775.8932499999994, "bbox": [86, 65, 220, 334]}
    ],
    "categories": [
        {"supercategory": "speaker","id": 0,"name": "echo"},
        {"supercategory": "speaker","id": 1,"name": "echo dot"}
    ]
    }
'''
# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}


def visdrone2coco(args):
    result = {
        "info": {
            "description": "Visdrone Dataset","url": "","version": "1.0","year": 2017,"contributor": "","date_created": ""
        },
        "licenses": [
            {"url": "","id": 0,"name": "Visdrone License"}
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "car","id": 0,"name": "car"},
            {"supercategory": "pedestrian","id": 1,"name": "pedestrian"}
        ]
    }
    category_visdrone_to_coco = {
        1: 1, # Pedestrian->pedestrian
        2: 1, # People->pedestrian
        4: 0, # Car->car
        5: 0, # Van->car
        6: 0, # Truck->car
        9: 0, # Bus->car
        10: 0  # Motor->car
    }
    sequence_file_list = os.listdir(os.path.join(args.inputfolder, "annotations"))
    new_image_id = 0
    for sequence in tqdm(sequence_file_list):
        annotation_path = os.path.join(args.inputfolder, "annotations", sequence)
        image_relative_folder = os.path.join("sequences", sequence.split(".txt")[0])
        image_folder = os.path.join(args.inputfolder, image_relative_folder)
        with open(annotation_path, "r") as f:
            lines = f.readlines()
        first_img_path = os.path.join(image_folder, "0000001.jpg")
        first_img = Image.open(first_img_path)
        image_width, image_height = first_img.size
        frame_id_to_image_id = {}
        for idx, line in enumerate(lines):
            parts = line.strip().split(",")
            frame_id = int(parts[0])
            category_id = int(parts[7])
            if category_id not in category_visdrone_to_coco:  # only keep car and pedestrian
                continue
            bbox_left = float(parts[2])
            bbox_top = float(parts[3])
            bbox_width = float(parts[4])
            bbox_height = float(parts[5])
            file_name = f"{frame_id:07d}.jpg"
            image_relative_path = os.path.join(image_relative_folder, file_name)
            image_path = os.path.join(args.inputfolder, image_relative_path)
            if not os.path.exists(image_path):
                continue
            # Add image info
            if not (frame_id in frame_id_to_image_id):
                frame_id_to_image_id[frame_id] = new_image_id
                new_image_id += 1
                image_id = frame_id_to_image_id[frame_id]
                result['images'].append({
                    "id": image_id,
                    "license": 0, 
                    "file_name": image_relative_path,
                    "width": image_width,
                    "height": image_height,
                    "date_captured": ""
                })
            # Add annotation info
            annotation_id = len(result['annotations']) + 1
            image_id = frame_id_to_image_id[frame_id]
            result['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_visdrone_to_coco[category_id],
                "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
                "segmentation": []
            })
    destination_output_path = os.path.join( args.inputfolder, args.output)
    with open(destination_output_path, "w+") as f:
        json.dump(result, f, indent=4)
    print(f"COCO format json saved to {destination_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("visdrone to coco format.", add_help=True)
    parser.add_argument("--inputfolder", '-f', required=True, type=str, help="input folder path of annotations and images")
    parser.add_argument("--output", '-o', required=True, type=str, help="output json name relative to input folder")
    args = parser.parse_args()

    visdrone2coco(args)
