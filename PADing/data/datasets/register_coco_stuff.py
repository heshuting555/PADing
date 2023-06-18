# Copyright (c) Facebook, Inc. and its affiliates.
import os
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy

COCO_CATEGORIES_Seen_old = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 0, 'name': 'person', 'trainId': 0},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 1, 'name': 'bicycle', 'trainId': 1},
    {'color': [0, 0, 142], 'isthing': 1, 'id': 2, 'name': 'car', 'trainId': 2},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 3, 'name': 'motorcycle', 'trainId': 3},
    {'color': [106, 0, 228], 'isthing': 1, 'id': 4, 'name': 'airplane', 'trainId': 4},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 5, 'name': 'bus', 'trainId': 5},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 6, 'name': 'train', 'trainId': 6},
    {'color': [0, 0, 70], 'isthing': 1, 'id': 7, 'name': 'truck', 'trainId': 7},
    {'color': [0, 0, 192], 'isthing': 1, 'id': 8, 'name': 'boat', 'trainId': 8},
    {'color': [250, 170, 30], 'isthing': 1, 'id': 9, 'name': 'traffic light', 'trainId': 9},
    {'color': [100, 170, 30], 'isthing': 1, 'id': 10, 'name': 'fire hydrant', 'trainId': 10},
    {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'stop sign', 'trainId': 11},
    {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'parking meter', 'trainId': 12},
    {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'bench', 'trainId': 13},
    {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'bird', 'trainId': 14},
    {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'cat', 'trainId': 15},
    {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'dog', 'trainId': 16},
    {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'horse', 'trainId': 17},
    {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'sheep', 'trainId': 18},
    {'color': [110, 76, 0], 'isthing': 1, 'id': 21, 'name': 'elephant', 'trainId': 19},
    {'color': [174, 57, 255], 'isthing': 1, 'id': 22, 'name': 'bear', 'trainId': 20},
    {'color': [199, 100, 0], 'isthing': 1, 'id': 23, 'name': 'zebra', 'trainId': 21},
    {'color': [255, 179, 240], 'isthing': 1, 'id': 26, 'name': 'backpack', 'trainId': 22},
    {'color': [0, 125, 92], 'isthing': 1, 'id': 27, 'name': 'umbrella', 'trainId': 23},
    {'color': [209, 0, 151], 'isthing': 1, 'id': 30, 'name': 'handbag', 'trainId': 24},
    {'color': [188, 208, 182], 'isthing': 1, 'id': 31, 'name': 'tie', 'trainId': 25},
    {'color': [92, 0, 73], 'isthing': 1, 'id': 34, 'name': 'skis', 'trainId': 26},
    {'color': [133, 129, 255], 'isthing': 1, 'id': 35, 'name': 'snowboard', 'trainId': 27},
    {'color': [78, 180, 255], 'isthing': 1, 'id': 36, 'name': 'sports ball', 'trainId': 28},
    {'color': [0, 228, 0], 'isthing': 1, 'id': 37, 'name': 'kite', 'trainId': 29},
    {'color': [174, 255, 243], 'isthing': 1, 'id': 38, 'name': 'baseball bat', 'trainId': 30},
    {'color': [45, 89, 255], 'isthing': 1, 'id': 39, 'name': 'baseball glove', 'trainId': 31},
    {'color': [145, 148, 174], 'isthing': 1, 'id': 41, 'name': 'surfboard', 'trainId': 32},
    {'color': [255, 208, 186], 'isthing': 1, 'id': 42, 'name': 'tennis racket', 'trainId': 33},
    {'color': [197, 226, 255], 'isthing': 1, 'id': 43, 'name': 'bottle', 'trainId': 34},
    {'color': [171, 134, 1], 'isthing': 1, 'id': 45, 'name': 'wine glass', 'trainId': 35},
    {'color': [109, 63, 54], 'isthing': 1, 'id': 46, 'name': 'cup', 'trainId': 36},
    {'color': [207, 138, 255], 'isthing': 1, 'id': 47, 'name': 'fork', 'trainId': 37},
    {'color': [151, 0, 95], 'isthing': 1, 'id': 48, 'name': 'knife', 'trainId': 38},
    {'color': [9, 80, 61], 'isthing': 1, 'id': 49, 'name': 'spoon', 'trainId': 39},
    {'color': [84, 105, 51], 'isthing': 1, 'id': 50, 'name': 'bowl', 'trainId': 40},
    {'color': [74, 65, 105], 'isthing': 1, 'id': 51, 'name': 'banana', 'trainId': 41},
    {'color': [166, 196, 102], 'isthing': 1, 'id': 52, 'name': 'apple', 'trainId': 42},
    {'color': [208, 195, 210], 'isthing': 1, 'id': 53, 'name': 'sandwich', 'trainId': 43},
    {'color': [255, 109, 65], 'isthing': 1, 'id': 54, 'name': 'orange', 'trainId': 44},
    {'color': [0, 143, 149], 'isthing': 1, 'id': 55, 'name': 'broccoli', 'trainId': 45},
    {'color': [209, 99, 106], 'isthing': 1, 'id': 57, 'name': 'hot dog', 'trainId': 46},
    {'color': [5, 121, 0], 'isthing': 1, 'id': 58, 'name': 'pizza', 'trainId': 47},
    {'color': [227, 255, 205], 'isthing': 1, 'id': 59, 'name': 'donut', 'trainId': 48},
    {'color': [147, 186, 208], 'isthing': 1, 'id': 60, 'name': 'cake', 'trainId': 49},
    {'color': [153, 69, 1], 'isthing': 1, 'id': 61, 'name': 'chair', 'trainId': 50},
    {'color': [3, 95, 161], 'isthing': 1, 'id': 62, 'name': 'couch', 'trainId': 51},
    {'color': [163, 255, 0], 'isthing': 1, 'id': 63, 'name': 'potted plant', 'trainId': 52},
    {'color': [119, 0, 170], 'isthing': 1, 'id': 64, 'name': 'bed', 'trainId': 53},
    {'color': [0, 182, 199], 'isthing': 1, 'id': 66, 'name': 'dining table', 'trainId': 54},
    {'color': [0, 165, 120], 'isthing': 1, 'id': 69, 'name': 'toilet', 'trainId': 55},
    {'color': [183, 130, 88], 'isthing': 1, 'id': 71, 'name': 'tv', 'trainId': 56},
    {'color': [95, 32, 0], 'isthing': 1, 'id': 72, 'name': 'laptop', 'trainId': 57},
    {'color': [130, 114, 135], 'isthing': 1, 'id': 73, 'name': 'mouse', 'trainId': 58},
    {'color': [110, 129, 133], 'isthing': 1, 'id': 74, 'name': 'remote', 'trainId': 59},
    {'color': [166, 74, 118], 'isthing': 1, 'id': 75, 'name': 'keyboard', 'trainId': 60},
    {'color': [219, 142, 185], 'isthing': 1, 'id': 76, 'name': 'cell phone', 'trainId': 61},
    {'color': [79, 210, 114], 'isthing': 1, 'id': 77, 'name': 'microwave', 'trainId': 62},
    {'color': [178, 90, 62], 'isthing': 1, 'id': 78, 'name': 'oven', 'trainId': 63},
    {'color': [65, 70, 15], 'isthing': 1, 'id': 79, 'name': 'toaster', 'trainId': 64},
    {'color': [127, 167, 115], 'isthing': 1, 'id': 80, 'name': 'sink', 'trainId': 65},
    {'color': [59, 105, 106], 'isthing': 1, 'id': 81, 'name': 'refrigerator', 'trainId': 66},
    {'color': [142, 108, 45], 'isthing': 1, 'id': 83, 'name': 'book', 'trainId': 67},
    {'color': [196, 172, 0], 'isthing': 1, 'id': 84, 'name': 'clock', 'trainId': 68},
    {'color': [95, 54, 80], 'isthing': 1, 'id': 85, 'name': 'vase', 'trainId': 69},
    {'color': [201, 57, 1], 'isthing': 1, 'id': 87, 'name': 'teddy bear', 'trainId': 70},
    {'color': [246, 0, 122], 'isthing': 1, 'id': 88, 'name': 'hair drier', 'trainId': 71},
    {'color': [191, 162, 208], 'isthing': 1, 'id': 89, 'name': 'toothbrush', 'trainId': 72},
    {'id': 91, 'name': 'banner', 'supercategory': 'textile', 'trainId': 73},
    {'id': 92, 'name': 'blanket', 'supercategory': 'textile', 'trainId': 74},
    {'id': 93, 'name': 'branch', 'supercategory': 'plant', 'trainId': 75},
    {'id': 94, 'name': 'bridge', 'supercategory': 'building', 'trainId': 76},
    {'id': 95, 'name': 'building-other', 'supercategory': 'building', 'trainId': 77},
    {'id': 96, 'name': 'bush', 'supercategory': 'plant', 'trainId': 78},
    {'id': 97, 'name': 'cabinet', 'supercategory': 'furniture-stuff', 'trainId': 79},
    {'id': 98, 'name': 'cage', 'supercategory': 'structural', 'trainId': 80},
    {'id': 100, 'name': 'carpet', 'supercategory': 'floor', 'trainId': 81},
    {'id': 101, 'name': 'ceiling-other', 'supercategory': 'ceiling', 'trainId': 82},
    {'id': 102, 'name': 'ceiling-tile', 'supercategory': 'ceiling', 'trainId': 83},
    {'id': 103, 'name': 'cloth', 'supercategory': 'textile', 'trainId': 84},
    {'id': 104, 'name': 'clothes', 'supercategory': 'textile', 'trainId': 85},
    {'id': 106, 'name': 'counter', 'supercategory': 'furniture-stuff', 'trainId': 86},
    {'id': 107, 'name': 'cupboard', 'supercategory': 'furniture-stuff', 'trainId': 87},
    {'id': 108, 'name': 'curtain', 'supercategory': 'textile', 'trainId': 88},
    {'id': 109, 'name': 'desk-stuff', 'supercategory': 'furniture-stuff', 'trainId': 89},
    {'id': 110, 'name': 'dirt', 'supercategory': 'ground', 'trainId': 90},
    {'id': 111, 'name': 'door-stuff', 'supercategory': 'furniture-stuff', 'trainId': 91},
    {'id': 112, 'name': 'fence', 'supercategory': 'structural', 'trainId': 92},
    {'id': 113, 'name': 'floor-marble', 'supercategory': 'floor', 'trainId': 93},
    {'id': 114, 'name': 'floor-other', 'supercategory': 'floor', 'trainId': 94},
    {'id': 115, 'name': 'floor-stone', 'supercategory': 'floor', 'trainId': 95},
    {'id': 116, 'name': 'floor-tile', 'supercategory': 'floor', 'trainId': 96},
    {'id': 117, 'name': 'floor-wood', 'supercategory': 'floor', 'trainId': 97},
    {'id': 118, 'name': 'flower', 'supercategory': 'plant', 'trainId': 98},
    {'id': 119, 'name': 'fog', 'supercategory': 'water', 'trainId': 99},
    {'id': 120, 'name': 'food-other', 'supercategory': 'food-stuff', 'trainId': 100},
    {'id': 121, 'name': 'fruit', 'supercategory': 'food-stuff', 'trainId': 101},
    {'id': 122, 'name': 'furniture-other', 'supercategory': 'furniture-stuff', 'trainId': 102},
    {'id': 124, 'name': 'gravel', 'supercategory': 'ground', 'trainId': 103},
    {'id': 125, 'name': 'ground-other', 'supercategory': 'ground', 'trainId': 104},
    {'id': 126, 'name': 'hill', 'supercategory': 'solid', 'trainId': 105},
    {'id': 127, 'name': 'house', 'supercategory': 'building', 'trainId': 106},
    {'id': 128, 'name': 'leaves', 'supercategory': 'plant', 'trainId': 107},
    {'id': 129, 'name': 'light', 'supercategory': 'furniture-stuff', 'trainId': 108},
    {'id': 130, 'name': 'mat', 'supercategory': 'textile', 'trainId': 109},
    {'id': 131, 'name': 'metal', 'supercategory': 'raw-material', 'trainId': 110},
    {'id': 132, 'name': 'mirror-stuff', 'supercategory': 'furniture-stuff', 'trainId': 111},
    {'id': 133, 'name': 'moss', 'supercategory': 'plant', 'trainId': 112},
    {'id': 134, 'name': 'mountain', 'supercategory': 'solid', 'trainId': 113},
    {'id': 135, 'name': 'mud', 'supercategory': 'ground', 'trainId': 114},
    {'id': 136, 'name': 'napkin', 'supercategory': 'textile', 'trainId': 115},
    {'id': 137, 'name': 'net', 'supercategory': 'structural', 'trainId': 116},
    {'id': 138, 'name': 'paper', 'supercategory': 'raw-material', 'trainId': 117},
    {'id': 139, 'name': 'pavement', 'supercategory': 'ground', 'trainId': 118},
    {'id': 140, 'name': 'pillow', 'supercategory': 'textile', 'trainId': 119},
    {'id': 141, 'name': 'plant-other', 'supercategory': 'plant', 'trainId': 120},
    {'id': 142, 'name': 'plastic', 'supercategory': 'raw-material', 'trainId': 121},
    {'id': 143, 'name': 'platform', 'supercategory': 'ground', 'trainId': 122},
    {'id': 145, 'name': 'railing', 'supercategory': 'structural', 'trainId': 123},
    {'id': 146, 'name': 'railroad', 'supercategory': 'ground', 'trainId': 124},
    {'id': 149, 'name': 'rock', 'supercategory': 'solid', 'trainId': 125},
    {'id': 150, 'name': 'roof', 'supercategory': 'building', 'trainId': 126},
    {'id': 151, 'name': 'rug', 'supercategory': 'textile', 'trainId': 127},
    {'id': 152, 'name': 'salad', 'supercategory': 'food-stuff', 'trainId': 128},
    {'id': 153, 'name': 'sand', 'supercategory': 'ground', 'trainId': 129},
    {'id': 154, 'name': 'sea', 'supercategory': 'water', 'trainId': 130},
    {'id': 155, 'name': 'shelf', 'supercategory': 'furniture-stuff', 'trainId': 131},
    {'id': 156, 'name': 'sky-other', 'supercategory': 'sky', 'trainId': 132},
    {'id': 157, 'name': 'skyscraper', 'supercategory': 'building', 'trainId': 133},
    {'id': 158, 'name': 'snow', 'supercategory': 'ground', 'trainId': 134},
    {'id': 159, 'name': 'solid-other', 'supercategory': 'solid', 'trainId': 135},
    {'id': 160, 'name': 'stairs', 'supercategory': 'furniture-stuff', 'trainId': 136},
    {'id': 161, 'name': 'stone', 'supercategory': 'solid', 'trainId': 137},
    {'id': 162, 'name': 'straw', 'supercategory': 'plant', 'trainId': 138},
    {'id': 163, 'name': 'structural-other', 'supercategory': 'structural', 'trainId': 139},
    {'id': 164, 'name': 'table', 'supercategory': 'furniture-stuff', 'trainId': 140},
    {'id': 165, 'name': 'tent', 'supercategory': 'building', 'trainId': 141},
    {'id': 166, 'name': 'textile-other', 'supercategory': 'textile', 'trainId': 142},
    {'id': 167, 'name': 'towel', 'supercategory': 'textile', 'trainId': 143},
    {'id': 169, 'name': 'vegetable', 'supercategory': 'food-stuff', 'trainId': 144},
    {'id': 170, 'name': 'wall-brick', 'supercategory': 'wall', 'trainId': 145},
    {'id': 172, 'name': 'wall-other', 'supercategory': 'wall', 'trainId': 146},
    {'id': 173, 'name': 'wall-panel', 'supercategory': 'wall', 'trainId': 147},
    {'id': 174, 'name': 'wall-stone', 'supercategory': 'wall', 'trainId': 148},
    {'id': 175, 'name': 'wall-tile', 'supercategory': 'wall', 'trainId': 149},
    {'id': 176, 'name': 'wall-wood', 'supercategory': 'wall', 'trainId': 150},
    {'id': 177, 'name': 'water-other', 'supercategory': 'water', 'trainId': 151},
    {'id': 178, 'name': 'waterdrops', 'supercategory': 'water', 'trainId': 152},
    {'id': 179, 'name': 'window-blind', 'supercategory': 'window', 'trainId': 153},
    {'id': 180, 'name': 'window-other', 'supercategory': 'window', 'trainId': 154},
    {'id': 181, 'name': 'wood', 'supercategory': 'solid', 'trainId': 155}]

# new version, with color added
COCO_CATEGORIES_Seen = [{'color': [220, 20, 60], 'isthing': 1, 'id': 0, 'name': 'person', 'trainId': 0},
                        {'color': [119, 11, 32], 'isthing': 1, 'id': 1, 'name': 'bicycle', 'trainId': 1},
                        {'color': [0, 0, 142], 'isthing': 1, 'id': 2, 'name': 'car', 'trainId': 2},
                        {'color': [0, 0, 230], 'isthing': 1, 'id': 3, 'name': 'motorcycle', 'trainId': 3},
                        {'color': [106, 0, 228], 'isthing': 1, 'id': 4, 'name': 'airplane', 'trainId': 4},
                        {'color': [0, 60, 100], 'isthing': 1, 'id': 5, 'name': 'bus', 'trainId': 5},
                        {'color': [0, 80, 100], 'isthing': 1, 'id': 6, 'name': 'train', 'trainId': 6},
                        {'color': [0, 0, 70], 'isthing': 1, 'id': 7, 'name': 'truck', 'trainId': 7},
                        {'color': [0, 0, 192], 'isthing': 1, 'id': 8, 'name': 'boat', 'trainId': 8},
                        {'color': [250, 170, 30], 'isthing': 1, 'id': 9, 'name': 'traffic light', 'trainId': 9},
                        {'color': [100, 170, 30], 'isthing': 1, 'id': 10, 'name': 'fire hydrant', 'trainId': 10},
                        {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'stop sign', 'trainId': 11},
                        {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'parking meter', 'trainId': 12},
                        {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'bench', 'trainId': 13},
                        {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'bird', 'trainId': 14},
                        {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'cat', 'trainId': 15},
                        {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'dog', 'trainId': 16},
                        {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'horse', 'trainId': 17},
                        {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'sheep', 'trainId': 18},
                        {'color': [110, 76, 0], 'isthing': 1, 'id': 21, 'name': 'elephant', 'trainId': 19},
                        {'color': [174, 57, 255], 'isthing': 1, 'id': 22, 'name': 'bear', 'trainId': 20},
                        {'color': [199, 100, 0], 'isthing': 1, 'id': 23, 'name': 'zebra', 'trainId': 21},
                        {'color': [255, 179, 240], 'isthing': 1, 'id': 26, 'name': 'backpack', 'trainId': 22},
                        {'color': [0, 125, 92], 'isthing': 1, 'id': 27, 'name': 'umbrella', 'trainId': 23},
                        {'color': [209, 0, 151], 'isthing': 1, 'id': 30, 'name': 'handbag', 'trainId': 24},
                        {'color': [188, 208, 182], 'isthing': 1, 'id': 31, 'name': 'tie', 'trainId': 25},
                        {'color': [92, 0, 73], 'isthing': 1, 'id': 34, 'name': 'skis', 'trainId': 26},
                        {'color': [133, 129, 255], 'isthing': 1, 'id': 35, 'name': 'snowboard', 'trainId': 27},
                        {'color': [78, 180, 255], 'isthing': 1, 'id': 36, 'name': 'sports ball', 'trainId': 28},
                        {'color': [0, 228, 0], 'isthing': 1, 'id': 37, 'name': 'kite', 'trainId': 29},
                        {'color': [174, 255, 243], 'isthing': 1, 'id': 38, 'name': 'baseball bat', 'trainId': 30},
                        {'color': [45, 89, 255], 'isthing': 1, 'id': 39, 'name': 'baseball glove', 'trainId': 31},
                        {'color': [145, 148, 174], 'isthing': 1, 'id': 41, 'name': 'surfboard', 'trainId': 32},
                        {'color': [255, 208, 186], 'isthing': 1, 'id': 42, 'name': 'tennis racket', 'trainId': 33},
                        {'color': [197, 226, 255], 'isthing': 1, 'id': 43, 'name': 'bottle', 'trainId': 34},
                        {'color': [171, 134, 1], 'isthing': 1, 'id': 45, 'name': 'wine glass', 'trainId': 35},
                        {'color': [109, 63, 54], 'isthing': 1, 'id': 46, 'name': 'cup', 'trainId': 36},
                        {'color': [207, 138, 255], 'isthing': 1, 'id': 47, 'name': 'fork', 'trainId': 37},
                        {'color': [151, 0, 95], 'isthing': 1, 'id': 48, 'name': 'knife', 'trainId': 38},
                        {'color': [9, 80, 61], 'isthing': 1, 'id': 49, 'name': 'spoon', 'trainId': 39},
                        {'color': [84, 105, 51], 'isthing': 1, 'id': 50, 'name': 'bowl', 'trainId': 40},
                        {'color': [74, 65, 105], 'isthing': 1, 'id': 51, 'name': 'banana', 'trainId': 41},
                        {'color': [166, 196, 102], 'isthing': 1, 'id': 52, 'name': 'apple', 'trainId': 42},
                        {'color': [208, 195, 210], 'isthing': 1, 'id': 53, 'name': 'sandwich', 'trainId': 43},
                        {'color': [255, 109, 65], 'isthing': 1, 'id': 54, 'name': 'orange', 'trainId': 44},
                        {'color': [0, 143, 149], 'isthing': 1, 'id': 55, 'name': 'broccoli', 'trainId': 45},
                        {'color': [209, 99, 106], 'isthing': 1, 'id': 57, 'name': 'hot dog', 'trainId': 46},
                        {'color': [5, 121, 0], 'isthing': 1, 'id': 58, 'name': 'pizza', 'trainId': 47},
                        {'color': [227, 255, 205], 'isthing': 1, 'id': 59, 'name': 'donut', 'trainId': 48},
                        {'color': [147, 186, 208], 'isthing': 1, 'id': 60, 'name': 'cake', 'trainId': 49},
                        {'color': [153, 69, 1], 'isthing': 1, 'id': 61, 'name': 'chair', 'trainId': 50},
                        {'color': [3, 95, 161], 'isthing': 1, 'id': 62, 'name': 'couch', 'trainId': 51},
                        {'color': [163, 255, 0], 'isthing': 1, 'id': 63, 'name': 'potted plant', 'trainId': 52},
                        {'color': [119, 0, 170], 'isthing': 1, 'id': 64, 'name': 'bed', 'trainId': 53},
                        {'color': [0, 182, 199], 'isthing': 1, 'id': 66, 'name': 'dining table', 'trainId': 54},
                        {'color': [0, 165, 120], 'isthing': 1, 'id': 69, 'name': 'toilet', 'trainId': 55},
                        {'color': [183, 130, 88], 'isthing': 1, 'id': 71, 'name': 'tv', 'trainId': 56},
                        {'color': [95, 32, 0], 'isthing': 1, 'id': 72, 'name': 'laptop', 'trainId': 57},
                        {'color': [130, 114, 135], 'isthing': 1, 'id': 73, 'name': 'mouse', 'trainId': 58},
                        {'color': [110, 129, 133], 'isthing': 1, 'id': 74, 'name': 'remote', 'trainId': 59},
                        {'color': [166, 74, 118], 'isthing': 1, 'id': 75, 'name': 'keyboard', 'trainId': 60},
                        {'color': [219, 142, 185], 'isthing': 1, 'id': 76, 'name': 'cell phone', 'trainId': 61},
                        {'color': [79, 210, 114], 'isthing': 1, 'id': 77, 'name': 'microwave', 'trainId': 62},
                        {'color': [178, 90, 62], 'isthing': 1, 'id': 78, 'name': 'oven', 'trainId': 63},
                        {'color': [65, 70, 15], 'isthing': 1, 'id': 79, 'name': 'toaster', 'trainId': 64},
                        {'color': [127, 167, 115], 'isthing': 1, 'id': 80, 'name': 'sink', 'trainId': 65},
                        {'color': [59, 105, 106], 'isthing': 1, 'id': 81, 'name': 'refrigerator', 'trainId': 66},
                        {'color': [142, 108, 45], 'isthing': 1, 'id': 83, 'name': 'book', 'trainId': 67},
                        {'color': [196, 172, 0], 'isthing': 1, 'id': 84, 'name': 'clock', 'trainId': 68},
                        {'color': [95, 54, 80], 'isthing': 1, 'id': 85, 'name': 'vase', 'trainId': 69},
                        {'color': [201, 57, 1], 'isthing': 1, 'id': 87, 'name': 'teddy bear', 'trainId': 70},
                        {'color': [246, 0, 122], 'isthing': 1, 'id': 88, 'name': 'hair drier', 'trainId': 71},
                        {'color': [191, 162, 208], 'isthing': 1, 'id': 89, 'name': 'toothbrush', 'trainId': 72},
                        {'id': 91, 'name': 'banner', 'supercategory': 'textile', 'trainId': 73, 'color': [145, 7, 119]},
                        {'id': 92, 'name': 'blanket', 'supercategory': 'textile', 'trainId': 74, 'color': [126, 162, 219]},
                        {'id': 93, 'name': 'branch', 'supercategory': 'plant', 'trainId': 75, 'color': [32, 127, 181]},
                        {'id': 94, 'name': 'bridge', 'supercategory': 'building', 'trainId': 76, 'color': [10, 63, 228]},
                        {'id': 95, 'name': 'building-other', 'supercategory': 'building', 'trainId': 77, 'color': [247, 81, 199]},
                        {'id': 96, 'name': 'bush', 'supercategory': 'plant', 'trainId': 78, 'color': [231, 46, 102]},
                        {'id': 97, 'name': 'cabinet', 'supercategory': 'furniture-stuff', 'trainId': 79, 'color': [191, 84, 87]},
                        {'id': 98, 'name': 'cage', 'supercategory': 'structural', 'trainId': 80, 'color': [251, 107, 196]},
                        {'id': 100, 'name': 'carpet', 'supercategory': 'floor', 'trainId': 81, 'color': [6, 14, 127]},
                        {'id': 101, 'name': 'ceiling-other', 'supercategory': 'ceiling', 'trainId': 82, 'color': [219, 72, 57]},
                        {'id': 102, 'name': 'ceiling-tile', 'supercategory': 'ceiling', 'trainId': 83, 'color': [87, 129, 198]},
                        {'id': 103, 'name': 'cloth', 'supercategory': 'textile', 'trainId': 84, 'color': [131, 171, 89]},
                        {'id': 104, 'name': 'clothes', 'supercategory': 'textile', 'trainId': 85, 'color': [249, 131, 179]},
                        {'id': 106, 'name': 'counter', 'supercategory': 'furniture-stuff', 'trainId': 86, 'color': [105, 185, 157]},
                        {'id': 107, 'name': 'cupboard', 'supercategory': 'furniture-stuff', 'trainId': 87, 'color': [180, 167, 229]},
                        {'id': 108, 'name': 'curtain', 'supercategory': 'textile', 'trainId': 88, 'color': [144, 104, 123]},
                        {'id': 109, 'name': 'desk-stuff', 'supercategory': 'furniture-stuff', 'trainId': 89, 'color': [99, 248, 65]},
                        {'id': 110, 'name': 'dirt', 'supercategory': 'ground', 'trainId': 90, 'color': [212, 57, 231]},
                        {'id': 111, 'name': 'door-stuff', 'supercategory': 'furniture-stuff', 'trainId': 91, 'color': [69, 46, 54]},
                        {'id': 112, 'name': 'fence', 'supercategory': 'structural', 'trainId': 92, 'color': [50, 31, 194]},
                        {'id': 113, 'name': 'floor-marble', 'supercategory': 'floor', 'trainId': 93, 'color': [32, 180, 46]},
                        {'id': 114, 'name': 'floor-other', 'supercategory': 'floor', 'trainId': 94, 'color': [139, 204, 224]},
                        {'id': 115, 'name': 'floor-stone', 'supercategory': 'floor', 'trainId': 95, 'color': [218, 120, 60]},
                        {'id': 116, 'name': 'floor-tile', 'supercategory': 'floor', 'trainId': 96, 'color': [189, 120, 186]},
                        {'id': 117, 'name': 'floor-wood', 'supercategory': 'floor', 'trainId': 97, 'color': [136, 227, 37]},
                        {'id': 118, 'name': 'flower', 'supercategory': 'plant', 'trainId': 98, 'color': [200, 185, 122]},
                        {'id': 119, 'name': 'fog', 'supercategory': 'water', 'trainId': 99, 'color': [171, 36, 68]},
                        {'id': 120, 'name': 'food-other', 'supercategory': 'food-stuff', 'trainId': 100, 'color': [250, 100, 186]},
                        {'id': 121, 'name': 'fruit', 'supercategory': 'food-stuff', 'trainId': 101, 'color': [213, 59, 199]},
                        {'id': 122, 'name': 'furniture-other', 'supercategory': 'furniture-stuff', 'trainId': 102, 'color': [65, 89, 248]},
                        {'id': 124, 'name': 'gravel', 'supercategory': 'ground', 'trainId': 103, 'color': [97, 26, 20]},
                        {'id': 125, 'name': 'ground-other', 'supercategory': 'ground', 'trainId': 104, 'color': [117, 2, 191]},
                        {'id': 126, 'name': 'hill', 'supercategory': 'solid', 'trainId': 105, 'color': [21, 208, 53]},
                        {'id': 127, 'name': 'house', 'supercategory': 'building', 'trainId': 106, 'color': [16, 128, 59]},
                        {'id': 128, 'name': 'leaves', 'supercategory': 'plant', 'trainId': 107, 'color': [55, 65, 93]},
                        {'id': 129, 'name': 'light', 'supercategory': 'furniture-stuff', 'trainId': 108, 'color': [43, 145, 180]},
                        {'id': 130, 'name': 'mat', 'supercategory': 'textile', 'trainId': 109, 'color': [243, 83, 95]},
                        {'id': 131, 'name': 'metal', 'supercategory': 'raw-material', 'trainId': 110, 'color': [24, 162, 73]},
                        {'id': 132, 'name': 'mirror-stuff', 'supercategory': 'furniture-stuff', 'trainId': 111, 'color': [188, 104, 145]},
                        {'id': 133, 'name': 'moss', 'supercategory': 'plant', 'trainId': 112, 'color': [3, 204, 201]},
                        {'id': 134, 'name': 'mountain', 'supercategory': 'solid', 'trainId': 113, 'color': [67, 124, 1]},
                        {'id': 135, 'name': 'mud', 'supercategory': 'ground', 'trainId': 114, 'color': [207, 84, 178]},
                        {'id': 136, 'name': 'napkin', 'supercategory': 'textile', 'trainId': 115, 'color': [167, 173, 172]},
                        {'id': 137, 'name': 'net', 'supercategory': 'structural', 'trainId': 116, 'color': [162, 101, 21]},
                        {'id': 138, 'name': 'paper', 'supercategory': 'raw-material', 'trainId': 117, 'color': [55, 50, 201]},
                        {'id': 139, 'name': 'pavement', 'supercategory': 'ground', 'trainId': 118, 'color': [131, 75, 158]},
                        {'id': 140, 'name': 'pillow', 'supercategory': 'textile', 'trainId': 119, 'color': [179, 250, 156]},
                        {'id': 141, 'name': 'plant-other', 'supercategory': 'plant', 'trainId': 120, 'color': [136, 42, 5]},
                        {'id': 142, 'name': 'plastic', 'supercategory': 'raw-material', 'trainId': 121, 'color': [232, 105, 64]},
                        {'id': 143, 'name': 'platform', 'supercategory': 'ground', 'trainId': 122, 'color': [166, 84, 129]},
                        {'id': 145, 'name': 'railing', 'supercategory': 'structural', 'trainId': 123, 'color': [59, 131, 81]},
                        {'id': 146, 'name': 'railroad', 'supercategory': 'ground', 'trainId': 124, 'color': [120, 172, 183]},
                        {'id': 149, 'name': 'rock', 'supercategory': 'solid', 'trainId': 125, 'color': [100, 243, 12]},
                        {'id': 150, 'name': 'roof', 'supercategory': 'building', 'trainId': 126, 'color': [31, 7, 234]},
                        {'id': 151, 'name': 'rug', 'supercategory': 'textile', 'trainId': 127, 'color': [10, 49, 1]},
                        {'id': 152, 'name': 'salad', 'supercategory': 'food-stuff', 'trainId': 128, 'color': [170, 236, 151]},
                        {'id': 153, 'name': 'sand', 'supercategory': 'ground', 'trainId': 129, 'color': [181, 50, 205]},
                        {'id': 154, 'name': 'sea', 'supercategory': 'water', 'trainId': 130, 'color': [30, 209, 226]},
                        {'id': 155, 'name': 'shelf', 'supercategory': 'furniture-stuff', 'trainId': 131, 'color': [27, 111, 93]},
                        {'id': 156, 'name': 'sky-other', 'supercategory': 'sky', 'trainId': 132, 'color': [72, 124, 141]},
                        {'id': 157, 'name': 'skyscraper', 'supercategory': 'building', 'trainId': 133, 'color': [161, 247, 118]},
                        {'id': 158, 'name': 'snow', 'supercategory': 'ground', 'trainId': 134, 'color': [243, 255, 203]},
                        {'id': 159, 'name': 'solid-other', 'supercategory': 'solid', 'trainId': 135, 'color': [56, 255, 59]},
                        {'id': 160, 'name': 'stairs', 'supercategory': 'furniture-stuff', 'trainId': 136, 'color': [193, 1, 201]},
                        {'id': 161, 'name': 'stone', 'supercategory': 'solid', 'trainId': 137, 'color': [222, 161, 30]},
                        {'id': 162, 'name': 'straw', 'supercategory': 'plant', 'trainId': 138, 'color': [188, 160, 220]},
                        {'id': 163, 'name': 'structural-other', 'supercategory': 'structural', 'trainId': 139, 'color': [162, 112, 49]},
                        {'id': 164, 'name': 'table', 'supercategory': 'furniture-stuff', 'trainId': 140, 'color': [237, 207, 135]},
                        {'id': 165, 'name': 'tent', 'supercategory': 'building', 'trainId': 141, 'color': [78, 193, 82]},
                        {'id': 166, 'name': 'textile-other', 'supercategory': 'textile', 'trainId': 142, 'color': [78, 162, 149]},
                        {'id': 167, 'name': 'towel', 'supercategory': 'textile', 'trainId': 143, 'color': [167, 226, 13]},
                        {'id': 169, 'name': 'vegetable', 'supercategory': 'food-stuff', 'trainId': 144, 'color': [230, 103, 156]},
                        {'id': 170, 'name': 'wall-brick', 'supercategory': 'wall', 'trainId': 145, 'color': [109, 163, 157]},
                        {'id': 172, 'name': 'wall-other', 'supercategory': 'wall', 'trainId': 146, 'color': [233, 54, 243]},
                        {'id': 173, 'name': 'wall-panel', 'supercategory': 'wall', 'trainId': 147, 'color': [254, 234, 8]},
                        {'id': 174, 'name': 'wall-stone', 'supercategory': 'wall', 'trainId': 148, 'color': [133, 47, 185]},
                        {'id': 175, 'name': 'wall-tile', 'supercategory': 'wall', 'trainId': 149, 'color': [21, 194, 101]},
                        {'id': 176, 'name': 'wall-wood', 'supercategory': 'wall', 'trainId': 150, 'color': [80, 14, 190]},
                        {'id': 177, 'name': 'water-other', 'supercategory': 'water', 'trainId': 151, 'color': [154, 90, 250]},
                        {'id': 178, 'name': 'waterdrops', 'supercategory': 'water', 'trainId': 152, 'color': [127, 170, 250]},
                        {'id': 179, 'name': 'window-blind', 'supercategory': 'window', 'trainId': 153, 'color': [106, 128, 43]},
                        {'id': 180, 'name': 'window-other', 'supercategory': 'window', 'trainId': 154, 'color': [164, 189, 146]},
                        {'id': 181, 'name': 'wood', 'supercategory': 'solid', 'trainId': 155, 'color': [189, 181, 199]}]

COCO_CATEGORIES_Unseen = [{'color': [120, 166, 157], 'isthing': 1, 'id': 20, 'name': 'cow', 'trainId': 0},
                        {'color': [72, 0, 118], 'isthing': 1, 'id': 24, 'name': 'giraffe', 'trainId': 1},
                        {'color': [0, 220, 176], 'isthing': 1, 'id': 32, 'name': 'suitcase', 'trainId': 2},
                        {'color': [255, 99, 164], 'isthing': 1, 'id': 33, 'name': 'frisbee', 'trainId': 3},
                        {'color': [134, 134, 103], 'isthing': 1, 'id': 40, 'name': 'skateboard', 'trainId': 4},
                        {'color': [179, 0, 194], 'isthing': 1, 'id': 56, 'name': 'carrot', 'trainId': 5},
                        {'color': [128, 76, 255], 'isthing': 1, 'id': 86, 'name': 'scissors', 'trainId': 6},
                        {'id': 99, 'name': 'cardboard', 'supercategory': 'raw-material', 'trainId': 7, 'color': [82, 168, 193]},
                        {'id': 105, 'name': 'clouds', 'supercategory': 'sky', 'trainId': 8, 'color': [15, 72, 36]},
                        {'id': 123, 'name': 'grass', 'supercategory': 'plant', 'trainId': 9, 'color': [135, 65, 45]},
                        {'id': 144, 'name': 'playingfield', 'supercategory': 'ground', 'trainId': 10, 'color': [170, 3, 219]},
                        {'id': 147, 'name': 'river', 'supercategory': 'water', 'trainId': 11, 'color': [223, 172, 138]},
                        {'id': 148, 'name': 'road', 'supercategory': 'ground', 'trainId': 12, 'color': [217, 84, 156]},
                        {'id': 168, 'name': 'tree', 'supercategory': 'plant', 'trainId': 13, 'color': [92, 69, 129]},
                        {'id': 171, 'name': 'wall-concrete', 'supercategory': 'wall', 'trainId': 14, 'color': [77, 122, 59]}]

COCO_CATEGORIES_Unseen2 = copy.deepcopy(COCO_CATEGORIES_Unseen)

for item in COCO_CATEGORIES_Unseen2:
    item['trainId'] = item['trainId'] + 156

COCO_CATEGORIES_ALL = COCO_CATEGORIES_Seen + COCO_CATEGORIES_Unseen2

def _get_coco_stuff_seen_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES_Seen]
    assert len(stuff_ids) == 156, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES_Seen]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES_Seen]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_coco_stuff_seen(root):
    root = os.path.join(root, "coco", "coco_stuff")
    meta = _get_coco_stuff_seen_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "../train2017", "annotations_detectron2/train2017"),
        ("test", "../val2017", "annotations_detectron2/val2017_seen"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"coco_2017_{name}_stuff_seen_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            unseen_index=torch.arange(156, 171),
            **meta,
        )

def _get_coco_stuff_unseen_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES_Unseen]
    assert len(stuff_ids) == 15, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES_Unseen]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES_Unseen]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_coco_stuff_unseen(root):
    root = os.path.join(root, "coco", "coco_stuff")
    meta = _get_coco_stuff_unseen_meta()

    name = 'val_unseen'
    image_dirname = "../val2017"
    sem_seg_dirname = "annotations_detectron2/val2017_unseen"
    image_dir = os.path.join(root, image_dirname)
    gt_dir = os.path.join(root, sem_seg_dirname)
    name = f"coco_2017_{name}_stuff_seen_sem_seg"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
    )
    val_extra_classes = [k["name"] for k in COCO_CATEGORIES_Unseen]
    MetadataCatalog.get(name).set(
        val_extra_classes=val_extra_classes,
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,
        unseen_index=torch.arange(156, 171),
        **meta,
    )

def _get_coco_stuff_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES_ALL]
    assert len(stuff_ids) == 171, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in COCO_CATEGORIES_ALL]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES_ALL]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_coco_stuff_val_all(root):
    root = os.path.join(root, "coco", "coco_stuff")
    meta = _get_coco_stuff_meta()
    name = 'val_all'
    image_dirname = "../val2017"
    sem_seg_dirname = "annotations_detectron2/val2017_all"
    image_dir = os.path.join(root, image_dirname)
    gt_dir = os.path.join(root, sem_seg_dirname)
    name = f"coco_2017_{name}_stuff_sem_seg"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
    )

    val_extra_classes = [k["name"] for k in COCO_CATEGORIES_Unseen]
    MetadataCatalog.get(name).set(
        val_extra_classes=val_extra_classes,
        image_root=image_dir,
        sem_seg_root=gt_dir,
        # evaluator_type="sem_seg",
        evaluator_type="sem_seg_gzero",
        ignore_label=255,
        unseen_index=torch.arange(156, 171),
        **meta,
    )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_stuff_seen(_root)
register_all_coco_stuff_unseen(_root)
register_all_coco_stuff_val_all(_root)