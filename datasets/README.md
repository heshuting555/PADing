# Prepare Datasets for PADing

### Expected dataset structure:

```
coco
├── coco_stuff
    ├── annotations
    ├── annotations_detectron2
    ├── word_vectors
    └── split
├── train2017
    ├── 000000189148.jpg
    └── ...
├── val2017
    ├── 000000213547.jpg
    └── ...
├── train2014
    ├── COCO_train2014_xxxxxxxxxxxx.jpg
    └── ...
├── val2014
    ├── COCO_val2014_xxxxxxxxxxxx.jpg
    └── ...
├── panoptic_{train,val}2017 # png annotations
├── panoptic_semseg_{train,val}2017 # generated from panoptic annotations
└── annotations
    ├── panoptic_train2017.json
    ├── panoptic_val2017.json
    ├── ZSP
        ├── panoptic_train2017_all.json
        ├── panoptic_train2017_seen.json
        ├── panoptic_train2017_unseen.json
        ├── panoptic_val2017_all.json
        ├── panoptic_val2017_seen.json
        └── panoptic_val2017_unseen.json
    └── ZSIS
        ├── instances_train2014_seen_48_17.json
        ├── instances_train2014_seen_65_15.json
        ├── instances_val2014_gzsi_48_17.json
        ├── instances_val2014_gzsi_65_15.json
        ├── instances_val2014_unseen_48_17.json
        └── instances_val2014_unseen_65_15.json

```

### Panoptic Segmentation

Split json into seen and unseen setting using the following command:

```
python datasets/get_zsp_json.py
```

### Instance Segmentation

Please follow the setting of the [ZSI](https://github.com/zhengye1995/Zero-shot-Instance-Segmentation).

### Semantic Segmentation

Please follow the setting of the [ZegFormer](https://github.com/dingjiansw101/ZegFormer).

