# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco_panoptic import load_coco_panoptic_json
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager
import json

unseen_name = [
        "cow",
        "giraffe",
        "suitcase",
        "frisbee",
        "skateboard",
        "carrot",
        "scissors",
        "cardboard",
        "playingfield",
        "river",
        "road",
        'tree-merged',
        'sky-other-merged',
        'grass-merged',
]


def get_coco_panoptic_seen_all_metadata():
    meta = {}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]
    assert len(stuff_classes) == 133, len(stuff_classes)
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_coco_panoptic_seen(
    name, metadata, image_root, panoptic_root, panoptic_json, instances_json=None
):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, metadata),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        evaluation_set=unseen_name,
        unseen_index=[19, 23,  28, 29,  36,  51,  76, 83, 97, 99, 100, 116, 119, 125],
        **metadata,
    )


def register_all_coco_panoptic_seen(root):
    prefix = "coco_2017_train_panoptic_zsi"
    panoptic_root = "coco/panoptic_train2017"
    panoptic_json = "coco/annotations/ZSP/panoptic_train2017_seen.json"

    prefix_instances = prefix[: -len("_panoptic_zsi")]
    instances_meta = MetadataCatalog.get(prefix_instances)
    image_root, instances_json = instances_meta.image_root, instances_meta.json_file

    register_coco_panoptic_seen(
            prefix,
            get_coco_panoptic_seen_all_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


def get_metadata_val_all():
    meta = {}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors


    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_coco_panoptic_json_semantic(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_coco_panoptic_annos_sem_seg(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    panoptic_name = name
    delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
    delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
        # thing_dataset_id_to_contiguous_id=metadata["thing_dataset_id_to_contiguous_id"],
    )

    # the name is "coco_2017_train_panoptic_with_sem_seg" and "coco_2017_val_panoptic_with_sem_seg"
    semantic_name = name + "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_coco_panoptic_json_semantic(panoptic_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        val_extra_classes=unseen_name,
        label_divisor=1000,
        evaluation_set=unseen_name,
        unseen_index=[19, 23,  28, 29,  36,  51,  76, 83, 97, 99, 100, 116, 119, 125],
        **metadata,
    )


def register_all_coco_panoptic_annos_sem_seg(root):
    prefix = "coco_2017_val_panoptic"
    panoptic_root = "coco/panoptic_val2017"
    panoptic_json ="coco/annotations/ZSP/panoptic_val2017_all.json"
    semantic_root = "coco/panoptic_semseg_val2017"

    prefix_instances = prefix[: -len("_panoptic")]
    instances_meta = MetadataCatalog.get(prefix_instances)
    image_root, instances_json = instances_meta.image_root, instances_meta.json_file

    register_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata_val_all(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_panoptic_seen(_root)
register_all_coco_panoptic_annos_sem_seg(_root)

