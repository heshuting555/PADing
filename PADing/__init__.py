# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .PADing_model import PADing
# evaluation
from .evaluation.panoptic_evaluation_gzero import PanopticzeroEvaluator
from .evaluation.sem_seg_evaluation_gzero import SemSegGzeroEvaluator
from .evaluation.ins_seg_evaluation_gzero import InsSegzeroEvaluator
from .evaluation.ins_seg_evaluation_zsi import InsSegzsiEvaluator

