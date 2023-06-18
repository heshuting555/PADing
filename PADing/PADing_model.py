# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion_Generator
from .modeling.matcher import HungarianMatcher
from PADing.utils.network_utils import GMMNnetwork, GMMNLoss, Decoder, RelationNet, Encoder
from .modeling.transformer_decoder.transformer import TransformerDecoderLayer, TransformerDecoder
from PADing.third_party import clip
from PADing.third_party import imagenet_templates
import numpy as np

@META_ARCH_REGISTRY.register()
class PADing(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        num_choose: int,
        class_weight_eco: float,
        structrue_weight: float,
        drop_rate: float,
        weight_1: float,
        weight_2: float,
        weight_3: float,
        vec_path,
        trans_num_layer: int = 1,
        trans_drop_rate: float,
        trans_num_query: int = 400,
        test_metadata,
        noise: bool,
        no_object_weight: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.unseen_classes_idx_metric = metadata.unseen_index
        print(self.unseen_classes_idx_metric)
        self.num_unseen = len(self.unseen_classes_idx_metric)
        self.num_seen = self.sem_seg_head.num_classes - len(self.unseen_classes_idx_metric)
        embedding_size = 256
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.class_weight_eco = class_weight_eco
        self.num_choose = num_choose

        self.new_fc = nn.Linear(256, self.sem_seg_head.num_classes + 1)

        self.criterion_generator = GMMNLoss(
            sigma=[2, 5, 10, 20, 40, 80]).build_loss()
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.weight_3 = weight_3
        self.noise = noise
        print('num_choose is {}'.format(num_choose))
        print('class_weight_eco is {}'.format(class_weight_eco))
        print('trans_drop_rate is {}'.format(trans_drop_rate))
        print('weight_1 is {}'.format(weight_1))
        print('weight_2 is {}'.format(weight_2))
        print('weight_3 is {}'.format(weight_3))
        print('noise is {}'.format(noise))
        print("trans_num_query is {}".format(trans_num_query))
        if vec_path is not None:
            vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')
            vec = vec_load[:, 1:self.num_seen + 1]
            vec_unseen = vec_load[:, self.num_seen + 1:]
            self.vec = torch.tensor(vec, dtype=torch.float32).cuda()
            self.vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32).cuda()
            self.noise_dim = 300
        else:
            self.vec = self.get_embeddings(test_metadata.stuff_classes)
            self.vec_unseen = self.vec[self.unseen_classes_idx_metric].t()
            self.vec = self.vec.t()
            self.noise_dim = 512
        self.fc_projection = nn.Linear(self.noise_dim, embedding_size)
        self.dict_feature = nn.Embedding(trans_num_query, embedding_size)
        decoder_layer = TransformerDecoderLayer(embedding_size, 8, 1024, trans_drop_rate, "relu", False, noise=noise)
        decoder_norm = nn.LayerNorm(embedding_size)
        self.generator = TransformerDecoder(decoder_layer, trans_num_layer, decoder_norm, return_intermediate=False)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.no_object_weight = no_object_weight
        self.projection_related = Encoder(drop_rate, embedding_size)
        self.projection_unrelated = Encoder(drop_rate, embedding_size)
        self.relation = RelationNet(embedding_size, embedding_size)
        self.decoder = Decoder(drop_rate, embedding_size)
        self.structrue_weight = structrue_weight

    def get_embeddings(self, class_names):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_pretrained = "ViT-B/16"
        clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
        print('Loading CLIP from : {}'.format(clip_pretrained))

        class_texts = class_names
        prompt_ensemble_type = "imagenet_select"
        with torch.no_grad():
            assert "A photo of" not in class_texts[0]
            if prompt_ensemble_type == "imagenet_select":
                prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
            elif prompt_ensemble_type == "imagenet":
                prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
            elif prompt_ensemble_type == "single":
                prompt_templates = ['A photo of a {} in the scene', ]
            else:
                raise NotImplementedError

            def zeroshot_classifier(classnames, templates, clip_modelp):
                with torch.no_grad():
                    zeroshot_weights = []
                    for classname in classnames:
                        if ', ' in classname:
                            classname_splits = classname.split(', ')
                            texts = []
                            for template in templates:
                                for cls_split in classname_splits:
                                    texts.append(template.format(cls_split))
                        else:
                            texts = [template.format(classname) for template in templates]  # format with class
                        texts = clip.tokenize(texts).cuda()  # tokenize, shape: [48, 77]
                        class_embeddings = clip_modelp.encode_text(texts)  # embed with text encoder
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        zeroshot_weights.append(class_embedding)
                    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
                return zeroshot_weights

            text_features = zeroshot_classifier(class_texts, prompt_templates, clip_model).permute(1, 0).float()
        return text_features

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion_Generator(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # new add
            "class_weight_eco": cfg.MODEL.MASK_FORMER.CLASS_ECO,
            "num_choose": cfg.MODEL.MASK_FORMER.NUM_CHOOSE,
            "structrue_weight": cfg.MODEL.MASK_FORMER.STRUCTURE_WEIGHT,
            "drop_rate": cfg.MODEL.MASK_FORMER.DROP_OUT_ED,
            "weight_1": cfg.MODEL.MASK_FORMER.WEIGHT_1,
            "weight_2": cfg.MODEL.MASK_FORMER.WEIGHT_2,
            "weight_3": cfg.MODEL.MASK_FORMER.WEIGHT_3,
            "vec_path":  cfg.MODEL.MASK_FORMER.VEC_PATH,
            "trans_num_layer": cfg.MODEL.MASK_FORMER.TRANS_NUM_LAYER,
            "trans_drop_rate": cfg.MODEL.MASK_FORMER.TRANS_DROP_RATE,
            "trans_num_query" : cfg.MODEL.MASK_FORMER.TRANS_MEMORY_NUM_QUERY,
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "noise": cfg.MODEL.MASK_FORMER.NOISE,
            "no_object_weight" : cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def set_grad_True(self):
        for p in self.generator.parameters():
            p.requires_grad = True
        for p in self.projection.parameters():
            p.requires_grad = True

    def set_grad_false(self):
        for p in self.sem_seg_head.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward_train_generator(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        vec = self.vec.detach()

        self.set_grad_false()

        with torch.no_grad():
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images)
            all_classes, target_classes, idx = self.criterion(outputs, targets)
            real_embeddings_all = outputs["pred_embed"] #100
            real_embeddings = outputs["pred_embed"][idx] #4

        vec = self.fc_projection(vec.t())
        vector_choose_seen = vec[target_classes]

        memory_bank = self.dict_feature.weight.unsqueeze(1).repeat(1, vector_choose_seen.shape[0], 1)
        fake_seen_embeddings = self.generator(vector_choose_seen.unsqueeze(0), memory_bank).squeeze(0).squeeze(0)
        generate_loss = self.criterion_generator(fake_seen_embeddings, real_embeddings)

        # cls_loss = self.cls_criterion(self.sem_seg_head.predictor.class_embed(fake_seen_embeddings)[:, :156], target_classes)

        cls_loss = self.cls_criterion(self.new_fc(fake_seen_embeddings)[:, :156], target_classes)
        cls_loss = 0.001 * cls_loss

        fake_seen_related = self.projection_related(fake_seen_embeddings)
        fake_seen_unrelated = self.projection_unrelated(fake_seen_embeddings)

        fake_seen_unrelated_noise = torch.FloatTensor(fake_seen_unrelated.shape[0], fake_seen_unrelated.shape[1]).to(self.device).normal_(0, 1)
        rec_fake_seen = self.decoder(torch.cat([fake_seen_related, fake_seen_unrelated], dim=-1))

        relations_1 = self.relation(fake_seen_related, vec).view(fake_seen_related.shape[0], -1)

        relation_seen_1 = F.cross_entropy(relations_1, target_classes, reduction="mean")
        kl_unrelated_1 = kl_loss(fake_seen_unrelated, fake_seen_unrelated_noise)

        kl_related_3 = kl_loss(cal_similarity(fake_seen_related, fake_seen_related), cal_similarity(vector_choose_seen, vector_choose_seen)) # 92 x 92
        rec_loss_1 = F.l1_loss(rec_fake_seen, fake_seen_embeddings)

        # ========================unseen=============================
        vec_unseen = self.vec_unseen
        vector_choose_unseen = self.fc_projection(vec_unseen.t())

        memory_bank = self.dict_feature.weight.unsqueeze(1).repeat(1, vector_choose_unseen.shape[0], 1)
        fake_unseen_embeddings = self.generator(vector_choose_unseen.unsqueeze(0), memory_bank).squeeze(0).squeeze(0)

        # fake unseen embedding
        fake_unseen_related = self.projection_related(fake_unseen_embeddings)
        fake_unseen_unrelated = self.projection_unrelated(fake_unseen_embeddings)
        fake_unseen_unrelated_noise = torch.FloatTensor(fake_unseen_unrelated.shape[0], fake_unseen_unrelated.shape[1]).to(self.device).normal_(0, 1)
        rec_fake_unseen = self.decoder(torch.cat([fake_unseen_related, fake_unseen_unrelated], dim=-1))

        relations = self.relation(fake_unseen_related, vector_choose_unseen).view(fake_unseen_related.shape[0], -1)
        # Loss
        relation_unseen_1 = F.cross_entropy(relations, torch.arange(self.num_unseen, device=self.device), reduction="mean")
        kl_unseen_1 = kl_loss(cal_similarity(fake_unseen_related, fake_unseen_related), cal_similarity(vector_choose_unseen, vector_choose_unseen))

        kl_unseen_2 = kl_loss(cal_similarity(fake_seen_related, fake_unseen_related), cal_similarity(vector_choose_seen, vector_choose_unseen))
        rec_loss_4 = F.l1_loss(rec_fake_unseen, fake_unseen_embeddings)
        kl_unrelated_4 = kl_loss(fake_unseen_unrelated, fake_unseen_unrelated_noise)

        kl_loss_all = (kl_related_3 + kl_unseen_1 + kl_unseen_2) / 3.0
        relation_loss = 0.2 * (relation_seen_1 + relation_unseen_1) / 2.0
        rec_loss = (rec_loss_1 + rec_loss_4) / 2.0
        unrelated = 0.5 * (kl_unrelated_1 + kl_unrelated_4)/2.0

        loss_structure = self.weight_1 * (kl_loss_all + unrelated) + self.weight_2 * relation_loss + self.weight_3 * rec_loss

        sperate_loss = cls_loss

        losses = {'generate_loss': generate_loss + sperate_loss + self.structrue_weight * loss_structure}

        return losses, real_embeddings_all, all_classes

    def forward_train_classifier(self, real_embeddings, target_classes):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        vec_unseen = self.vec_unseen.detach()
        vec_unseen = self.fc_projection(vec_unseen.t())

        index_choose = torch.randperm(self.num_unseen)[:self.num_choose].to(self.device)

        vector_choose = vec_unseen[:self.num_unseen][index_choose]

        memory_bank = self.dict_feature.weight.unsqueeze(1).repeat(1, vector_choose.shape[0], 1)
        fake_unseen_embeddings = self.generator(vector_choose.unsqueeze(0), memory_bank).squeeze(0).squeeze(0)

        labels_unseen = []
        for idx in index_choose:
            labels_unseen.append(self.unseen_classes_idx_metric[idx])

        target_classes = target_classes.flatten()
        labels_unseen = torch.tensor(labels_unseen).to(target_classes.device)
        labels = torch.cat([target_classes, labels_unseen], dim=0)
        real_embeddings = real_embeddings.reshape(-1, real_embeddings.shape[-1])
        embeddings = torch.cat([real_embeddings, fake_unseen_embeddings.detach()], dim=0)

        output = self.new_fc(embeddings)
        class_weight = torch.ones(self.sem_seg_head.num_classes+1).to(self.device)
        class_weight[self.unseen_classes_idx_metric] = self.class_weight_eco

        class_weight[-1] = self.no_object_weight

        loss_ce = F.cross_entropy(output, labels, class_weight)
        losses = {'dis_loss': loss_ce}

        return losses

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            embedding_results = outputs["pred_embed"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, embedding_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, embedding_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, embedding_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result, embedding_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, embedding_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, embedding_result):
        mask_cls = self.new_fc(embedding_result)
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, embedding_result):

        mask_cls = self.new_fc(embedding_result)
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, embedding_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        mask_cls = self.new_fc(embedding_result)
        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # scores[:, -self.num_unseen:] = scores[:, -self.num_unseen:]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='cosine',
                   temperature=-1):
    assert method in ['dot_product', 'cosine', 'euclidean']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'euclidean':
        return euclidean_dist(key_embeds, ref_embeds)
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def kl_loss(prediction, targets):
    T = 0.1
    return F.kl_div(F.log_softmax(prediction / T, dim=1),
             F.log_softmax(targets / T, dim=1),  # 1.2 0.1 0.2 0.3
             reduction='sum', log_target=True) / prediction.numel()

