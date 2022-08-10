import os
import sys
import types

import torch
from torch.nn import functional as F

from src.infer.base_predictor import BasePredictor


def stripped_forward(self, images):
    images = (images - self.pixel_mean) / self.pixel_std

    features = self.backbone(images)
    outputs = self.sem_seg_head(features)

    mask_cls_results = outputs["pred_logits"]
    mask_pred_results = outputs["pred_masks"]
    del outputs
    # upsample masks
    mask_pred_results = F.interpolate(
        mask_pred_results,
        size=(images.shape[-2], images.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )

    mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
    mask_pred = mask_pred_results.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred).argmax(dim=1)


class DetectronPredictor(BasePredictor):
    """Class for Detectron2 based predictor."""

    @classmethod
    def setup_cfg(cls, our_model_cfg):
        # load config from file and command-line arguments
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        maskformer_home_path = os.environ.get('MASKFORMER_HOME')
        mask2former_home_path = os.environ.get('MASK2FORMER_HOME')
        try:
            sys.path.insert(1, mask2former_home_path)
            from mask2former import add_maskformer2_config
            add_maskformer2_config(cfg)
        except:
            print("[ERROR] Coudn't import mask2former, please set MASK2FORMER_HOME env var")
            pass
        try:
            sys.path.insert(1, maskformer_home_path)
            from mask_former import add_mask_former_config
            add_mask_former_config(cfg)
        except:
            pass
        cfg.merge_from_file(our_model_cfg.config_file)
        cfg.merge_from_list(our_model_cfg.opts)
        cfg.freeze()
        return cfg

    def __init__(self, our_model_cfg):
        super().__init__(our_model_cfg)
        self.det2_cfg = self.setup_cfg(our_model_cfg)
        self.use_fp16 = our_model_cfg.use_fp16
        # Load model
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        self.model = build_model(self.det2_cfg)
        self.model.eval()
        self.model._forward = self.model.forward
        self.model.forward = types.MethodType(stripped_forward, self.model)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(our_model_cfg.weights_path)
        assert our_model_cfg.image_format == self.det2_cfg.INPUT.FORMAT.lower(), \
            f'Input format must be the same! Got {our_model_cfg.image_format} and {self.det2_cfg.INPUT.FORMAT.lower()}'
    
    def __call__(self, batch):
        with torch.no_grad():
            with torch.cuda.amp.autocast(self.use_fp16):
                return self.model(batch)
