import torch
from torch.nn import functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

def forward(self, images):
    images = (images - self.pixel_mean) / self.pixel_std

    features = self.backbone(images)
    outputs = self.sem_seg_head(features)

    mask_cls_results = outputs["pred_logits"]
    mask_pred_results = outputs["pred_masks"]
    # upsample masks
    mask_pred_results = F.interpolate(
        mask_pred_results,
        size=(images.shape[-2], images.shape[-1]),
        mode="bilinear",
        align_corners=False,
    )

    del outputs

    mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]
    mask_pred = mask_pred_results.sigmoid()
    return torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred).argmax(dim=1)
