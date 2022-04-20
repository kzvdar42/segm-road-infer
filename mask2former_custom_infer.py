
from torch.nn import functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

def forward(self, images):
    images = images.to(self.device)
    images = (images - self.pixel_mean) / self.pixel_std
    # images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in image_batch]

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

    processed_results = []
    for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):
        processed_results.append({})
        # semantic segmentation inference
        processed_results[-1]["sem_seg"] = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)

    return processed_results
