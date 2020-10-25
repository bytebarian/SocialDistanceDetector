import numpy as np
import torch
from torchvision.ops.boxes import batched_nms
import torchvision.transforms as T


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform, device='cpu'):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).to(device)
    model.to(device)
    # propagate through the model
    outputs = model(img)
    # keep only predictions with confidence above threshold
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1].cpu()
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0,].cpu(), im.size)
    return probas, bboxes_scaled


def filter_boxes(scores, boxes, confidence=0.7, apply_nms=True, iou=0.5):
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    results = []

    for i in range(boxes.shape[0]):
        class_id = scores[i].argmax()
        x0, y0, x1, y1 = boxes[i]
        if class_id == 1:
            width = x1 - x0
            height = y1 - y0
            centerX = int(x0 + (width / 2))
            centerY = int(y0 + (height / 2))
            r = ((x0, y0, int(x0 + width), int(y0 + height)), (centerX, centerY))
            results.append(r)

    return results
