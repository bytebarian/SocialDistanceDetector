import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_cxcy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [x_c, y_c]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def rescale_centroids(out_bbox, size):
    img_w, img_h = size
    c = box_cxcy(out_bbox)
    c = c * torch.tensor([img_w, img_h], dtype=torch.float32)
    return c

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    centroids_scaled = rescale_centroids(outputs['pred_boxes'][0, keep], im.size)

    result = []

    for p, (xmin, ymin, xmax, ymax), (xc, yc) in zip(probas[keep], bboxes_scaled.tolist(), centroids_scaled.tolist()):
        cl = p.argmax()
        if cl == 1:
            result.append((p[cl], (xmin, ymin, xmax, ymax), (xc, yc)))

    return result