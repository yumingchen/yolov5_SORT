import cv2
import numpy as np


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    thickness = ( line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)  # line/font thickness
    point1, point2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, point1, point2, color, thickness=thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(thickness - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=tf)[0]
        point2 = point1[0] + t_size[0], point1[1] - t_size[1] - 3
        cv2.rectangle(img, point1, point2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (point1[0], point1[1] - 2), 0, thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA,)


def preprocess_image(image_raw, INPUT_W, INPUT_H):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    origin_h, origin_w, channel = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    ratio_w = INPUT_W / origin_w
    ratio_h = INPUT_H / origin_h
    if ratio_h > ratio_w:
        tw = INPUT_W
        th = int(ratio_w * origin_h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)
        ty2 = INPUT_H - th - ty1
    else:
        tw = int(ratio_h * origin_w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, origin_h, origin_w


def xywh2xyxy(origin_h, origin_w, det, INPUT_W, INPUT_H):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        det:          A boxes tensor, each row is a box [center_x, center_y, w, h]
    return:
        bbox_xyxy:          A boxes tensor, each row is a box [x1, y1, x2, y2]
    """
    # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    bbox_xyxy = np.zeros_like(det)
    ratio_w = INPUT_W / origin_w
    ratio_h = INPUT_H / origin_h
    if ratio_h > ratio_w:
        bbox_xyxy[:, 0] = det[:, 0] - det[:, 2] / 2
        bbox_xyxy[:, 2] = det[:, 0] + det[:, 2] / 2
        bbox_xyxy[:, 1] = det[:, 1] - det[:, 3] / 2 - (INPUT_H - ratio_w * origin_h) / 2
        bbox_xyxy[:, 3] = det[:, 1] + det[:, 3] / 2 - (INPUT_H - ratio_w * origin_h) / 2
        bbox_xyxy /= ratio_w
    else:
        bbox_xyxy[:, 0] = det[:, 0] - det[:, 2] / 2 - (INPUT_W - ratio_h * origin_w) / 2
        bbox_xyxy[:, 2] = det[:, 0] + det[:, 2] / 2 - (INPUT_W - ratio_h * origin_w) / 2
        bbox_xyxy[:, 1] = det[:, 1] - det[:, 3] / 2
        bbox_xyxy[:, 3] = det[:, 1] + det[:, 3] / 2
        bbox_xyxy /= ratio_h

    return bbox_xyxy


def nms_np(pred, thresh):
    '''
    param pred: ndarray [N,6], eg:[xmin,ymin,xmax,ymax,score, classid]
    param thresh: float
    return keep: list[index]
    '''
    x1 = pred[:, 0]
    y1 = pred[:, 1]
    x2 = pred[:, 2]
    y2 = pred[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = pred[:, 4].argsort()[::-1]

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        over = (w * h) / (area[i] + area[order[1:]] - w * h)
        index = np.where(over <= thresh)[0]
        order = order[index + 1]
    return keep


def post_process(output, origin_h, origin_w, CONF_THRESH, IOU_THRESHOLD, INPUT_W, INPUT_H):
    """
    description: postprocess the prediction
    param:
        output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
        origin_h:   height of original image
        origin_w:   width of original image
    return:
        result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
        result_scores: finally scores, a tensor, each element is the score correspoing to box
        result_classid: finally classid, a tensor, each element is the classid correspoing to box
    """
    # Get the num of boxes detected
    num = int(output[0])

    # Reshape to a two dimentional ndarray
    pred = np.reshape(output[1:], (-1, 6))[:num, :]
    # Choose those boxes that score > CONF_THRESH
    conf_id = pred[:, 4] > CONF_THRESH
    pred = pred[conf_id, :]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    pred[:, :4] = xywh2xyxy(origin_h, origin_w, pred[:, :4], INPUT_W, INPUT_H)

    #####################NMS in all  boxes ##################
    keep = nms_np(pred, IOU_THRESHOLD)
    nms_pred = pred[keep]  # [x1,y1,x2,y2,score,cls_id]

    return nms_pred


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def get_classes_name():
    categories = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    return categories


# 计算由A，B，C三点构成的向量AC，AB之间的关系
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# 检测AB和CD两条直线是否相交
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def show_statistics_info(frame, counts, counts_left, counts_right, target_id, categories, colors):
    fontFace = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fontScale = 1.0
    start_width = 10
    height_start = 50
    height_gap = 15
    thickness = 1
    text = 'sum: {}'.format(sum(counts.values()))
    cv2.putText(frame, text, (start_width, height_start), fontFace, fontScale, (0, 255, 0), thickness, lineType=line)
    text = 'sum_left: {}'.format(sum(counts_left.values()))
    cv2.putText(frame, text, (start_width + 150, height_start), fontFace, fontScale, (0, 255, 0), thickness,
                lineType=line)
    text = 'sum_right: {}'.format(sum(counts_right.values()))
    cv2.putText(frame, text, (start_width + 350, height_start), fontFace, fontScale, (0, 255, 0), thickness)
    for i, cls_id in enumerate(target_id):
        text = '{}: {}'.format(categories[cls_id], counts[cls_id])
        cv2.putText(frame, text, (start_width, height_start + (i + 1) * height_gap), fontFace, fontScale,
                    colors[cls_id],
                    thickness)
        text = '{}: {}'.format(categories[cls_id], counts_left[cls_id])
        cv2.putText(frame, text, (start_width + 150, height_start + (i + 1) * height_gap), fontFace, fontScale,
                    colors[cls_id],
                    thickness)
        text = '{}: {}'.format(categories[cls_id], counts_right[cls_id])
        cv2.putText(frame, text, (start_width + 350, height_start + (i + 1) * height_gap), fontFace, fontScale,
                    colors[cls_id],
                    thickness)
    return frame