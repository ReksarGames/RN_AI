import cv2
import numpy as np
from typing import List, NamedTuple, Tuple

try:
    import numba as nb

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
if NUMBA_AVAILABLE:

    @nb.njit(parallel=True, fastmath=True)
    def numba_convert_new_array(src):
        """Numba优化转换,返回新数组"""
        dst = np.empty_like(src, dtype=np.float32)
        for i in nb.prange(src.shape[0]):
            for j in nb.prange(src.shape[1]):
                for k in range(src.shape[2]):
                    dst[i, j, k] = src[i, j, k] * 0.00392156862745098
        return dst

    @nb.njit(parallel=True, fastmath=True)
    def numba_resize_and_normalize(src, target_h, target_w):
        """Numba优化的resize和归一化"""
        dst = np.empty((target_h, target_w, 3), dtype=np.float32)
        scale_h = src.shape[0] / target_h
        scale_w = src.shape[1] / target_w
        for i in nb.prange(target_h):
            for j in nb.prange(target_w):
                src_i = int(i * scale_h)
                src_j = int(j * scale_w)
                if src_i >= src.shape[0]:
                    src_i = src.shape[0] - 1
                if src_j >= src.shape[1]:
                    src_j = src.shape[1] - 1
                dst[i, j, 0] = src[src_i, src_j, 2] * 0.00392156862745098
                dst[i, j, 1] = src[src_i, src_j, 1] * 0.00392156862745098
                dst[i, j, 2] = src[src_i, src_j, 0] * 0.00392156862745098
        return dst


class Detection(NamedTuple):
    box: Tuple[int, int, int, int]
    confidence: float
    class_id: int


def rect_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def sunone_nms(
    detections: List[Detection],
    nms_threshold: float,
    adaptive_nms: bool,
    max_detections: int = 0,
) -> List[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    if max_detections > 0 and len(detections) > max_detections:
        detections = detections[:max_detections]
    areas = [rect_area(det.box) for det in detections]
    small_area_threshold = 0.0
    if adaptive_nms and areas:
        area_sorted = sorted(areas)
        median_idx = len(area_sorted) // 2
        small_area_threshold = area_sorted[median_idx] * 0.5
    suppressed = [False] * len(detections)
    result: List[Detection] = []
    for i, det_i in enumerate(detections):
        if suppressed[i]:
            continue
        result.append(det_i)
        xi1, yi1, xi2, yi2 = det_i.box
        area_i = areas[i]
        adaptive_threshold = (
            min(nms_threshold * 1.5, 0.8)
            if adaptive_nms and area_i > 0.0 and area_i < small_area_threshold
            else nms_threshold
        )
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            xj1, yj1, xj2, yj2 = detections[j].box
            inter_w = max(0, min(xi2, xj2) - max(xi1, xj1))
            inter_h = max(0, min(yi2, yj2) - max(yi1, yj1))
            if inter_w <= 0 or inter_h <= 0:
                continue
            inter_area = inter_w * inter_h
            union_area = area_i + areas[j] - inter_area
            if union_area <= 0:
                continue
            if inter_area / union_area > adaptive_threshold:
                suppressed[j] = True
    return result


SUNONE_VARIANT_ALIASES = {
    "yolo8": "yolo11",
    "yolo9": "yolo11",
    "yolo10": "yolo10",
    "yolo11": "yolo11",
    "yolo12": "yolo11",
    "auto": "yolo11",
}


def normalize_sunone_variant(variant: str) -> str:
    if not variant:
        return "yolo11"
    normalized = str(variant).strip().lower()
    return SUNONE_VARIANT_ALIASES.get(normalized, "yolo11")


def _prepare_sunone_matrix(output: np.ndarray, features: int) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, features), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % features == 0:
            return arr.reshape(-1, features)
        if features % arr.size == 0:
            return arr.reshape(features, -1)
        return arr.reshape(-1, features)
    if arr.ndim == 2:
        if arr.shape[1] == features:
            return arr
        if arr.shape[0] == features:
            return arr.transpose()
        total = arr.size
        if total % features == 0:
            return arr.reshape(-1, features)
    try:
        return arr.reshape(-1, features)
    except ValueError:
        flat = arr.flatten()
        if flat.size % features == 0:
            return flat.reshape(-1, features)
        return np.resize(flat, (max(1, flat.size // features), features))


def _prepare_sunone_columns(output: np.ndarray, rows_expected: int) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    if arr.size == 0 or rows_expected <= 0:
        return np.empty((rows_expected, 0), dtype=np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        if arr.shape[0] == rows_expected:
            return arr
        if arr.shape[1] == rows_expected:
            return arr.transpose()
    if arr.ndim == 1:
        if arr.size % rows_expected == 0:
            return arr.reshape(rows_expected, -1)
    flat = arr.flatten()
    if flat.size % rows_expected == 0:
        return flat.reshape(rows_expected, -1)
    if arr.ndim == 2:
        return arr
    return np.empty((rows_expected, 0), dtype=np.float32)


def _build_box(cx, cy, ow, oh, img_scale):
    half_ow = 0.5 * ow
    half_oh = 0.5 * oh
    x1 = int((cx - half_ow) * img_scale)
    y1 = int((cy - half_oh) * img_scale)
    x2 = int((cx + half_ow) * img_scale)
    y2 = int((cy + half_oh) * img_scale)
    return (x1, y1, x2, y2)


def sunone_decode_yolo10(
    output: np.ndarray,
    img_scale: float,
    conf_threshold: float,
) -> List[Detection]:
    detections: List[Detection] = []
    matrix = _prepare_sunone_matrix(output, 6)
    if matrix.size == 0:
        return detections
    for det in matrix:
        if det.size < 6:
            continue
        confidence = float(det[4])
        if confidence <= conf_threshold:
            continue
        class_id = int(det[5])
        cx = float(det[0])
        cy = float(det[1])
        dx = float(det[2])
        dy = float(det[3])
        x1 = int(cx * img_scale)
        y1 = int(cy * img_scale)
        x2 = int(dx * img_scale)
        y2 = int(dy * img_scale)
        detections.append(Detection((x1, y1, x2, y2), confidence, class_id))
    return detections


def sunone_decode_yolo11(
    output: np.ndarray,
    num_classes: int,
    img_scale: float,
    conf_threshold: float,
) -> List[Detection]:
    detections: List[Detection] = []
    if num_classes <= 0:
        return detections
    rows_expected = 4 + num_classes
    matrix = _prepare_sunone_columns(output, rows_expected)
    if matrix.size == 0 or matrix.shape[0] < rows_expected:
        return detections
    for i in range(matrix.shape[1]):
        classes_scores = matrix[4 : 4 + num_classes, i]
        if classes_scores.size == 0:
            continue
        class_id = int(np.argmax(classes_scores))
        score = float(classes_scores[class_id])
        if score <= conf_threshold:
            continue
        cx = float(matrix[0, i])
        cy = float(matrix[1, i])
        ow = float(matrix[2, i])
        oh = float(matrix[3, i])
        box = _build_box(cx, cy, ow, oh, img_scale)
        detections.append(Detection(box, score, class_id))
    return detections


def sunone_postprocess(
    pred: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    img_scale: float,
    num_classes: int,
    adaptive_nms: bool,
    max_detections: int = 0,
    variant: str = "yolo11",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = pred.shape
    detections: List[Detection] = []
    variant = normalize_sunone_variant(variant)
    if variant == "yolo10":
        detections = sunone_decode_yolo10(pred, img_scale, conf_threshold)
    else:
        dims = [int(d) for d in pred.shape if int(d) > 4]
        if dims:
            rows = min(dims)
            inferred_classes = max(1, rows - 4)
        else:
            inferred_classes = max(1, int(num_classes))
        detections = sunone_decode_yolo11(
            pred, inferred_classes, img_scale, conf_threshold
        )
    filtered = sunone_nms(detections, iou_threshold, adaptive_nms, max_detections)
    if not filtered:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    boxes_xyxy = np.array([det.box for det in filtered], dtype=np.float32)
    x1, y1, x2, y2 = (
        boxes_xyxy[:, 0],
        boxes_xyxy[:, 1],
        boxes_xyxy[:, 2],
        boxes_xyxy[:, 3],
    )
    centers = np.vstack(
        (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            np.maximum(0.0, x2 - x1),
            np.maximum(0.0, y2 - y1),
        )
    ).T
    boxes = centers
    scores = np.array([det.confidence for det in filtered], dtype=np.float32)
    classes = np.array([det.class_id for det in filtered], dtype=np.int32).reshape(-1)
    return boxes, scores, classes


def draw_boxes(image, boxes, scores, classes):
    for box, score, classe in zip(boxes, scores, classes):
        box = box[:4]
        class_id = np.argmax(classe)
        c_x, c_y, w, h = box.astype(np.int32)
        x_min, y_min, x_max, y_max = convert_box_coordinates(c_x, c_y, w, h)
        color = get_color(class_id)
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        text = f"{class_id} {score:.2f}"
        cv2.putText(
            image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness
        )
    return image


def draw_boxes_v8(image, boxes, scores, classes, class_names=None):
    """
    在图像上绘制检测框
    Args:
        image: 原始图像
        boxes: 边界框坐标 [x1, y1, x2, y2]
        scores: 置信度分数
        classes: 类别ID
        class_names: 类别名称列表(可选)
    Returns:
        image: 绘制了检测框的图像
    """
    image = image.copy()
    height, width = image.shape[:2]
    for box, score, class_id in zip(boxes, scores, classes):
        class_id = int(class_id)
        box = box[:4]
        c_x, c_y, w, h = box.astype(np.int32)
        x1, y1, x2, y2 = convert_box_coordinates(c_x, c_y, w, h)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        color = get_color(class_id)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text = f"{class_id} {score:.2f}"
        cv2.putText(
            image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness
        )
    return image


def get_color(class_id):
    """
    为每个类别生成唯一且易于区分的颜色
    """
    predefined_colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 255, 128),
        (255, 0, 128),
    ]
    if class_id < len(predefined_colors):
        return predefined_colors[class_id]
    np.random.seed(class_id)
    color = tuple(map(int, np.random.randint(0, 255, size=3)))
    return color


def draw_fps(image, fps):
    text = f"FPS: {fps:.2f}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    return image


def convert_box_coordinates(center_x, center_y, width, height):
    x_min = int(center_x - width / 2)
    y_min = int(center_y - height / 2)
    x_max = int(center_x + width / 2)
    y_max = int(center_y + height / 2)
    return (x_min, y_min, x_max, y_max)


def convert_box_coordinates_float(center_x, center_y, width, height):
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    return (x_min, y_min, x_max, y_max)


def nms_v8(pred, conf_thres, iou_thres, adaptive_nms=True):
    """
    处理YOLO预测输出，执行置信度过滤和NMS，返回最终的检测结果。
    优化版本：对小目标使用更宽松的IoU阈值，减少误删除。

    Args:
        pred: 模型输出的特征图 (1, 84, 8400)。
        conf_thres: 置信度阈值。
        iou_thres: IoU阈值。

    Returns:
        boxes: 最终保留的边界框坐标 (x1, y1, x2, y2)。
        scores: 置信度分数。
        classes: 类别标签。
    """

    def xywh2xyxy(box):
        x, y, w, h = box.T
        return np.vstack([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).T

    def calculate_iou(box, boxes):
        x1, y1, x2, y2 = box
        boxes_x1, boxes_y1, boxes_x2, boxes_y2 = boxes.T
        inter_w = np.maximum(0, np.minimum(x2, boxes_x2) - np.maximum(x1, boxes_x1))
        inter_h = np.maximum(0, np.minimum(y2, boxes_y2) - np.maximum(y1, boxes_y1))
        inter_area = inter_w * inter_h
        box_area = (x2 - x1) * (y2 - y1)
        boxes_area = (boxes_x2 - boxes_x1) * (boxes_y2 - boxes_y1)
        union_area = box_area + boxes_area - inter_area
        return inter_area / union_area

    pred = np.squeeze(pred)
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = np.transpose(pred, (1, 0))
    pred_class = pred[:, 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    pred = pred[pred[:, 4] > conf_thres]
    if len(pred) == 0:
        return (np.empty((0, 4)), np.empty((0,)), np.empty((0,)))
    pred_boxes = pred[:, :4]
    pred_scores = pred[:, 4]
    pred_classes = np.argmax(pred[:, 5:], axis=-1)
    MAX_DETS = 300
    if pred_scores.shape[0] > MAX_DETS:
        idx = np.argpartition(pred_scores, -MAX_DETS)[-MAX_DETS:]
        pred_boxes = pred_boxes[idx]
        pred_scores = pred_scores[idx]
        pred_classes = pred_classes[idx]
    start_x = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    start_y = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    end_x = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    end_y = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(pred_scores)
    median_area = np.median(areas)
    small_target_threshold = median_area * 0.5
    keep = []
    while order.size > 0:
        index = order[-1]
        keep.append(index)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        if adaptive_nms:
            current_area = areas[index]
            if current_area < small_target_threshold:
                adaptive_iou_thres = min(iou_thres * 1.5, 0.8)
            else:
                adaptive_iou_thres = iou_thres
        else:
            adaptive_iou_thres = iou_thres
        left = np.where(ratio <= adaptive_iou_thres)
        order = order[left]
    if not keep:
        return ([], [], [])
    keep = np.array(keep)
    boxes = pred_boxes[keep]
    scores = pred_scores[keep]
    classes = pred_classes[keep]
    return (boxes, scores, classes)


def nms_v5(pred, confidence_threshold, iou_threshold, class_num):
    pred = np.array(pred)
    if len(pred) == 0:
        return ([], [], [])
    pred = pred.reshape(-1, 5 + class_num)
    boxes = pred[:, :4]
    scores = pred[:, 4]
    class_probs = pred[:, 5:]
    mask = scores > confidence_threshold
    if not np.any(mask):
        return ([], [], [])
    boxes = boxes[mask]
    scores = scores[mask]
    class_probs = class_probs[mask]
    start_x = boxes[:, 0] - boxes[:, 2] / 2
    start_y = boxes[:, 1] - boxes[:, 3] / 2
    end_x = boxes[:, 0] + boxes[:, 2] / 2
    end_y = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.stack([start_x, start_y, end_x, end_y], axis=1)
    areas = (end_x - start_x) * (end_y - start_y)
    results_boxes, results_scores, results_classes = ([], [], [])
    for cls_idx in range(class_num):
        cls_scores = class_probs[:, cls_idx]
        mask = cls_scores > 0
        if not np.any(mask):
            continue
        cls_boxes = boxes[mask]
        cls_scores = scores[mask] * cls_scores[mask]
        order = np.argsort(cls_scores)[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
            yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
            xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
            yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])
            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            intersection = inter_w * inter_h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            order = order[1:][iou <= iou_threshold]
        results_boxes.append(cls_boxes[keep])
        results_scores.append(cls_scores[keep])
        results_classes.extend([cls_idx] * len(keep))
    if results_boxes:
        results_boxes = np.vstack(results_boxes)
        results_scores = np.hstack(results_scores)
    else:
        results_boxes, results_scores = (np.array([]), np.array([]))
    return (results_boxes, results_scores, np.array(results_classes))


def nms(pred, confidence_threshold, iou_threshold, class_num):
    pred = np.array(pred)
    if len(pred) == 0:
        return ([], [], [])
    pred = pred.reshape(-1, 5 + class_num)
    boxes = pred[:, :4]
    scores = pred[:, 4]
    classes = pred[:, 5:]
    indexs = np.where(scores > confidence_threshold)
    boxes, scores, classes = (boxes[indexs], scores[indexs], classes[indexs])
    start_x = boxes[:, 0] - boxes[:, 2] / 2
    start_y = boxes[:, 1] - boxes[:, 3] / 2
    end_x = boxes[:, 0] + boxes[:, 2] / 2
    end_y = boxes[:, 1] + boxes[:, 3] / 2
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(scores)
    keep = []
    while order.size > 0:
        index = order[-1]
        keep.append(index)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio <= iou_threshold)
        order = order[left]
    if not keep:
        return ([], [], [])
    keep = np.array(keep)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    return (boxes, scores, classes)


def read_img(img_data, size=(320, 320)):
    target_w, target_h = size
    # if NUMBA_AVAILABLE and target_w == 640 and (target_h == 640):
    #     try:
    #         resized_img = cv2.resize(img_data, (target_w, target_h))
    #         normalized_img = numba_convert_new_array(resized_img)
    #         processed_img = normalized_img[:, :, [2, 1, 0]]
    #         blob = np.transpose(processed_img, (2, 0, 1))
    #         blob = np.expand_dims(blob, axis=0)
    #         return blob
    #     except Exception as e:
    #         pass
    blob = cv2.dnn.blobFromImage(
        image=img_data,
        scalefactor=0.00392156862745098,
        size=(target_w, target_h),
        mean=(0.0, 0.0, 0.0),
        swapRB=True,
        crop=False,
    )
    return blob
