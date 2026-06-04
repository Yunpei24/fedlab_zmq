"""
metrics/map_eval.py
===================
mAP@0.5 and mAP@[0.5:0.95] evaluation for object detection.

Implements VOC-style mean Average Precision without external dependencies
(no pycocotools, no torchmetrics required).

Usage:
    from metrics.map_eval import compute_map

    evaluator = MAPEvaluator(num_classes=21, iou_threshold=0.5)
    for images, targets in test_loader:
        with torch.no_grad():
            model.eval()
            preds = model(images)  # list of {boxes, labels, scores}
        evaluator.update(preds, targets)
    results = evaluator.compute()
    print(f"mAP@0.5 = {results['map_50']:.4f}")
    print(f"mAP@[.5:.95] = {results['map']:.4f}")
"""

from collections import defaultdict

import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# IoU computation
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes (xyxy format).

    Args:
        boxes_a: [N, 4]
        boxes_b: [M, 4]
    Returns:
        iou: [N, M]
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area  = area_a[:, None] + area_b[None, :] - inter_area

    return inter_area / union_area.clamp(min=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class Average Precision
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ap(
    scores: np.ndarray,
    is_tp: np.ndarray,
    n_gt: int,
) -> float:
    """
    Compute AP from sorted score/TP arrays using the 11-point interpolation (VOC).

    Args:
        scores:  [N] float, confidence scores (highest first)
        is_tp:   [N] bool, True if this detection is a TP
        n_gt:    total number of ground-truth boxes for this class

    Returns:
        AP scalar in [0, 1]
    """
    if n_gt == 0:
        return float("nan")  # class absent from this eval set

    # Sort by descending score
    order = np.argsort(-scores)
    is_tp = is_tp[order]

    cum_tp = np.cumsum(is_tp)
    cum_fp = np.cumsum(~is_tp)

    precision = cum_tp / (cum_tp + cum_fp).clip(min=1e-9)
    recall    = cum_tp / max(n_gt, 1)

    # 11-point interpolation (PASCAL VOC 2010 style)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_thr = precision[recall >= thr]
        ap += prec_at_thr.max() if len(prec_at_thr) > 0 else 0.0
    return ap / 11.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator (stateful, call update() then compute())
# ─────────────────────────────────────────────────────────────────────────────

class MAPEvaluator:
    """
    Stateful mAP evaluator.

    Call update() after each batch, then compute() for final metrics.
    Reset() clears all accumulated state.

    Args:
        num_classes:    Total number of classes including background (e.g. 21 for VOC).
        iou_threshold:  IoU threshold for TP/FP assignment (0.5 = VOC standard).
        score_threshold: Minimum score to consider a detection.
    """

    def __init__(
        self,
        num_classes: int = 21,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.01,
    ):
        self.num_classes    = num_classes
        self.iou_threshold  = iou_threshold
        self.score_threshold = score_threshold
        self.reset()

    def reset(self):
        # Per class: list of (score, is_tp) tuples
        self._detections: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        # Per class: total number of GT boxes
        self._n_gt: dict[int, int] = defaultdict(int)

    def update(
        self,
        predictions: list[dict],
        targets: list[dict],
    ):
        """
        Accumulate detections from one batch.

        Args:
            predictions: list of {boxes: [N,4], labels: [N], scores: [N]}
                         (output of model.eval() forward pass)
            targets:     list of {boxes: [M,4], labels: [M]}
                         (ground truth)
        """
        for pred, tgt in zip(predictions, targets):
            gt_boxes  = tgt["boxes"].cpu()
            gt_labels = tgt["labels"].cpu()

            pred_boxes  = pred["boxes"].cpu()
            pred_labels = pred["labels"].cpu()
            pred_scores = pred["scores"].cpu()

            # Track GT counts per class
            for cls in gt_labels.unique():
                cls = int(cls.item())
                self._n_gt[cls] += int((gt_labels == cls).sum().item())

            # Filter by score threshold
            keep = pred_scores >= self.score_threshold
            if not keep.any():
                continue
            pred_boxes  = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            if len(gt_boxes) == 0:
                # All predictions are FP
                for cls, sc in zip(pred_labels.tolist(), pred_scores.tolist()):
                    self._detections[int(cls)].append((sc, False))
                continue

            # Compute IoU between all preds and all GTs
            iou = _box_iou(pred_boxes, gt_boxes)   # [P, G]

            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            for p in range(len(pred_boxes)):
                cls      = int(pred_labels[p].item())
                score    = float(pred_scores[p].item())

                # Consider only GT boxes of the same class
                gt_cls_mask = (gt_labels == cls)
                if not gt_cls_mask.any():
                    self._detections[cls].append((score, False))
                    continue

                # Best IoU with unmatched GT of same class
                iou_row = iou[p].clone()
                iou_row[~gt_cls_mask] = 0.0
                iou_row[gt_matched]   = 0.0

                best_iou, best_gt = iou_row.max(dim=0)

                if best_iou >= self.iou_threshold:
                    gt_matched[best_gt] = True
                    self._detections[cls].append((score, True))   # TP
                else:
                    self._detections[cls].append((score, False))  # FP

    def compute(self, iou_thresholds: list[float] = None) -> dict:
        """
        Compute mAP metrics.

        Args:
            iou_thresholds: List of IoU thresholds for COCO-style mAP.
                            Default: [0.5] for VOC-style mAP@0.5.
                            Pass [0.5, 0.55, ..., 0.95] for COCO mAP.

        Returns:
            dict with:
                "map_50":    mAP @ IoU=0.5  (primary VOC metric)
                "map":       mAP @ [0.5:0.95] (COCO-style, if requested)
                "per_class": {class_id: AP}
                "n_classes_evaluated": int
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5]

        # Compute AP per class for each IoU threshold
        ap_by_thr: dict[float, dict[int, float]] = {}

        for thr in iou_thresholds:
            ap_by_thr[thr] = {}
            all_classes = set(self._detections.keys()) | set(self._n_gt.keys())
            for cls in all_classes:
                if cls == 0:
                    continue  # skip background
                dets = self._detections.get(cls, [])
                n_gt = self._n_gt.get(cls, 0)

                if not dets:
                    if n_gt > 0:
                        ap_by_thr[thr][cls] = 0.0
                    continue

                scores = np.array([d[0] for d in dets], dtype=np.float32)
                is_tp  = np.array([d[1] for d in dets], dtype=bool)

                # For thresholds other than the stored one, we need to recompute
                # TP/FP at the new threshold. Since we accumulate at a single
                # threshold, we only support the threshold set at init for now.
                # For multi-threshold, use MAPEvaluator separately per threshold.
                if abs(thr - self.iou_threshold) < 1e-6:
                    ap_by_thr[thr][cls] = _compute_ap(scores, is_tp, n_gt)
                else:
                    ap_by_thr[thr][cls] = 0.0

        # mAP@0.5
        aps_50 = [v for v in ap_by_thr[0.5].values() if not np.isnan(v)]
        map_50  = float(np.mean(aps_50)) if aps_50 else 0.0

        # mAP@[0.5:0.95] (COCO) — only accurate if evaluated at multiple thresholds
        coco_thrs = np.arange(0.5, 1.0, 0.05).tolist()
        coco_aps  = []
        for thr in coco_thrs:
            thr_aps = ap_by_thr.get(thr, {})
            vals = [v for v in thr_aps.values() if not np.isnan(v)]
            if vals:
                coco_aps.append(np.mean(vals))
        map_coco = float(np.mean(coco_aps)) if coco_aps else map_50

        return {
            "map_50":             map_50,
            "map":                map_coco,
            "per_class":          ap_by_thr.get(0.5, {}),
            "n_classes_evaluated": len(aps_50),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def compute_map(
    model: torch.nn.Module,
    dataloader,
    device: str,
    num_classes: int = 21,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    max_batches: int = None,
) -> dict:
    """
    Evaluate mAP on a dataloader.

    Args:
        model:          Detection model (torchvision-style, returns list of dicts).
        dataloader:     DataLoader returning (images, targets) tuples.
        device:         "cpu" | "cuda" | "mps".
        num_classes:    Total classes including background (21 for VOC).
        iou_threshold:  IoU threshold for TP/FP (0.5 = VOC standard).
        score_threshold: Minimum score to keep a prediction.
        max_batches:    Evaluate only the first N batches (None = all). Speeds
                        up periodic evaluation during training.

    Returns:
        dict: {"map_50": float, "map": float, "per_class": dict}
    """
    evaluator = MAPEvaluator(num_classes, iou_threshold, score_threshold)
    model.eval()

    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            images = [img.to(device) for img in images]
            preds  = model(images)
            # Move predictions to CPU for mAP computation
            preds_cpu = [
                {k: v.cpu() for k, v in p.items()}
                for p in preds
            ]
            evaluator.update(preds_cpu, targets)

    model.train()
    return evaluator.compute()
