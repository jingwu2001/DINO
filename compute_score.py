import argparse
from typing import List, Tuple, Dict


def parse_predictions(path: str) -> List[Dict]:
    """Load prediction boxes; expects: image_id class score x1 y1 x2 y2."""
    preds = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            image_id = parts[0]
            cls = parts[1]
            score = float(parts[2])
            x1, y1, x2, y2 = map(float, parts[-4:])
            preds.append(
                {"image_id": image_id, "cls": cls, "score": score, "bbox": (x1, y1, x2, y2)}
            )
    # sort by confidence descending
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def parse_ground_truths(path: str) -> List[Dict]:
    """Load ground-truth boxes; accepts an optional confidence column."""
    gts = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            image_id = parts[0]
            cls = parts[1]
            x1, y1, x2, y2 = map(float, parts[-4:])
            gts.append({"image_id": image_id, "cls": cls, "bbox": (x1, y1, x2, y2), "used": False})
    return gts


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def greedy_match(preds: List[Dict], gts: List[Dict], iou_threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    tps, fps = [], []
    for pred in preds:
        candidates = [
            (idx, gt) for idx, gt in enumerate(gts)
            if not gt["used"] and gt["image_id"] == pred["image_id"] and gt["cls"] == pred["cls"]
        ]

        best_iou, best_idx = 0.0, None
        for idx, gt in candidates:
            current_iou = iou(pred["bbox"], gt["bbox"])
            if current_iou > best_iou:
                best_iou, best_idx = current_iou, idx

        if best_idx is not None and best_iou >= iou_threshold:
            gts[best_idx]["used"] = True
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    return tps, fps


def trapezoidal_ap(recalls: List[float], precisions: List[float]) -> float:
    # Add starting and ending points as described in the scoring rule.
    mrec = [0.0] + recalls + [1.0]
    mpre = [1.0] + precisions + [0.0]

    area = 0.0
    for i in range(len(mrec) - 1):
        area += (mrec[i + 1] - mrec[i]) * (mpre[i + 1] + mpre[i]) / 2.0
    return area


def compute_score(pred_path: str, gt_path: str, iou_threshold: float = 0.5) -> float:
    preds = parse_predictions(pred_path)
    gts = parse_ground_truths(gt_path)

    if not preds or not gts:
        return 0.0

    tps, fps = greedy_match(preds, gts, iou_threshold=iou_threshold)
    total_gt = len(gts)

    recalls, precisions = [], []
    cum_tp = cum_fp = 0
    for tp, fp in zip(tps, fps):
        cum_tp += tp
        cum_fp += fp
        recalls.append(cum_tp / total_gt)
        precisions.append(cum_tp / (cum_tp + cum_fp))

    return trapezoidal_ap(recalls, precisions)


def main():
    parser = argparse.ArgumentParser(description="Compute detection score with greedy IoU matching.")
    parser.add_argument("--predictions", default="predictions_test.txt", help="Path to predictions file.")
    parser.add_argument("--ground_truth", default="gt_test.txt", help="Path to ground-truth file.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for a true positive.")
    args = parser.parse_args()

    score = compute_score(args.predictions, args.ground_truth, iou_threshold=args.iou_threshold)
    print(f"Score: {score:.3f}")


if __name__ == "__main__":
    main()
