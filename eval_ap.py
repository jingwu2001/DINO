#!/usr/bin/env python3
import argparse
import numpy as np


def iou(box_a, box_b):
    """Compute IoU between two boxes: [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_ground_truth(gt_path):
    """
    gt_test.txt format (one per line):
    image_id class_id dummy x1 y1 x2 y2
    """
    gts = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, cls_id, _dummy, x1, y1, x2, y2 = line.split()
            key = (img_id, cls_id)
            box = [float(x1), float(y1), float(x2), float(y2)]
            gts.setdefault(key, []).append(box)
    return gts


def load_predictions(pred_path):
    """
    predictions_test.txt format (one per line):
    image_id class_id score x1 y1 x2 y2
    """
    preds = []
    with open(pred_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, cls_id, score, x1, y1, x2, y2 = line.split()
            preds.append(
                {
                    "img": img_id,
                    "cls": cls_id,
                    "score": float(score),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

    # sort by confidence, descending
    preds.sort(key=lambda p: p["score"], reverse=True)
    return preds


def match_and_get_tp_fp(preds, gts, iou_thresh=0.5):
    """
    Run the greedy matching (each GT can be matched at most once)
    and return TP/FP arrays and total number of GT boxes.
    """
    matches = {k: [False] * len(v) for k, v in gts.items()}
    tps = []
    fps = []
    total_gt = sum(len(v) for v in gts.values())

    for p in preds:
        key = (p["img"], p["cls"])
        gt_boxes = gts.get(key)

        if not gt_boxes:  # no GT for this image/class
            tps.append(0)
            fps.append(1)
            continue

        gt_matches = matches[key]
        best_iou = 0.0
        best_idx = -1

        for i, gt_box in enumerate(gt_boxes):
            if gt_matches[i]:
                continue  # already matched
            iou_val = iou(p["box"], gt_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i

        if best_iou >= iou_thresh and best_idx >= 0:
            gt_matches[best_idx] = True
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    return np.array(tps, dtype=float), np.array(fps, dtype=float), total_gt


def ap_from_tp_fp(tps, fps, total_gt):
    """Compute AP from TP/FP arrays using area under PR curve."""
    if total_gt == 0:
        return 0.0

    tps_c = np.cumsum(tps)
    fps_c = np.cumsum(fps)

    recalls = tps_c / float(total_gt)
    precisions = tps_c / np.maximum(tps_c + fps_c, np.finfo(float).eps)

    # add (0,1) and (1,0) endpoints
    rc = np.concatenate(([0.0], recalls, [1.0]))
    pc = np.concatenate(([1.0], precisions, [0.0]))

    ap = np.trapz(pc, rc)
    return ap


def main():
    parser = argparse.ArgumentParser(
        description="Compute AP@IoU and optionally loop over confidence thresholds."
    )
    parser.add_argument("predictions", help="Path to predictions_test.txt")
    parser.add_argument("ground_truth", help="Path to gt_test.txt")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for a match (default: 0.5)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        help=(
            "Optional list of confidence thresholds to evaluate. "
            "For each t, only predictions with score >= t are kept."
        ),
    )

    args = parser.parse_args()

    gts = load_ground_truth(args.ground_truth)
    preds = load_predictions(args.predictions)

    # Overall AP with all predictions (like original script)
    tps, fps, total_gt = match_and_get_tp_fp(preds, gts, args.iou)
    ap = ap_from_tp_fp(tps, fps, total_gt)
    print(f"Overall AP@{args.iou:.2f} (no confidence filtering): {ap:.4f}")

    # Optional threshold loop
    if args.thresholds:
        thresholds = sorted(set(args.thresholds))
        print("\nThreshold loop (keeping predictions with score >= threshold):")
        print("thr\t#pred\tTP\tFP\tPrecision\tRecall\tF1\tAP")

        for thr in thresholds:
            filtered = [p for p in preds if p["score"] >= thr]

            if not filtered:
                print(
                    f"{thr:.3f}\t0\t0\t0\t0.0000\t\t0.0000\t0.0000\t0.0000"
                )
                continue

            tps_thr, fps_thr, total_gt_thr = match_and_get_tp_fp(
                filtered, gts, args.iou
            )

            tp_last = tps_thr.sum()
            fp_last = fps_thr.sum()

            precision = (
                tp_last / (tp_last + fp_last) if (tp_last + fp_last) > 0 else 0.0
            )
            recall = tp_last / total_gt_thr if total_gt_thr > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            ap_thr = ap_from_tp_fp(tps_thr, fps_thr, total_gt_thr)

            print(
                f"{thr:.3f}\t{len(filtered)}\t{int(tp_last)}\t{int(fp_last)}\t"
                f"{precision:.4f}\t\t{recall:.4f}\t{f1:.4f}\t{ap_thr:.4f}"
            )


if __name__ == "__main__":
    main()