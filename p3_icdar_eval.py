# p3_icdar_eval.py
# -*- coding: utf-8 -*-
"""
Phase 1 (formal): DBNet + real ICDAR2015 GT evaluation
======================================================
Uses REAL ICDAR2015 COCO-format annotations as ground truth
(no more pseudo-GT). Computes detection Precision/Recall/H-mean
for normal vs lowlight on the same test set.

Requires: ICDAR2015 prepared via prepare_dataset.py
Output:  data/icdar2015/textdet_test.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shapely.geometry import Polygon, box
from mmocr.apis import MMOCRInferencer
from mmocr.utils import poly_iou


def load_icdar_gt(ann_file, data_dir):
    """
    Load ICDAR2015 COCO-style test annotations.
    Returns: list of dicts with keys: file_name, img_path, gt_polygons, ignore_flags
    """
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id -> image_info map
    img_map = {}
    for img in coco["images"]:
        img_map[img["id"]] = img

    # Collect anns per image
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        img_anns.setdefault(img_id, []).append(ann)

    results = []
    for img_id, img_info in img_map.items():
        anns = img_anns.get(img_id, [])
        polygons = []
        ignores = []
        for ann in anns:
            seg = ann.get("segmentation", [[]])
            if seg and seg[0]:
                polygons.append(seg[0])  # [x1,y1,x2,y2,...]
                ignores.append(ann.get("iscrowd", False) or ann.get("ignore", False))
        if polygons:
            img_path = os.path.join(data_dir, img_info["file_name"])
            results.append(dict(
                file_name=img_info["file_name"],
                img_path=img_path,
                gt_polygons=polygons,
                ignore_flags=ignores,
            ))
    return results


def shapely_from_poly(flat_poly):
    """Convert a flat polygon list [x1,y1,...] to shapely Polygon."""
    if len(flat_poly) < 6:
        return None
    coords = [(flat_poly[i], flat_poly[i+1]) for i in range(0, len(flat_poly), 2)]
    try:
        p = Polygon(coords)
        return p if p.is_valid and not p.is_empty else None
    except:
        return None


def compute_hmean(gt_data, pred_polygons, match_iou_thr=0.5):
    """
    Compute detection Precision/Recall/H-mean for a single image.
    gt_data: dict with gt_polygons and ignore_flags
    pred_polygons: list of flat polygon lists from DBNet
    """
    # Filter non-ignored GT
    valid_gt_polys = []
    for poly, ig in zip(gt_data["gt_polygons"], gt_data["ignore_flags"]):
        if not ig:
            sp = shapely_from_poly(poly)
            if sp:
                valid_gt_polys.append(sp)

    gt_num = len(valid_gt_polys)

    # Convert predictions
    pred_polys = []
    for poly in pred_polygons:
        sp = shapely_from_poly(poly)
        if sp:
            pred_polys.append(sp)
    pred_num = len(pred_polys)

    if gt_num == 0 and pred_num == 0:
        return dict(precision=1.0, recall=1.0, hmean=1.0, gt_num=0, pred_num=0, match_num=0)
    if gt_num == 0:
        return dict(precision=0.0, recall=1.0, hmean=0.0, gt_num=0, pred_num=pred_num, match_num=0)
    if pred_num == 0:
        return dict(precision=0.0, recall=0.0, hmean=0.0, gt_num=gt_num, pred_num=0, match_num=0)

    # IoU matrix
    iou_mat = np.zeros((gt_num, pred_num))
    for i, g in enumerate(valid_gt_polys):
        for j, p in enumerate(pred_polys):
            try:
                iou_mat[i, j] = poly_iou(g, p)
            except:
                iou_mat[i, j] = 0.0

    # Vanilla matching
    matched_gt, matched_pred = set(), set()
    for gi, pi in zip(*np.where(iou_mat >= match_iou_thr)):
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    m = len(matched_gt)
    precision = m / pred_num
    recall = m / gt_num
    d = precision + recall
    hmean = 0.0 if d == 0 else 2.0 * precision * recall / d
    return dict(precision=round(precision, 4), recall=round(recall, 4),
                hmean=round(hmean, 4), gt_num=gt_num, pred_num=pred_num, match_num=m)


def main():
    parser = argparse.ArgumentParser(description="ICDAR2015 DBNet eval with real GT")
    parser.add_argument("--ann_file", default="data/icdar2015/textdet_test.json",
                        help="Path to COCO-format ICDAR GT annotations")
    parser.add_argument("--normal_dir", default="data/icdar2015/textdet_imgs/test",
                        help="Normal-light test images directory")
    parser.add_argument("--lowlight_dir", default="lowlight_icdar_test",
                        help="Low-light test images directory")
    parser.add_argument("--output_dir", default="outputs/phase3_icdar",
                        help="Output directory for results")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit to first N images (0=all)")
    parser.add_argument("--no_save_vis", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ann_file = args.ann_file if os.path.isabs(args.ann_file) else os.path.join(script_dir, args.ann_file)
    normal_dir = args.normal_dir if os.path.isabs(args.normal_dir) else os.path.join(script_dir, args.normal_dir)
    lowlight_dir = args.lowlight_dir if os.path.isabs(args.lowlight_dir) else os.path.join(script_dir, args.lowlight_dir)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load GT
    print(f"Loading ICDAR GT from: {ann_file}")
    gt_list = load_icdar_gt(ann_file, normal_dir)
    print(f"  Found {len(gt_list)} images with GT annotations")

    if args.max_images > 0:
        gt_list = gt_list[:args.max_images]
        print(f"  Limited to {len(gt_list)} images")

    # Verify lowlight dir exists
    lowlight_exists = os.path.isdir(lowlight_dir)
    if not lowlight_exists:
        print(f"\nWARNING: Low-light dir not found: {lowlight_dir}")
        print("  Will only run normal-light inference and skip lowlight comparison.")
        print("  Generate lowlight images first with: python create_lowlight_dataset.py")
        print()

    # Load model
    print("\nLoading DBNet ...")
    ocr = MMOCRInferencer(det="DBNet", rec="CRNN")
    print("  Done.")

    print(f"\nRunning inference on {len(gt_list)} images ...")
    save_vis = not args.no_save_vis

    normal_results = []
    lowlight_results = []

    for idx, gt in enumerate(gt_list):
        fname = gt["file_name"]
        normal_path = gt["img_path"]
        lowlight_path = os.path.join(lowlight_dir, fname) if lowlight_exists else None

        print(f"\n  [{idx+1}/{len(gt_list)}] {fname}")

        # Normal light
        t0 = time.time()
        on_dir = os.path.join(output_dir, "normal") if save_vis else None
        if save_vis and on_dir:
            os.makedirs(on_dir, exist_ok=True)
            res_n = ocr(normal_path, out_dir=on_dir, return_vis=True)
        else:
            res_n = ocr(normal_path, return_vis=False)
        tn = time.time() - t0

        pred_n = res_n["predictions"][0]
        normal_polys = pred_n.get("det_polygons", [])
        det_n = compute_hmean(gt, normal_polys)
        normal_results.append(det_n)

        # Low light
        if lowlight_path and os.path.exists(lowlight_path):
            t0 = time.time()
            ol_dir = os.path.join(output_dir, "lowlight") if save_vis else None
            if save_vis and ol_dir:
                os.makedirs(ol_dir, exist_ok=True)
                res_l = ocr(lowlight_path, out_dir=ol_dir, return_vis=True)
            else:
                res_l = ocr(lowlight_path, return_vis=False)
            tl = time.time() - t0

            pred_l = res_l["predictions"][0]
            lowlight_polys = pred_l.get("det_polygons", [])
            det_l = compute_hmean(gt, lowlight_polys)
            lowlight_results.append(det_l)

            print(f"      GT={det_n['gt_num']} | "
                  f"normal  H={det_n['hmean']:.4f} P={det_n['precision']:.4f} R={det_n['recall']:.4f} "
                  f"({det_n['pred_num']} preds)")
            print(f"                       | "
                  f"lowlight H={det_l['hmean']:.4f} P={det_l['precision']:.4f} R={det_l['recall']:.4f} "
                  f"({det_l['pred_num']} preds)")
            print(f"      time: {tn:.2f}s | {tl:.2f}s")
        else:
            lowlight_results.append(None)
            print(f"      GT={det_n['gt_num']} | "
                  f"normal H={det_n['hmean']:.4f} P={det_n['precision']:.4f} R={det_n['recall']:.4f}")
            print(f"      time: {tn:.2f}s")
            print(f"      (lowlight image not found, skipped)")

    # Summary
    print("\n" + "=" * 65)
    print("ICDAR2015 Detection Evaluation (REAL GT)")
    print("=" * 65)

    def summarize(results, label):
        valid = [r for r in results if r is not None]
        if not valid:
            print(f"\n  {label}: No results.")
            return None
        avg = {k: round(np.mean([r[k] for r in valid]), 4)
               for k in ["precision", "recall", "hmean"]}
        tg = sum(r["gt_num"] for r in valid)
        tp = sum(r["pred_num"] for r in valid)
        tm = sum(r["match_num"] for r in valid)
        mp = tm / tp if tp else 0
        mr = tm / tg if tg else 0
        mh = (2 * mp * mr / (mp + mr)) if (mp + mr) else 0
        print(f"\n  --- {label} ---")
        print(f"  Per-image avg  P={avg['precision']:.4f}  R={avg['recall']:.4f}  H={avg['hmean']:.4f}")
        print(f"  Global         P={mp:.4f} ({tm}/{tp})  R={mr:.4f} ({tm}/{tg})  H={mh:.4f}")
        return dict(avg=avg, macro_p=mp, macro_r=mr, macro_h=mh, gt_total=tg, pred_total=tp, match_total=tm)

    sum_n = summarize(normal_results, "Normal Light")
    sum_l = summarize(lowlight_results, "Low Light")

    if sum_n and sum_l:
        delta_h = sum_n["avg"]["hmean"] - sum_l["avg"]["hmean"]
        print(f"\n  === Degradation ===")
        print(f"  H-mean drop:  {delta_h:.4f}  ({(delta_h/sum_n['avg']['hmean']*100):.1f}% relative)")
        print(f"  Precision drop: {sum_n['avg']['precision'] - sum_l['avg']['precision']:.4f}")
        print(f"  Recall drop:    {sum_n['avg']['recall'] - sum_l['avg']['recall']:.4f}")

    # Save JSON
    result_file = os.path.join(output_dir, "icdar_metrics.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(dict(
            N_images=len(gt_list),
            normal=dict(
                avg_det={k: np.mean([r[k] for r in normal_results]) for k in ["precision","recall","hmean"]},
                macro=dict(
                    prec=sum_n["macro_p"], rec=sum_n["macro_r"], hmean=sum_n["macro_h"],
                    gt=sum_n["gt_total"], pred=sum_n["pred_total"], match=sum_n["match_total"])
            ) if sum_n else None,
            lowlight=dict(
                avg_det={k: np.mean([r[k] for r in lowlight_results if r is not None]) for k in ["precision","recall","hmean"]},
                macro=dict(
                    prec=sum_l["macro_p"], rec=sum_l["macro_r"], hmean=sum_l["macro_h"],
                    gt=sum_l["gt_total"], pred=sum_l["pred_total"], match=sum_l["match_total"])
            ) if sum_l else None,
        ), f, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to: {result_file}")


if __name__ == "__main__":
    main()
