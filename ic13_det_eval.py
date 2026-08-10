#!/usr/bin/env python3
# ============================================================
# ICDAR2013 检测评估: DBNet 正常 vs 低光照 vs SigLIP增强
# 233 张测试图, 真标注, 标准 P/R/H-mean
# ============================================================

from mmocr.apis import MMOCRInferencer
import json, os, time, cv2, numpy as np, random

PROJECT_DIR = r'/mnt/c/Users/yanhao.ding/Desktop/graduation-project'
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'icdar2013')
LOWLIGHT_DIR = os.path.join(DATA_DIR, 'textdet_imgs_test_lowlight')
ENHANCED_DIR = os.path.join(PROJECT_DIR, 'outputs', 'ic13_siglip_enhanced')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'ic13_det_results')
for d in [LOWLIGHT_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

print('='*70)
print('ICDAR2013 检测评估: DBNet 正常 vs 低光照 vs SigLIP增强')
print('='*70)

# ====== Step 1: 读标注 ======
ann_path = os.path.join(DATA_DIR, 'textdet_test.json')
with open(ann_path, 'r', encoding='utf-8') as f:
    ann_data = json.load(f)
data_list = ann_data['data_list']
print(f'测试集: {len(data_list)} 张 (带标注)')

gt_bboxes = {}   # {img_name: [(x1,y1,x2,y2), ...]}
gt_ignores = {}  # {img_name: [bool, ...]}
for item in data_list:
    fname = os.path.basename(item['img_path'])
    bboxes = []
    ignores = []
    for inst in item['instances']:
        bboxes.append(inst['bbox'])   # [x1,y1,x2,y2]
        ignores.append(inst['ignore'])
    gt_bboxes[fname] = bboxes
    gt_ignores[fname] = ignores

test_dir = os.path.join(DATA_DIR, 'textdet_imgs', 'test')
all_images = sorted(gt_bboxes.keys())
print(f'标注图片: {len(all_images)}')

# ====== Step 2: 生成低光照版本 ======
print(f'\n生成低光照测试集 -> {LOWLIGHT_DIR}')
low_count = 0
for fname in all_images:
    dst = os.path.join(LOWLIGHT_DIR, fname)
    if os.path.exists(dst):
        low_count += 1; continue
    img = cv2.imread(os.path.join(test_dir, fname))
    if img is None: continue
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h, l, s = cv2.split(hls)
    brightness = random.uniform(0.25, 0.55)
    noise_sigma = random.uniform(10, 25)
    l = np.clip(l * brightness, 0, 255)
    gauss = np.random.normal(0, noise_sigma, l.shape).astype(np.float32)
    l = np.clip(l + gauss, 0, 255)
    hls = cv2.merge([h, l, s])
    cv2.imwrite(dst, cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR))
    low_count += 1
print(f'低光照生成: {low_count} 张')

# 检查增强图是否已存在
has_enhanced = os.path.exists(ENHANCED_DIR) and \
    len(os.listdir(ENHANCED_DIR)) >= len(all_images)
if not has_enhanced:
    print(f'\n⚠ 增强图目录不存在或不全: {ENHANCED_DIR}')
    print(f'请在 siglip 环境跑 siglip_enhance.py (改路径指向 IC13 低光照目录)')
    print(f'或跳过增强组, 先跑正常 vs 低光照')

# ====== Step 3: 评估函数 ======
def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def poly_to_bbox(poly):
    xs = poly[0::2]; ys = poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

def eval_detection(ocr, img_dir_path, gt_bboxes, gt_ignores, condition, all_images):
    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt = sum(len(b) for b in gt_bboxes.values())
    
    print(f'\n  [DBNet] {condition}: {len(all_images)} 张...')
    t0 = time.time()
    
    for idx, fname in enumerate(all_images):
        img_path = os.path.join(img_dir_path, fname)
        if not os.path.exists(img_path):
            total_fn += len(gt_bboxes[fname])
            continue
        
        try:
            result = ocr(img_path, return_vis=False)
            pred_polys = result['predictions'][0]['det_polygons']
        except Exception:
            pred_polys = []
        
        gt_b = gt_bboxes[fname]
        ig = gt_ignores[fname]
        pr_b = [poly_to_bbox(p) for p in pred_polys]
        
        if len(gt_b) == 0 and len(pr_b) == 0:
            continue
        if len(gt_b) == 0:
            total_fp += len(pr_b)
            continue
        if len(pr_b) == 0:
            total_fn += len(gt_b) - sum(ig)
            continue
        
        ious = np.zeros((len(pr_b), len(gt_b)))
        for i, pb in enumerate(pr_b):
            for j, gb in enumerate(gt_b):
                ious[i,j] = bbox_iou(pb, gb)
        
        matches = [(ious[i,j], i, j) for i in range(len(pr_b))
                   for j in range(len(gt_b)) if ious[i,j] >= 0.5]
        matches.sort(reverse=True)
        
        matched_pr, matched_gt = set(), set()
        for _, i, j in matches:
            if i not in matched_pr and j not in matched_gt:
                matched_pr.add(i); matched_gt.add(j)
        
        tp = len(matched_pr)
        fp = len(pr_b) - tp
        fn = len(gt_b) - tp - sum(ig[j] for j in range(len(gt_b)) if j not in matched_gt)
        
        total_tp += tp; total_fp += fp; total_fn += fn
        
        if (idx+1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'    [{idx+1}/{len(all_images)}] {elapsed:.0f}s')
    
    elapsed = time.time() - t0
    p = total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0.0
    r = total_tp/(total_tp+total_fn) if (total_tp+total_fn) > 0 else 0.0
    h = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    print(f'    Done: P={p:.4f} R={r:.4f} H={h:.4f} ({total_tp}TP/{total_fp}FP/{total_fn}FN), {elapsed:.0f}s')
    return p, r, h

# ====== Step 4: 跑评估 ======
print(f'\n{"="*70}')
print('开始检测评估')

ocr = MMOCRInferencer(det='DBNet', rec='SVTR-small')

# 正常
p_n, r_n, h_n = eval_detection(ocr, test_dir, gt_bboxes, gt_ignores, '正常', all_images)
# 低光照
p_l, r_l, h_l = eval_detection(ocr, LOWLIGHT_DIR, gt_bboxes, gt_ignores, '低光照', all_images)
# 增强 (如果有)
if has_enhanced:
    p_e, r_e, h_e = eval_detection(ocr, ENHANCED_DIR, gt_bboxes, gt_ignores, 'SigLIP增强', all_images)
else:
    p_e, r_e, h_e = None, None, None

# ====== Step 5: 总结 ======
print(f'\n{"="*70}')
print('ICDAR2013 检测评估 — 总结')
print('='*70)
print(f'测试集: {len(all_images)} 张, 真标注')
print(f'')
print(f'  {"Condition":<15} {"Precision":>12} {"Recall":>12} {"H-mean":>12}')
print(f'  {"-"*53}')
print(f'  {"正常":<15} {p_n:>12.4f} {r_n:>12.4f} {h_n:>12.4f}')
print(f'  {"低光照":<15} {p_l:>12.4f} {r_l:>12.4f} {h_l:>12.4f}')
if has_enhanced:
    print(f'  {"SigLIP增强":<15} {p_e:>12.4f} {r_e:>12.4f} {h_e:>12.4f}')
    print(f'\n  低光照 H={h_l:.4f} -> 增强 H={h_e:.4f} ({"恢复" if h_e>h_l else "下降"} {h_e-h_l:+.4f})')

print(f'\n  论文可用:')
print(f'  正常 H-mean: {h_n:.4f}')
print(f'  低光照 H-mean: {h_l:.4f} (下降 {h_n-h_l:+.4f})')

if not has_enhanced:
    print(f'\n  下一步: 在 siglip 环境跑 siglip_enhance.py 生成增强图, 再重新跑本脚本')
    print(f'  低光照目录: {LOWLIGHT_DIR}')
    print(f'  增强输出目录: {ENHANCED_DIR}')
print(f'='*70)
