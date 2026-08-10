#!/usr/bin/env python3
# ============================================================
# IIIT5K 识别评估: CRNN vs SVTR, 正常 vs 低光照
# 3000 张测试图, 只做识别不做检测 (图已经是裁剪好的单词)
# ============================================================

from mmocr.apis import MMOCRInferencer
import json, os, time, cv2, numpy as np, random

PROJECT_DIR = r'/mnt/c/Users/yanhao.ding/Desktop/graduation-project'
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'iiit5k')
LOWLIGHT_DIR = os.path.join(DATA_DIR, 'textrecog_imgs_test_lowlight')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'iiit5k_results')
os.makedirs(LOWLIGHT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Step 1: 读标注 ======
print('=' * 70)
print('IIIT5K 识别评估: CRNN vs SVTR (正常 vs 低光照)')
print('=' * 70)

ann_path = os.path.join(DATA_DIR, 'textrecog_test.json')
with open(ann_path, 'r', encoding='utf-8') as f:
    ann_data = json.load(f)

data_list = ann_data['data_list']
print(f'测试集共 {len(data_list)} 张图片')

# 构建 {文件名: 标注文本} 字典
gt_map = {}
for item in data_list:
    img_path = item['img_path']  # e.g. textrecog_imgs/test/1001_1.png
    fname = os.path.basename(img_path)
    gt_map[fname] = item['instances'][0]['text']

img_dir = os.path.join(DATA_DIR, 'textrecog_imgs', 'test')
all_images = sorted(gt_map.keys())
print(f'标注条目: {len(all_images)}')

# ====== Step 2: 生成低光照版本 ======
print(f'\n生成低光照测试集 -> {LOWLIGHT_DIR}')
lowlight_count = 0
for fname in all_images:
    src = os.path.join(img_dir, fname)
    dst = os.path.join(LOWLIGHT_DIR, fname)
    if os.path.exists(dst):
        lowlight_count += 1
        continue
    img = cv2.imread(src)
    if img is None:
        continue
    # 和 create_lowlight_dataset.py 一样的退化逻辑
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h, l, s = cv2.split(hls)
    brightness = random.uniform(0.25, 0.55)
    noise_sigma = random.uniform(10, 25)
    l = np.clip(l * brightness, 0, 255)
    gauss = np.random.normal(0, noise_sigma, l.shape).astype(np.float32)
    l = np.clip(l + gauss, 0, 255)
    hls = cv2.merge([h, l, s])
    low = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
    cv2.imwrite(dst, low)
    lowlight_count += 1
print(f'生成完成: {lowlight_count} 张')

# ====== Step 3: 评估函数 ======
def evaluate(ocr, img_dir_path, gt, model_name, condition, limit=None):
    imgs = sorted(gt.keys())
    if limit:
        imgs = imgs[:limit]
    total = len(imgs)
    correct = 0
    results = []
    
    print(f'\n  [{model_name}] {condition}: {total} 张...')
    t0 = time.time()
    
    for idx, fname in enumerate(imgs):
        img_path = os.path.join(img_dir_path, fname)
        if not os.path.exists(img_path):
            results.append((fname, gt[fname], '', False))
            continue
        
        try:
            result = ocr(img_path, return_vis=False)
            pred_texts = result['predictions'][0]['rec_texts']
            pred = pred_texts[0] if pred_texts else ''
        except Exception:
            pred = ''
        
        # 不区分大小写比较
        is_correct = (pred.strip().lower() == gt[fname].strip().lower())
        if is_correct:
            correct += 1
        results.append((fname, gt[fname], pred, is_correct))
        
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            acc = correct / (idx + 1)
            eta = (total - idx - 1) / speed
            print(f'    [{idx+1}/{total}] acc={acc:.3f} speed={speed:.1f} img/s ETA={eta:.0f}s')
    
    elapsed = time.time() - t0
    acc = correct / total
    print(f'    Done: acc={acc:.4f} ({correct}/{total}), {elapsed:.0f}s')
    return acc, results

# ====== Step 4: 跑四组评估 ======
print('\n' + '=' * 70)
print('开始评估 (这需要一段时间, 请耐心等待)')

# CRNN 正常
ocr_crnn = MMOCRInferencer(rec='CRNN')
acc_crnn_n, res_crnn_n = evaluate(ocr_crnn, img_dir, gt_map, 'CRNN', '正常')
# CRNN 低光照
acc_crnn_l, res_crnn_l = evaluate(ocr_crnn, LOWLIGHT_DIR, gt_map, 'CRNN', '低光照')

# SVTR 正常
ocr_svtr = MMOCRInferencer(rec='SVTR-small')
acc_svtr_n, res_svtr_n = evaluate(ocr_svtr, img_dir, gt_map, 'SVTR', '正常')
# SVTR 低光照
acc_svtr_l, res_svtr_l = evaluate(ocr_svtr, LOWLIGHT_DIR, gt_map, 'SVTR', '低光照')

# ====== Step 5: 总结 ======
print('\n' + '=' * 70)
print('IIIT5K 识别评估 — 总结')
print('=' * 70)
print(f'测试集: {len(all_images)} 张')
print('')
print(f'  {"Model":<10} {"正常":>10} {"低光照":>10} {"下降":>10}')
print('  ' + '-' * 42)
print(f'  {"CRNN":<10} {acc_crnn_n:>10.4f} {acc_crnn_l:>10.4f} {acc_crnn_n-acc_crnn_l:>+10.4f}')
print(f'  {"SVTR":<10} {acc_svtr_n:>10.4f} {acc_svtr_l:>10.4f} {acc_svtr_n-acc_svtr_l:>+10.4f}')

# 论文可用数据
print('\n  论文可用:')
print(f'  低光照导致 CRNN 识别率从 {acc_crnn_n:.4f} 降至 {acc_crnn_l:.4f} (下降 {acc_crnn_n-acc_crnn_l:.4f})')
print(f'  低光照导致 SVTR 识别率从 {acc_svtr_n:.4f} 降至 {acc_svtr_l:.4f} (下降 {acc_svtr_n-acc_svtr_l:.4f})')
if (acc_svtr_n-acc_svtr_l) < (acc_crnn_n-acc_crnn_l):
    print('  ✓ SVTR 比 CRNN 更扛低光照 (下降幅度更小): ViT 结构天然优势')
else:
    print('  注意: SVTR 在本数据集上不比 CRNN 更鲁棒')
print('=' * 70)