#!/usr/bin/env python3
# ============================================================
# IIIT5K SigLIP 增强 + 识别对比 (siglip=增强, mmcv=识别, 分两步)
# 第一步 (siglip 环境): 生成增强图
# 第二步 (mmcv_stable): 跑 SVTR 增强组对比
# ============================================================

import torch, torch.nn.functional as F, numpy as np
from PIL import Image
import cv2, os, sys

PROJECT_DIR = r'C:/Users/yanhao.ding/Desktop/graduation-project'
LOWLIGHT_DIR = os.path.join(PROJECT_DIR, 'data', 'iiit5k', 'textrecog_imgs_test_lowlight')
OUTPUT_DIR = os.path.join(LOWLIGHT_DIR + '_enhanced')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*60)
print('IIIT5K SigLIP 增强生成')
print('='*60)

from transformers import AutoProcessor, AutoModel
from scipy.ndimage import zoom

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

MODEL_NAME = 'google/siglip-base-patch16-224'
print(f'加载: {MODEL_NAME}')
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()
print('加载成功!')

PROMPT = '一段清晰可读的文字，边缘锐利笔画分明'
print(f'prompt: {PROMPT}')

ti = processor(text=[PROMPT], padding='max_length', truncation=True, max_length=64, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    text_embed = model.get_text_features(**ti).pooler_output
    text_embed = F.normalize(text_embed, p=2, dim=-1)

def compute_similarity_map(img_path, grid=20):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    pw, ph = w//grid, h//grid
    sim_map = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch = img.crop((j*pw, i*ph, (j+1)*pw, (i+1)*ph))
            inp = processor(images=patch, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                feat = model.get_image_features(**inp).pooler_output
                feat = F.normalize(feat, p=2, dim=-1)
                sim_map[i,j] = (feat @ text_embed.T).item()
    zh, zw = h/grid, w/grid
    return zoom(sim_map, (zh, zw), order=1)

def enhance(img_path, sim_map, strength=1.5):
    bgr = cv2.imread(img_path)
    if bgr is None: return None
    h, w = bgr.shape[:2]
    s_min, s_max = sim_map.min(), sim_map.max()
    if s_max - s_min < 1e-6: return bgr
    sim_norm = (sim_map - s_min) / (s_max - s_min)
    if sim_norm.shape[:2] != (h, w):
        sim_norm = zoom(sim_norm, (h/sim_norm.shape[0], w/sim_norm.shape[1]), order=1)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_ch, l_ch, s_ch = cv2.split(hls)
    l_ch = l_ch * (1.0 + strength * sim_norm)
    l_ch = np.clip(l_ch, 0, 255)
    return cv2.cvtColor(cv2.merge([h_ch, l_ch, s_ch]).astype(np.uint8), cv2.COLOR_HLS2BGR)

imgs = sorted(f for f in os.listdir(LOWLIGHT_DIR) if f.lower().endswith(('.jpg','.png','.jpeg','.bmp')))
if not imgs:
    print(f'在 {LOWLIGHT_DIR} 找不到图片!')
    print('请先在 mmcv_stable 环境跑 iiit5k_eval.py 生成低光照图')
    sys.exit(1)
print(f'\n找到 {len(imgs)} 张, 开始增强...\n')
for idx, fname in enumerate(imgs):
    img_path = os.path.join(LOWLIGHT_DIR, fname)
    sim_map = compute_similarity_map(img_path, grid=20)
    enhanced = enhance(img_path, sim_map, strength=1.5)
    if enhanced is None: continue
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), enhanced)
    if (idx+1) % 200 == 0:
        print(f'  [{idx+1}/{len(imgs)}]')

print(f'\n完成! 增强图: {OUTPUT_DIR}')
print(f'切回 mmcv_stable 跑 iiit5k_eval.py (v2), 会自动检测增强目录')
