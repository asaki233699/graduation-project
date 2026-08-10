#!/usr/bin/env python3
# ============================================================
# ICDAR2013 低光照增强生成 (siglip 环境跑)
# 改自 siglip_enhance.py, 指向 IC13 低光照目录
# ============================================================

import torch, torch.nn.functional as F, numpy as np
from PIL import Image
import cv2, os, sys

PROJECT_DIR = r'/mnt/c/Users/yanhao.ding/Desktop/graduation-project'
LOWLIGHT_DIR = os.path.join(PROJECT_DIR, 'data', 'icdar2013', 'textdet_imgs_test_lowlight')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'ic13_siglip_enhanced')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('='*60)
print('IC13 SigLIP 增强生成')

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
    sim_full = zoom(sim_map, (zh, zw), order=1)
    return sim_full

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
    sys.exit(1)
print(f'\n找到 {len(imgs)} 张, 开始增强...\n')
for idx, fname in enumerate(imgs):
    img_path = os.path.join(LOWLIGHT_DIR, fname)
    sim_map = compute_similarity_map(img_path, grid=20)
    enhanced = enhance(img_path, sim_map, strength=1.5)
    if enhanced is None: continue
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), enhanced)
    if (idx+1) % 50 == 0:
        print(f'  [{idx+1}/{len(imgs)}]')

print(f'\n完成! 输出: {OUTPUT_DIR}')
print(f'然后切回 mmcv_stable 重跑 ic13_det_eval.py')
