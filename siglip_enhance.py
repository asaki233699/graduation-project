#!/usr/bin/env python3
# SigLIP 语义引导低光照增强 (兼容老版 SigLIP)
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
import cv2, os, sys

PROJECT_DIR = r'/mnt/c/Users/yanhao.ding/Desktop/graduation-project'
LOWLIGHT_DIR = os.path.join(PROJECT_DIR, 'lowlight_dataset')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'siglip_enhanced')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('=' * 60)
print('SigLIP 语义引导增强')

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
print(f'\n使用 prompt: \"{PROMPT}\"')

ti = processor(text=[PROMPT], padding='max_length', truncation=True, max_length=64, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    text_embed = model.get_text_features(**ti).pooler_output
    text_embed = F.normalize(text_embed, p=2, dim=-1)
print(f'文本编码完成: {text_embed.shape}')

def compute_similarity_map(img_path, grid=20):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    pw, ph = w // grid, h // grid
    sim_map = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch = img.crop((j*pw, i*ph, (j+1)*pw, (i+1)*ph))
            inp = processor(images=patch, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                feat = model.get_image_features(**inp).pooler_output
                feat = F.normalize(feat, p=2, dim=-1)
                sim_map[i, j] = (feat @ text_embed.T).item()
    zoom_h = h / grid
    zoom_w = w / grid
    sim_full = zoom(sim_map, (zoom_h, zoom_w), order=1)
    return sim_full, img

def enhance_image(img_path, sim_map, strength=1.5):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    s_min, s_max = sim_map.min(), sim_map.max()
    if s_max - s_min < 1e-6:
        return bgr
    sim_norm = (sim_map - s_min) / (s_max - s_min)
    if sim_norm.shape[:2] != (h, w):
        sim_norm = zoom(sim_norm, (h/sim_norm.shape[0], w/sim_norm.shape[1]), order=1)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_ch, l_ch, s_ch = cv2.split(hls)
    boost = 1.0 + strength * sim_norm
    l_ch = l_ch * boost
    l_ch = np.clip(l_ch, 0, 255)
    hls_enhanced = cv2.merge([h_ch, l_ch, s_ch])
    return cv2.cvtColor(hls_enhanced.astype(np.uint8), cv2.COLOR_HLS2BGR)

def save_heatmap_viz(sim_map, orig_path, out_path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    orig = cv2.imread(orig_path)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w = orig.shape[:2]
    if sim_map.shape[:2] != (h, w):
        sim_resized = zoom(sim_map, (h/sim_map.shape[0], w/sim_map.shape[1]), order=1)
    else:
        sim_resized = sim_map
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(orig_rgb); ax1.set_title('原始低光照图', fontsize=12); ax1.axis('off')
    ax2.imshow(sim_resized, cmap='hot'); ax2.set_title('SigLIP 文字区域热力图', fontsize=12); ax2.axis('off')
    ax3.imshow(orig_rgb); ax3.imshow(sim_resized, cmap='jet', alpha=0.4); ax3.set_title('叠加效果', fontsize=12); ax3.axis('off')
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()

exts = ('.jpg', '.png', '.jpeg', '.bmp')
image_files = sorted(f for f in os.listdir(LOWLIGHT_DIR) if f.lower().endswith(exts))
if not image_files:
    print(f'在 {LOWLIGHT_DIR} 中找不到图片!')
    sys.exit(1)

print(f'\n找到 {len(image_files)} 张低光照图片\n')
for fname in image_files:
    img_path = os.path.join(LOWLIGHT_DIR, fname)
    print(f'处理: {fname}')
    print('  计算 SigLIP 文字区域热力图...')
    sim_map, _ = compute_similarity_map(img_path, grid=20)
    print('  生成增强图片...')
    enhanced = enhance_image(img_path, sim_map, strength=1.5)
    if enhanced is None:
        print(f'  跳过: 无法读取 {fname}')
        continue
    out_img_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_img_path, enhanced)
    print(f'  增强图已保存: {out_img_path}')
    name_no_ext = os.path.splitext(fname)[0]
    viz_path = os.path.join(OUTPUT_DIR, f'{name_no_ext}_heatmap.png')
    save_heatmap_viz(sim_map, img_path, viz_path)
    print(f'  热力图已保存: {viz_path}')
    s_min, s_max, s_mean = sim_map.min(), sim_map.max(), sim_map.mean()
    print(f'  相似度: min={s_min:.4f}, max={s_max:.4f}, mean={s_mean:.4f}')
    print()

print('=' * 60)
print('完成!')
print(f'增强图片: {OUTPUT_DIR}/')
print(f'热力图:   {OUTPUT_DIR}/*_heatmap.png')
print(f'\n下一步: conda activate mmcv_stable && python p2_compare.py')
print('=' * 60)
