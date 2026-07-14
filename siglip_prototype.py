#!/usr/bin/env python3
# SigLIP 2 热力图原型 (兼容老版 SigLIP)
import torch, torch.nn.functional as F, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

PROJECT_DIR = r'/mnt/c/Users/yanhao.ding/Desktop/graduation-project'
NORMAL_DIR = os.path.join(PROJECT_DIR, 'demo')
LOWLIGHT_DIR = os.path.join(PROJECT_DIR, 'lowlight_dataset')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'siglip_heatmaps')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 60)
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

from transformers import AutoProcessor, AutoModel
from scipy.ndimage import zoom

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

MODEL_NAME = 'google/siglip-base-patch16-224'
print(f'\n加载: {MODEL_NAME}')
print('(首次运行下载 ~500MB)')
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.eval()
print('加载成功!')

PROMPTS = [
    '一段清晰可读的文字，边缘锐利',
    '一段模糊难以辨认的文字',
    '暗光低光照条件下的文字',
    '明亮充足光线的文字',
]
N = len(PROMPTS)

print('\n编码 prompts...')
ti = processor(text=PROMPTS, padding='max_length', truncation=True, max_length=64, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    te = model.get_text_features(**ti).pooler_output
    te = F.normalize(te, p=2, dim=-1)
print(f'完成: {te.shape}')

def heatmap(img_path, prompt_embeds, grid=14):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    pw, ph = w // grid, h // grid
    hm = np.zeros((grid, grid, N), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch = img.crop((j*pw, i*ph, (j+1)*pw, (i+1)*ph))
            inp = processor(images=patch, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                f = model.get_image_features(**inp).pooler_output
                f = F.normalize(f, p=2, dim=-1)
                hm[i,j,:] = (f @ prompt_embeds.T).cpu().numpy()[0]
    return hm, img

def draw_hm(ax, orig, hm2d, title):
    w, h = orig.size
    zh = h / hm2d.shape[0]
    zw = w / hm2d.shape[1]
    big = zoom(hm2d, (zh, zw), order=1)
    ax.imshow(orig)
    ax.imshow(big, cmap='jet', alpha=0.4, vmin=0.15, vmax=0.45)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def process(name):
    npth = os.path.join(NORMAL_DIR, name)
    lpth = os.path.join(LOWLIGHT_DIR, name)
    if not os.path.exists(npth) or not os.path.exists(lpth):
        return
    print(f'\n处理: {name}')
    print('  正常光照...')
    hn, in_ = heatmap(npth, te)
    print('  低光照...')
    hl, il = heatmap(lpth, te)
    nc = 1 + N
    fig, axes = plt.subplots(3, nc, figsize=(nc*4, 12))
    axes[0,0].imshow(in_); axes[0,0].set_title('正常', fontsize=12); axes[0,0].axis('off')
    for k in range(N):
        draw_hm(axes[0,k+1], in_, hn[:,:,k], f'正常+{PROMPTS[k][:14]}')
    axes[1,0].imshow(il); axes[1,0].set_title('低光照', fontsize=12); axes[1,0].axis('off')
    for k in range(N):
        draw_hm(axes[1,k+1], il, hl[:,:,k], f'低光照+{PROMPTS[k][:14]}')
    axes[2,0].axis('off')
    axes[2,0].text(0.5,0.5,'差异\n(正常-低光照)', ha='center',va='center',fontsize=11, transform=axes[2,0].transAxes)
    for k in range(N):
        d = hn[:,:,k] - hl[:,:,k]
        v = max(abs(d.min()), abs(d.max()))
        w,h = in_.size
        db = zoom(d, (h/d.shape[0], w/d.shape[1]), order=1)
        axes[2,k+1].imshow(in_, alpha=0.5)
        axes[2,k+1].imshow(db, cmap='RdBu_r', alpha=0.5, vmin=-v, vmax=v)
        axes[2,k+1].set_title(f'差异:{PROMPTS[k][:12]}', fontsize=10)
        axes[2,k+1].axis('off')
    plt.tight_layout(pad=1.0)
    out = os.path.join(OUTPUT_DIR, f'{os.path.splitext(name)[0]}_siglip.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  已保存: {out}')
    an = hn.mean(axis=(0,1))
    al = hl.mean(axis=(0,1))
    print(f'\n  {\"Prompt\":<42} {\"正常\":>6} {\"低光照\":>6} {\"变化\":>7}')
    print(f'  {\"-\"*62}')
    for k in range(N):
        print(f'  {PROMPTS[k]:<42} {an[k]:>6.4f} {al[k]:>6.4f} {an[k]-al[k]:>+7.4f}')

exts = ('.jpg','.png','.jpeg','.bmp')
ns = set(f for f in os.listdir(NORMAL_DIR) if f.lower().endswith(exts))
ls = set(f for f in os.listdir(LOWLIGHT_DIR) if f.lower().endswith(exts))
common = sorted(ns & ls)
if not common:
    print('找不到共同图片!')
    sys.exit(1)
print(f'\n找到 {len(common)} 对图片\n')
for c in common:
    process(c)
print('\n' + '=' * 60)
print('完成! 结果在:', OUTPUT_DIR)
print('=' * 60)
