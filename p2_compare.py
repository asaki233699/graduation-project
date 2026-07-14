#!/usr/bin/env python3
# Phase 2: 正常 vs 低光照 vs SigLIP增强 对比
from mmocr.apis import MMOCRInferencer
import os, time

PROJECT_DIR = r'/mnt/c/Users/yanhao.ding/Desktop/graduation-project'

NORMAL_DIR = os.path.join(PROJECT_DIR, 'demo')
LOWLIGHT_DIR = os.path.join(PROJECT_DIR, 'lowlight_dataset')
ENHANCED_DIR = os.path.join(PROJECT_DIR, 'outputs', 'siglip_enhanced')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs', 'phase2_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)

for sub in ['normal', 'lowlight', 'enhanced']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

SEP70 = '=' * 70
SEP60 = '-' * 70

print(SEP70)
print('Phase 2: 正常 vs 低光照 vs SigLIP增强 对比测试')
print(SEP70)
print(f'正常图:   {NORMAL_DIR}')
print(f'低光照:   {LOWLIGHT_DIR}')
print(f'SigLIP增强: {ENHANCED_DIR}')
print(SEP70)

print('加载模型 DBNet + CRNN ...')
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
print('加载完成!\n')

exts = ('.jpg', '.png', '.jpeg', '.bmp')
n_set = set(f for f in os.listdir(NORMAL_DIR) if f.lower().endswith(exts))
l_set = set(f for f in os.listdir(LOWLIGHT_DIR) if f.lower().endswith(exts))
e_set = set(f for f in os.listdir(ENHANCED_DIR) if f.lower().endswith(exts))
common = sorted(n_set & l_set & e_set)

if not common:
    print('找不到三组共有的图片!')
    print(f'正常:   {sorted(n_set)}')
    print(f'低光照: {sorted(l_set)}')
    print(f'增强:   {sorted(e_set)}')
    exit(1)

print(f'找到 {len(common)} 张共有图片\n')

stats = {'normal': [], 'lowlight': [], 'enhanced': []}

for img_name in common:
    print(SEP70)
    print(f'图片: {img_name}')
    print(SEP70)
    
    paths = {
        'normal': os.path.join(NORMAL_DIR, img_name),
        'lowlight': os.path.join(LOWLIGHT_DIR, img_name),
        'enhanced': os.path.join(ENHANCED_DIR, img_name),
    }
    
    for condition in ['normal', 'lowlight', 'enhanced']:
        out_sub = os.path.join(OUTPUT_DIR, condition)
        t0 = time.time()
        result = ocr(paths[condition], out_dir=out_sub, return_vis=True)
        elapsed = time.time() - t0
        pred = result['predictions'][0]
        n_det = len(pred['det_polygons'])
        n_rec = len(pred['rec_texts'])
        stats[condition].append({'name': img_name, 'detect': n_det, 'rec': n_rec, 'time': elapsed})
        label = {'normal': '正常', 'lowlight': '低光照', 'enhanced': '增强'}[condition]
        print(f'  {label}: 检测 {n_det} 个, 识别 {n_rec} 个, 耗时 {elapsed:.1f}s')
    
    n_base = stats['lowlight'][-1]['detect']
    n_enh = stats['enhanced'][-1]['detect']
    gain = n_enh - n_base
    if n_base > 0:
        pct = gain / n_base * 100
        print(f'  -> 增强后检测数: {n_base} -> {n_enh} (变化 {gain:+d}, {pct:+.0f}%)')
    print()

print('\n' + SEP70)
print('Phase 2 对比测试 -- 总结')
print(SEP70)
print(f'{"图片":<25} {"正常检测":<10} {"低光照检测":<12} {"增强检测":<12}')
print(f'{"":25} {"正常识别":<10} {"低光照识别":<12} {"增强识别":<12}')
print(SEP60)

sum_det_n, sum_det_l, sum_det_e = 0, 0, 0
sum_rec_n, sum_rec_l, sum_rec_e = 0, 0, 0

for ns, ls, es in zip(stats['normal'], stats['lowlight'], stats['enhanced']):
    print(f'{ns["name"]:<25} {ns["detect"]:<10} {ls["detect"]:<12} {es["detect"]:<12}')
    print(f'{"":25} {ns["rec"]:<10} {ls["rec"]:<12} {es["rec"]:<12}')
    sum_det_n += ns['detect']; sum_det_l += ls['detect']; sum_det_e += es['detect']
    sum_rec_n += ns['rec'];   sum_rec_l += ls['rec'];   sum_rec_e += es['rec']

n = len(common)
print(SEP60)
print(f'{"平均":<25} {sum_det_n/n:<10.1f} {sum_det_l/n:<12.1f} {sum_det_e/n:<12.1f}')
print(f'{"":25} {sum_rec_n/n:<10.1f} {sum_rec_l/n:<12.1f} {sum_rec_e/n:<12.1f}')
print()

det_drop = sum_det_n/n - sum_det_l/n
det_recover = sum_det_e/n - sum_det_l/n
print(f'低光照导致检测下降: {det_drop:+.1f} 个/图')
print(f'SigLIP 增强恢复:     {det_recover:+.1f} 个/图')
if det_drop > 0:
    print(f'恢复率: {det_recover/det_drop*100:.0f}%')

print(f'\n低光照导致识别下降: {sum_rec_n/n - sum_rec_l/n:+.1f} 个/图')
print(f'SigLIP 增强恢复:     {sum_rec_e/n - sum_rec_l/n:+.1f} 个/图')

print('\n' + SEP70)
print('完成!')
print(f'结果: {OUTPUT_DIR}/')
print(SEP70)
