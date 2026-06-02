# mmocr/p1.py
"""
毕设 Phase 1: DBNet + CRNN Baseline 对比测试
支持自动对比正常图片与低照度图片的检测结果
"""

from mmocr.apis import MMOCRInferencer
import os
import time

# ==========================================
# 配置区域 (请根据你的实际路径修改)
# ==========================================
# 原始/正常图片目录
normal_dir = 'C:/Users/dingy/PycharmProjects/bishe/mmocr/demo'

# 生成好的低照度图片目录
lowlight_dir = 'C:/Users/dingy/PycharmProjects/bishe/mmocr/lowlight_dataset'

# 输出目录
output_dir = 'outputs/phase1_comparison'
os.makedirs(output_dir, exist_ok=True)

# 创建子目录用于保存结果
output_normal = os.path.join(output_dir, 'normal')
output_lowlight = os.path.join(output_dir, 'lowlight')
os.makedirs(output_normal, exist_ok=True)
os.makedirs(output_lowlight, exist_ok=True)

print("=" * 70)
print("毕设 Phase 1: DBNet + CRNN 性能对比测试")
print("=" * 70)
print(f"正常图片目录: {normal_dir}")
print(f"低照度图片目录: {lowlight_dir}")
print(f"结果保存位置: {output_dir}")
print("=" * 70)

# 初始化 OCR 推理器 (DBNet + CRNN)
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
print("\n✓ 模型加载完成：DBNet (检测) + CRNN (识别)")
print("=" * 70)

# 统计数据收集
stats = {
    'normal': [],
    'lowlight': []
}

# 遍历图片文件
all_images = set(os.listdir(normal_dir)).intersection(set(os.listdir(lowlight_dir)))
all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not all_images:
    print("⚠️ 警告：未找到可对比的图片！请检查目录路径。")
else:
    for img_name in all_images:
        print(f"\n{'='*70}")
        print(f"正在测试: {img_name}")
        print(f"{'='*70}")

        # 1. 处理正常图片
        normal_path = os.path.join(normal_dir, img_name)
        lowlight_path = os.path.join(lowlight_dir, img_name)

        # --- 处理正常图片逻辑 ---
        start_time = time.time()
        result_normal = ocr(normal_path, out_dir=output_normal, return_vis=True)
        time_normal = time.time() - start_time

        pred_normal = result_normal['predictions'][0]
        count_normal = len(pred_normal['det_polygons'])
        rec_texts_normal = pred_normal['rec_texts']

        # --- 处理低照度图片逻辑 ---
        start_time = time.time()
        result_lowlight = ocr(lowlight_path, out_dir=output_lowlight, return_vis=True)
        time_lowlight = time.time() - start_time

        pred_lowlight = result_lowlight['predictions'][0]
        count_lowlight = len(pred_lowlight['det_polygons'])
        rec_texts_lowlight = pred_lowlight['rec_texts']

        # --- 统计与打印 ---
        stats['normal'].append({
            'name': img_name,
            'detect': count_normal,
            'rec': len(rec_texts_normal),
            'time': f"{time_normal:.2f}s"
        })
        stats['lowlight'].append({
            'name': img_name,
            'detect': count_lowlight,
            'rec': len(rec_texts_lowlight),
            'time': f"{time_lowlight:.2f}s"
        })

        print(f"✓ 正常图: 检测 {count_normal} 个, 识别 {len(rec_texts_normal)} 个")
        print(f"✓ 低照度: 检测 {count_lowlight} 个, 识别 {len(rec_texts_lowlight)} 个")
        print(f"  (正常耗时: {time_normal:.2f}s | 低照度耗时: {time_lowlight:.2f}s)")

# 打印总结报告
print("\n" + "=" * 70)
print("Phase 1 对比测试总结")
print("=" * 70)
print(f"{'图片':<25} {'正常检测':<10} {'低照度检测':<12} {'正常识别':<10} {'低照度识别':<12}")
print("-" * 70)

# 计算平均值
avg_detect_diff = 0
avg_rec_diff = 0

for n, l in zip(stats['normal'], stats['lowlight']):
    print(f"{n['name']:<25} {n['detect']:<10} {l['detect']:<12} {n['rec']:<10} {l['rec']:<12}")
    avg_detect_diff += (n['detect'] - l['detect'])
    avg_rec_diff += (n['rec'] - l['rec'])

avg_detect_diff /= len(all_images)
avg_rec_diff /= len(all_images)

print("-" * 70)
print(f"平均检测数量差异: {avg_detect_diff:+.1f}")
print(f"平均识别文本差异: {avg_rec_diff:+.1f}")
print("=" * 70)
print("\n✓ 对比测试完成！")
print(f"\n📁 结果已保存到：")
print(f"   1. 正常图片结果: {output_normal}/")
print(f"   2. 低照度图片结果: {output_lowlight}/")
print("\n📌 下一步建议：")
print("   1. 打开两个文件夹，直观对比检测框和识别文字的差异")
print("   2. 如果低照度效果显著下降，说明你的优化方向是正确的")
print("   3. 开始设计低照度增强模块（如 Zero-DCE）")