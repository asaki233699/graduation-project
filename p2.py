# mmocr/p3.py
"""
毕设 Phase 3: 手动绘制对比图 (绕过 mmocr 可视化报错)
"""

import cv2
import numpy as np
import os
import time
from mmocr.apis import MMOCRInferencer

# ==========================================
# 配置区域
# ==========================================
normal_dir = 'C:/Users/dingy/PycharmProjects/bishe/mmocr/demo'
lowlight_dir = 'C:/Users/dingy/PycharmProjects/bishe/mmocr/lowlight_dataset'

output_base = 'outputs/phase1_comparison'
output_combined = os.path.join(output_base, 'comparison')
os.makedirs(output_combined, exist_ok=True)

print("=" * 70)
print("毕设 Phase 3: 手动绘制对比图")
print("=" * 70)
print(f"输出位置: {output_combined}")
print("=" * 70)

# 初始化 OCR 推理器
ocr = MMOCRInferencer(det='DBNet', rec='CRNN')
print("✓ 模型加载完成")
print("=" * 70)

# 准备数据
all_images = set(os.listdir(normal_dir)).intersection(set(os.listdir(lowlight_dir)))
all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not all_images:
    print("⚠️ 未找到可对比的图片！")
else:
    print(f"找到 {len(all_images)} 张图片待处理...\n")

    for i, img_name in enumerate(all_images):
        print(f"[{i + 1}/{len(all_images)}] 正在处理: {img_name}")

        path_n = os.path.join(normal_dir, img_name)
        path_l = os.path.join(lowlight_dir, img_name)

        # --- 步骤 1: 获取预测数据 ---
        result_n = ocr(path_n, return_vis=False)
        polygons_n = result_n['predictions'][0]['det_polygons']
        texts_n = result_n['predictions'][0]['rec_texts']

        result_l = ocr(path_l, return_vis=False)
        polygons_l = result_l['predictions'][0]['det_polygons']
        texts_l = result_l['predictions'][0]['rec_texts']

        # 打印调试信息，确认是否真的检测到了
        print(f"  -> [调试] 正常图检测到 {len(polygons_n)} 个框")
        print(f"  -> [调试] 低照度图检测到 {len(polygons_l)} 个框")

        print("  -> 获取低照度图预测数据...")
        result_l = ocr(path_l, return_vis=False)
        polygons_l = result_l['predictions'][0]['det_polygons']

        # --- 步骤 2: 读取原始图片 ---
        img_n = cv2.imread(path_n)
        img_l = cv2.imread(path_l)

        if img_n is None or img_l is None:
            print(f"  ⚠️ 跳过: 图片读取失败")
            continue


        # --- 步骤 3: 手动绘制检测框和文字 ---
        def draw_boxes_and_texts(img, polygons, texts):
            if polygons is None or len(polygons) == 0:
                return img

            for i, poly in enumerate(polygons):
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)

                # 画框 (绿色线条，加粗到 3)
                cv2.polylines(img, [pts], True, (0, 255, 0), 3)

                # 计算文字放置的位置 (取多边形最上面的点)
                # pts 形状是 (N, 1, 2)，我们取所有点的 y 坐标最小的那个点上方
                x_coords = pts[:, 0, 0]
                y_coords = pts[:, 0, 1]
                text_x = min(x_coords)
                text_y = min(y_coords) - 10

                # 确保文字不会画在图片外面
                text_y = max(text_y, 20)

                # 获取识别的文字
                text = texts[i] if i < len(texts) else ""

                # 写文字 (白色背景，黑色字体，方便看清)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, text, (text_x, text_y), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            return img


        img_n = draw_boxes_and_texts(img_n, polygons_n, texts_n)
        img_l = draw_boxes_and_texts(img_l, polygons_l, texts_l)

        # --- 步骤 4: 并排拼接 ---
        h = max(img_n.shape[0], img_l.shape[0])
        w = img_n.shape[1] + img_l.shape[1]

        # 创建画布 (黑色背景)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:img_n.shape[0], :img_n.shape[1]] = img_n
        canvas[:img_l.shape[0], img_n.shape[1]:] = img_l

        # --- 步骤 5: 保存 ---
        save_path = os.path.join(output_combined, img_name)
        cv2.imwrite(save_path, canvas)

        print(f"  ✓ 已保存对比图: {save_path}")

print("\n" + "=" * 70)
print("✅ 对比图生成完成！")
print(f"📁 请查看: {output_combined}")
print("=" * 70)