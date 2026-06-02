import cv2
import numpy as np
import os

# 配置参数
input_dir = '/mmocr/demo'
output_dir = '/mmocr/lowlight_dataset'
os.makedirs(output_dir, exist_ok=True)

# 低照度参数
BRIGHTNESS_FACTOR = 0.3  # 亮度衰减系数 (0.3 表示只有 30% 亮度)
NOISE_LEVEL = 30  # 高斯噪声强度


def add_gaussian_noise(image, mean=0, sigma=NOISE_LEVEL):
    """添加高斯噪声"""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_low_light_effect(image):
    """应用低照度效果：降低亮度 + 添加噪声"""
    # 1. 降低亮度
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l = cv2.multiply(l, BRIGHTNESS_FACTOR)  # 降低亮度
    l = np.clip(l, 0, 255)
    l = cv2.merge([h, l, s])
    low_light_img = cv2.cvtColor(l, cv2.COLOR_HLS2BGR)

    # 2. 添加噪声
    noisy_img = add_gaussian_noise(low_light_img)

    return noisy_img


print("=" * 60)
print("开始生成低照度测试数据...")
print("=" * 60)

# 获取所有图片文件
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not image_files:
    print("错误：未在 demo 目录找到图片！")
else:
    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ 跳过无法读取的文件: {img_name}")
            continue

        # 处理图片
        low_light_img = apply_low_light_effect(img)

        # 保存
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, low_light_img)

        print(f"✓ 已转换并保存: {img_name} -> {save_path}")

print("\n" + "=" * 60)
print("✓ 低照度数据集生成完毕！")
print(f"📁 保存位置: {output_dir}")
print("=" * 60)
print("\n📌 接下来的操作：")
print("  1. 运行上述脚本生成数据")
print("  2. 修改 p1.py 中的 test_images 列表，指向新生成的 lowlight_dataset")
print("  3. 运行 p1.py 对比正常图与低照度图的结果差异")