import cv2
import numpy as np
import os
import random

# ==================== 配置参数 ====================
INPUT_DIR = 'C:/Users/dingy/PycharmProjects/bishe/mmocr/demo'
OUTPUT_DIR = 'C:/Users/dingy/PycharmProjects/bishe/mmocr/lowlight_dataset'

# 控制随机范围，避免“一刀切”
MIN_BRIGHTNESS = 0.25  # 最暗为原图的 25%
MAX_BRIGHTNESS = 0.55  # 最亮为原图的 55%
MIN_NOISE = 10  # 最小噪声标准差
MAX_NOISE = 25  # 最大噪声标准差
# =================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_low_light_effect(image):
    """
    自适应低照度退化算法：
    1. 在 HLS 空间对 L 通道进行随机比例衰减。
    2. 仅在 L 通道上叠加单通道高斯噪声，模拟真实的低照度亮度噪点。
    """
    # 随机生成当前图片的退化参数
    brightness_factor = random.uniform(MIN_BRIGHTNESS, MAX_BRIGHTNESS)
    noise_sigma = random.uniform(MIN_NOISE, MAX_NOISE)

    # 1. 转换到 HLS 颜色空间
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    # 2. 降低亮度通道 (L)
    l = cv2.multiply(l, brightness_factor)
    l = np.clip(l, 0, 255)

    # 3. 构造单通道高斯噪声并叠加到亮度通道 (避免产生不真实的彩色雪花点)
    row, col = l.shape
    gauss = np.random.normal(0, noise_sigma, (row, col))
    noisy_l = l + gauss
    noisy_l = np.clip(noisy_l, 0, 255).astype(np.uint8)

    # 4. 合并通道并转回 BGR
    low_light_hls = cv2.merge([h, noisy_l, s])
    low_light_rgb = cv2.cvtColor(low_light_hls, cv2.COLOR_HLS2BGR)

    return low_light_rgb, brightness_factor, noise_sigma


def main():
    print("=" * 60)
    print("🚀  开始生成低照度测试数据集 (改进版) ...")
    print(f"输入路径: {INPUT_DIR}")
    print(f"输出路径: {OUTPUT_DIR}")
    print("=" * 60)

    # 检查输入目录是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误：找不到输入目录 {INPUT_DIR}，请检查路径！")
        return

    # 支持常见的图片格式
    valid_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("⚠️ 提示：未在指定目录找到任何图片文件！")
        return

    success_count = 0

    for img_name in image_files:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"❌ 跳过损坏或无法读取的文件: {img_name}")
            continue

        # 核心算法处理
        processed_img, b_factor, n_sigma = apply_low_light_effect(img)

        # 保存图片
        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, processed_img)

        success_count += 1
        print(f"[{success_count:02d}] 转换成功: {img_name} -> (亮度: {b_factor:.2f}, 噪声: {n_sigma:.1f})")

    print("=" * 60)
    print(f"🎉 处理完成！成功转换并保存了 {success_count} 张低照度图像。")
    print("=" * 60)


if __name__ == '__main__':
    main()