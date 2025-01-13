import argparse
from PIL import Image

def convert_white_to_black(image_path):
    # 打开图像
    image = Image.open(image_path)

    # 将图像转换为RGBA模式以处理透明度
    image = image.convert("RGBA")

    # 获取图像的像素数据
    pixels = image.load()

    # 获取图像尺寸
    width, height = image.size

    # 遍历每个像素并将纯白色像素转换为黑色
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r == 255 and g == 255 and b == 255 and a == 255:  # 检查纯白色像素
                pixels[x, y] = (0, 0, 0, a)  # 将纯白色像素变为黑色

    # 保存修改后的图像
    image.save(image_path)

    # 显示修改后的图像
    image.show()

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Convert pure white pixels in an image to black.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image file.")
    
    args = parser.parse_args()
    
    # 调用函数处理图像
    convert_white_to_black(args.input)

if __name__ == "__main__":
    main()
