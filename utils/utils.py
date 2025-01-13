import os
import pdb
import nltk
import torch
import spacy
import base64
import gradio as gr
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

ASCII_CHARS = "@%#*+=-:. "
MAX_COLORS = 12

# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")
### please undate you path of nltk_data
nltk.data.path.append('xxxx/miniconda3/envs/omost/nltk_data')
# nltk.download('wordnet')

def return_noun_indices(DECODED_PROMPTS):
    # pdb.set_trace()
    text = " ".join(DECODED_PROMPTS[1:])
    doc = nlp(text)
    noun_indices = [i+1 for i, token in enumerate(doc) if token.pos_ in ['NOUN', 'PROPN']]
    return noun_indices

def is_color_word(word):
    # 获取词的 synsets（同义词集）
    exempt_words = ["plain"]
    if word in exempt_words:
        return False    
    synsets = wn.synsets(word)
    
    # 检查同义词集中是否有颜色的语义（词义的定义中包含 'color'）
    for synset in synsets:
        if 'color' in synset.definition():
            return True
    return False

def return_mask_with_spacy(DECODED_PROMPTS, beta=1., beta_color=0.3, beta_adj=1.0, beta_det=1.0):
    # pdb.set_trace()
    mask_noun = torch.ones((77,1))
    text = " ".join(DECODED_PROMPTS[1:])
    doc = nlp(text)
    noun_indices = [i+1 for i, token in enumerate(doc) if token.pos_ in ['NOUN', 'PROPN']]
    adjective_indices = [i+1 for i, token in enumerate(doc) if token.pos_ == 'ADJ']
    determiner_indices = [i+1 for i, token in enumerate(doc) if token.pos_ == 'DET']
    color_indices = [i + 1 for i, token in enumerate(doc) if is_color_word(token.text.lower())]
    mask_noun[noun_indices] *= beta
    mask_noun[adjective_indices] *= beta_adj
    mask_noun[determiner_indices] *= beta_det
    mask_noun[color_indices] *= beta_color
    # print(mask_noun[:30])
    return mask_noun


def save_plot(data, filename, line_num=10, lines_per_subplot=2):
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    
    num_subplots = (line_num + lines_per_subplot - 1) // lines_per_subplot  # 向上取整

    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 4 * num_subplots))  # 动态调整高度

    if num_subplots == 1:
        axs = [axs]

    for i in range(num_subplots):
        start_idx = i * lines_per_subplot
        end_idx = min(start_idx + lines_per_subplot, line_num)  # 防止越界

        for j in range(start_idx, end_idx):
            axs[i].plot(list(range(1, len(data[:, j])+1)), data[:, j], marker="s", linewidth=2, 
                        label=f'index {j}_M{data[:, j].mean():.3f}_S{data[:, j].std():.3f}')
        
        axs[i].legend()
        axs[i].set_xlabel('Token index')
        axs[i].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_imshow(data, output_path='im.png', scale1=None, scale2=None, cmap='inferno'):
    # viridis， inferno, magma, terrain
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    plt.figure()  # 创建一个新的图形窗口
    if scale1 is None: scale1 = data.min()
    if scale2 is None: scale2 = data.max()
    plt.imshow(data, cmap=cmap, vmin=scale1, vmax=scale2)  # 显示带有 colormap 的图像
    cbar = plt.colorbar()  # 添加颜色条
    cbar.ax.tick_params(labelsize=16)
    plt.tick_params(labelsize=16)
    # output_path = os.path.join("visual", output_path)
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches='tight')  # 保存图像到文件，去除多余的空白
    plt.close()  # 关闭图形窗口，释放内存


def torch_to_pil_image(torch_tensor):
    """
    Converts a PyTorch tensor to a PIL image.
    Assumes the input is a 2D grayscale tensor or a 3D tensor with a single channel.
    """
    if torch_tensor.ndim == 3 and torch_tensor.shape[0] == 1:  # Single-channel image
        torch_tensor = torch_tensor.squeeze(0)
    torch_tensor = 255*(torch_tensor-torch_tensor.min())/(torch_tensor.max()-torch_tensor.min())
    # Convert to a PIL image
    pil_image = Image.fromarray(torch_tensor.to(torch.uint8).cpu().numpy(), mode='L')
    return pil_image

def resize_image(image, new_width=100):
    """
    Resize the image while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(new_width * aspect_ratio)
    return image.resize((new_width, new_height))

def grayscale(image):
    """
    Convert image to grayscale.
    """
    return image.convert("L")

def pixel_to_ascii(image):
    """
    Convert each pixel to an ASCII character based on its gray level.
    """
    pixels = np.array(image)
    ascii_str = ""
    for row in pixels:
        for pixel in row:
            ascii_str += ASCII_CHARS[pixel // 32]  # Scale pixel value to ASCII range
        ascii_str += "\n"
    return ascii_str

def print_ascii(torch_tensor, new_width=100):
    """
    Convert a PyTorch tensor to an ASCII representation and print it.
    """
    # Convert torch tensor to PIL image
    pil_image = torch_to_pil_image(torch_tensor)
    
    # Resize image
    pil_image = resize_image(pil_image, new_width)
    
    # Convert image to grayscale
    pil_image = grayscale(pil_image)
    
    # Convert grayscale image to ASCII
    ascii_str = pixel_to_ascii(pil_image)
    
    # Print ASCII string
    print(ascii_str)


def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix

def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

def process_sketch(canvas_data):
    binary_matrixes = []
    base64_img = canvas_data['image']
    image_data = base64.b64decode(base64_img.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    im2arr = np.array(image)
    colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in canvas_data['colors']]
    colors_fixed = []
    
    r, g, b = 255, 255, 255
    binary_matrix = create_binary_matrix(im2arr, (r,g,b))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    
    for color in colors:
        r, g, b = color
        if any(c != 255 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r,g,b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(0,0,0)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
            
    visibilities = []
    colors = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]

    color_merge = np.zeros_like(colors[0]["value"])
    for idx in range(1, len(colors)):
        color_single = colors[idx]
        if "value" in color_single:
            color_merge = color_merge + np.array(color_single['value'])
    color_merge = Image.fromarray(color_merge.astype(np.uint8))
    
    return [color_merge, gr.update(visible=True), binary_matrixes, *visibilities, *colors]

def process_prompts(binary_matrixes, *seg_prompts):
    return [gr.update(visible=True), gr.update(value=' , '.join(seg_prompts[:len(binary_matrixes)]))]

def process_example(layout_path, all_prompts, seed_, switch_prompt):
    
    all_prompts = all_prompts.split('***')
    
    binary_matrixes = []
    colors_fixed = []
    
    im2arr = np.array(Image.open(layout_path))[:,:,:3]
    unique, counts = np.unique(np.reshape(im2arr,(-1,3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    binary_matrix = create_binary_matrix(im2arr, (0,0,0))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_*(255,255,255) + (1-binary_matrix_)*(50,50,50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    
    for i in range(len(all_prompts)-1):
        r, g, b = unique[sorted_idx[i]]
        if any(c != 255 for c in (r, g, b)) and any(c != 0 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r,g,b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
            
    visibilities = []
    colors = []
    prompts = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
        prompts.append(gr.update())
        
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
        prompts[n] = all_prompts[n+1]
        
    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors, *prompts,
            gr.update(visible=True), gr.update(value=all_prompts[0]), 
            int(seed_), Image.open(layout_path), switch_prompt]