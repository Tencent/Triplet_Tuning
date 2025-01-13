import gradio as gr
import numpy as np
import torch
import requests 
import random
import os
import pdb
import sys
import copy
import json
import math
import types
import pickle
from PIL import Image
import base64
from io import BytesIO

from tqdm.auto import tqdm
from datetime import datetime
from safetensors.torch import load_file
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import Attention
from torchvision.utils import save_image
from diffusers import DDIMScheduler, DiffusionPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline

from .convert import convert_white_to_black
from .utils import *

Layers_control_net = {
                  "down_blocks":{
                                # "1":{
                                #     "0": list(range(0, 2)), 
                                #     "1": list(range(0, 2))
                                #     }, 
                                 "2":{
                                    "0": list(range(0, 10)), 
                                    "1": list(range(0, 10))
                                    }
                                 },
                  "mid_block":{"0":{"0": list(range(0, 10))}}, 
                }

Layers_ALL = {
                  "down_blocks":{
                                "1":{
                                    "0": list(range(0, 2)), 
                                    "1": list(range(0, 2))
                                    }, 
                                 "2":{
                                    "0": list(range(0, 10)), 
                                    "1": list(range(0, 10))
                                    }
                                 },
                  "mid_block":{"0":{"0": list(range(0, 10))}}, 
                  "up_blocks":{
                                "0":{
                                    "0": list(range(0, 10)), 
                                    "1": list(range(0, 10)), 
                                     "2": list(range(0, 10))
                                     }, 
                               "1":{
                                   "0": list(range(0, 2)), 
                                   "1": list(range(0, 2)), 
                                   "2": list(range(0, 2))
                                   }
                                }
                }



def process_layers_enabled(Layers_enabled):
    Layers_enabled_list = []
    for high_block, blocks in Layers_enabled.items():
        for block_key, block_two in blocks.items():
            for block_key2, block_list in block_two.items():
                for attention_index in block_list:
                    if high_block=="mid_block":
                        path = f"{high_block}.attentions.{block_key2}.transformer_blocks.{attention_index}"
                    else:
                        path = f"{high_block}.{block_key}.attentions.{block_key2}.transformer_blocks.{attention_index}"
                    Layers_enabled_list.append(path)
    return Layers_enabled_list

Layers_Enabled_ALL = process_layers_enabled(Layers_ALL)
Layers_Enabled_Control = process_layers_enabled(Layers_control_net)

def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (
        1,
        num_channels_latents,
        int(height) // self.vae_scale_factor,
        int(width) // self.vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents.repeat(batch_size, 1, 1, 1)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * self.scheduler.init_noise_sigma
    # pdb.set_trace()
    return latents


def blance_text_embeddings(cond_embeddings_first, DECODED_PROMPTS, high_noun_indices, beta=1., beta_color=0.75, beta_adj=1, beta_det=1, switch=2):
    mask_embeddings = torch.ones((77,1)).to(cond_embeddings_first.device)
    energy = torch.sum(cond_embeddings_first**2, axis=-1, keepdim=False).squeeze(0)
    # switch = 1 # 3 is good to explain
    if switch==1:
        cof = torch.sqrt(beta*energy[len(DECODED_PROMPTS)]/energy[high_noun_indices].max())
        for idx in high_noun_indices:
            mask_embeddings[idx] = cof
    elif switch==2: 
        for idx in high_noun_indices:
            cof = torch.sqrt(beta*energy[len(DECODED_PROMPTS)]/energy[idx])
            # print(energy[len(DECODED_PROMPTS)], energy[idx], cof)
            mask_embeddings[idx] = cof
    else:
        mask_embeddings[high_noun_indices] *= torch.sqrt(beta*energy[high_noun_indices].max()/energy[high_noun_indices]).unsqueeze(-1)

        # for idx in high_noun_indices:
        #     cof = torch.sqrt(beta*energy[high_noun_indices].max()/energy[idx])
        #     # print(energy[len(DECODED_PROMPTS)], energy[idx], cof)
        #     mask_embeddings[idx] = cof

    text = " ".join(DECODED_PROMPTS[1:])
    doc = nlp(text)

    color_indices = [i + 1 for i, token in enumerate(doc) if is_color_word(token.text.lower())]
    adjective_indices = [i+1 for i, token in enumerate(doc) if token.pos_ == 'ADJ']
    determiner_indices = [i+1 for i, token in enumerate(doc) if token.pos_ == 'DET'] # "a, the, my"
    # pdb.set_trace()
    if beta_color!=1:
        # mask_embeddings[color_indices] *= torch.sqrt(beta_color*energy[high_noun_indices].max()/energy[color_indices]).unsqueeze(-1)
        mask_embeddings[adjective_indices] *= torch.sqrt(beta_adj*energy[high_noun_indices].max()/energy[adjective_indices]).unsqueeze(-1)

    cond_embeddings_first = mask_embeddings*cond_embeddings_first

    return cond_embeddings_first


def amplify_max_min(tensor, alpha=2.):
    max_vals, _ = torch.max(tensor, dim=-2, keepdim=True)
    min_vals, _ = torch.min(tensor, dim=-2, keepdim=True) 
    # 对最大值的处理
    amplified_tensor = torch.where(
        (tensor == max_vals) & (max_vals > 0), tensor * alpha,  # 放大正的最大值
        torch.where((tensor == max_vals) & (max_vals < 0), tensor / alpha, tensor)  # 缩小负的最大值
    )

    # 对最小值的处理
    amplified_tensor = torch.where(
        (tensor == min_vals) & (min_vals > 0), amplified_tensor / alpha,  # 缩小正的最小值
        torch.where((tensor == min_vals) & (min_vals < 0), amplified_tensor * alpha, amplified_tensor)  # 放大负的最小值
    )
    return amplified_tensor


def amplify_feature_topk(hidden_states, value, mask, N=1, alpha=1, high_noun_indices=None, max_token=None):
    if N==0: return hidden_states

    # if len(hidden_states.shape)==4: pdb.set_trace()
    mask_expanded = mask.unsqueeze(-1).repeat_interleave(value.shape[-1], dim=-1)
    if len(hidden_states.shape)==3:
        high_noun_indices = torch.tensor(high_noun_indices).to(value.device).unsqueeze(-1).unsqueeze(0)
        high_noun_indices = high_noun_indices.repeat(value.shape[0], 1, value.shape[-1])
    else:
        high_noun_indices = torch.tensor(high_noun_indices).to(value.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        high_noun_indices = high_noun_indices.repeat(value.shape[0], value.shape[1], 1, value.shape[-1])
    topk_max_values, topk_max_indices = torch.topk(torch.abs(value[..., 1:max_token, :] if max_token is not None else value), N, dim=-2)
    topk_max_indices = topk_max_indices + 1
    value_mask = torch.zeros_like(value)
    value_mask_high_noun = torch.zeros_like(value)
    value_mask.scatter_(-2, topk_max_indices, 1)
    value_mask_high_noun.scatter_(-2, high_noun_indices, 1)
    value_mask_combine = value_mask * value_mask_high_noun
    hidden_states_mask = torch.sum(value_mask_combine[value_mask_combine.shape[0]//2:].unsqueeze(-3) * mask_expanded, dim=-2)
    hidden_states[int(hidden_states.size(0)/2):] = (hidden_states_mask>0)*alpha*hidden_states[int(hidden_states.size(0)/2):]
    return hidden_states


def init_latent_with_boxes(in_channels, height, width, batch_size, layouts, seed):
    # pdb.set_trace()
    # Step 1: 初始化背景latent0
    init_latent = torch.zeros((batch_size, in_channels, height, width))
    generator = torch.Generator().manual_seed(seed)
    # init_latent = torch.randn(
    #     (batch_size, in_channels, height, width),
    #     generator=generator,
    # )
    # Step 2: 对每个box生成新的latent片段并替换对应位置
    for i, layout in enumerate(layouts):
        # 为每个segment随机选择一个种子
        segment_seed = random.randint(0, 1000000000)
        generator.manual_seed(segment_seed)
        
        # 创建segment mask
        mask = layout.unsqueeze(0).repeat(batch_size, in_channels, 1, 1)
        
        # 生成segment对应的latent片段
        segment_latent = torch.randn((batch_size, in_channels, height, width), generator=generator) * (1.0 + 0.1*i)
        
        # 将segment latent叠加到背景latent
        init_latent = init_latent + segment_latent * mask.cpu()
    
        # pdb.set_trace()
    # save_image(init_latent[0, 0], "init_latent.png")
    init_latent /= init_latent.std()
        
    return init_latent


def save_attn_map(sim, sub_path, DECODED_PROMPTS, sa_, PROMPTS_LIST, postfix=""):
    os.makedirs(sub_path, exist_ok=True)
    if len(sim.size())==4:
        sim = sim[2]
        idx_fix = 1
    else:
        idx_fix = int(sim.shape[0]//2)
    hh = int(np.sqrt(sim.shape[1]))
    if sa_:
        save_imshow(sim[idx_fix,:,:], 
                    f"{sub_path}/self_iso_all{postfix}.png")
        for idx in [34, 175, 512, 856]:    
            save_imshow(sim[idx_fix,idx,:].view(hh, hh), 
                    f"{sub_path}/self_iso_postion_{idx}{postfix}.png")
    else:
        for idx in PROMPTS_LIST:    
            if idx<len(DECODED_PROMPTS):
                save_imshow(sim[idx_fix,:,idx].view(hh, hh), 
                            f"{sub_path}/cross_iso_{DECODED_PROMPTS[idx]}{postfix}.png",
                            # scale2=sim[idx_fix,:,:].max()
                            )

def save_value_map(value, sub_path, num=20, postfix=""):
    os.makedirs(sub_path, exist_ok=True)
    if len(value.size())==4:
        value = value[2]
        idx_fix = 1
    else:
        idx_fix = int(value.shape[0]//2)
        # for idx in [40, 41, 42, 44]:
    for idx in [idx_fix]:
        save_imshow(value[idx,0:40,:], 
                f"{sub_path}/value{idx}{postfix}.png")
        save_plot(value[idx,0:40,:], f"{sub_path}/value_plot_{idx}{postfix}.png", line_num=num, lines_per_subplot=1)
    np.save(f"{sub_path}/value{postfix}.npy", value.cpu().numpy())

def save_hidden_map(hidden_states, sub_path, num=20, postfix=""):
    os.makedirs(sub_path, exist_ok=True)
    if len(hidden_states.size())==4:
        hidden_states = hidden_states[2]
        idx_fix = 1
    else:
        idx_fix = int(hidden_states.shape[0]//2)
    for idx in range(num):    
        hh = int(np.sqrt(hidden_states.shape[1]))
        save_imshow(hidden_states[idx_fix,:,idx].view(hh, hh), 
                    f"{sub_path}/hidden_ch{idx}.png",
                    # scale2=hidden_states[idx_fix,:,:num].max()
                    )