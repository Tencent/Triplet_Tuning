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
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import Attention
from torchvision.utils import save_image
from diffusers import DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DiffusionPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from utils.convert import convert_white_to_black
from utils.utils import preprocess_mask, process_sketch, process_example, print_ascii, save_imshow, save_plot
from utils.control import *

#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''


#################################################
#################################################
DECODED_PROMPTS = None
HIGH_NOUN = None
PROMPT_MASK = None
DEBUG_PATH = "visual"
# DEBUG_STEP=[0, 3, 7, 31]
DEBUG_STEP=[0, 1, 2, 3, 7, 15, 23, 31]
# ControlNet: D2-first, Mid-first
# DEBUG_Layer_Control=[0, 1, 40, 41] 
DEBUG_Layer_Control=[38, 39, 58, 59] 
# ControlNet: D1-first; Dense: D1/D2/Mid/U0/U1-first
# DEBUG_Layer_Dense=[0, 1, 68, 69, 76, 77, 116, 117, 136, 137, 196, 197] 
DEBUG_Layer_Dense=[6, 7, 74, 75, 114, 115, 134, 135, 194, 195, 206, 207]
VALUE_INDICES={}
self_maps_dict1 = {i: [] for i in DEBUG_STEP}
self_maps_dict2 = {i: [] for i in DEBUG_STEP}
self_feat_dict = {i: [] for i in DEBUG_STEP}
cross_maps_dict1 = {i: [] for i in DEBUG_STEP}
cross_maps_dict2 = {i: [] for i in DEBUG_STEP}
cross_feat_dict = {i: [] for i in DEBUG_STEP}
#################################################
#################################################
global sreg, creg, sizereg, COUNT, COUNT_DUAL, creg_maps, sreg_maps, pipe, text_cond


sreg = 0
creg = 0
sizereg = 0
dense_step = 0
alpha = 1
FEAT_V = 2
FEAT_NUM = 2
COUNT = 0
COUNT_DUAL = 0
reg_sizes = {}
creg_maps = {}
sreg_maps = {}
text_cond = 0
device="cuda"
MAX_COLORS = 12
ALL_LAYERS = 140+34*2 # cross attention + self-attention
DENSE_LAYERS = 0 # cross attention + self-attention
HF_TOKEN = ''

View = "Isometric"

### please undate you path of base models
HF_HOME = "xxxxx"
t2i_base_model = f"{HF_HOME}/huggingface/stable-diffusion-xl-base-1.0"
contorl_path= f"{HF_HOME}/huggingface/controlnet-union-sdxl-1.0"

out_width, out_height = 1024, 1024

# pipe = DiffusionPipeline.from_pretrained(t2i_base_model, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
# configure ip-adapter scales. down_b_0-ResNet/down_b_1-2x2/down_b_2-2x10 + mid-1x10 + up_b_0-3x10/up_b_1-3x2/
# 70 self-attention + 70 cross-attention
controlnet = ControlNetModel.from_pretrained(contorl_path, torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
	t2i_base_model, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")
print(pipe.unet)
pipe.enable_model_cpu_offload()


# base_num
num_inference_global = 50
pipe.scheduler.set_timesteps(num_inference_global)
timesteps = pipe.scheduler.timesteps
sp_sz = pipe.unet.sample_size

def load_json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

val_prompt = load_json_to_dict('./dataset/prompts.json')
val_layout = './dataset/valset_layout/'
# pdb.set_trace()
#################################################
#################################################
def process_prompts(binary_matrixes, *seg_prompts):
    return [gr.update(visible=True), gr.update(value=' , '.join(seg_prompts[:len(binary_matrixes)])), \
            gr.update(value=(' , '.join(seg_prompts[:len(binary_matrixes)])).replace("game scene", "bare terrain"))]

def set_seed_default(seed_int = 4562489):
    if seed_int==-1:
        seed_int = random.randint(0, 1000000000)
        print(f"Using random seed: {seed_int}")
    np.random.seed(seed_int) # set random seed for numpy
    random.seed(seed_int) # set random seed for random
    torch.manual_seed(seed_int) # set random seed for cpu
    torch.cuda.manual_seed(seed_int) # set random seed for gpu
    torch.cuda.manual_seed_all(seed_int) # set random seed for gpu
    torch.backends.cudnn.benchmark = True # benchmark mode is good whenever your input sizes for your network do not vary

def generate_examples(val_layout, val_prompt):
    examples = []
    for key, value in val_prompt.items():
        example = [
            val_layout + f"{key}.png",
            '***'.join([value['textual_condition']] + value['segment_descriptions']),
            value['seed'],  # 生成一个伪随机种子
            value["switch_prompt"]
        ]
        examples.append(example)
    return examples


def mod_forward_dual(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        sa_ = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            sa_ = True
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        global COUNT_DUAL, alpha
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for self.scale when we move to Torch 2.1
        if sa_:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # key[key.shape[0]//2:] *= PROMPT_MASK.unsqueeze(0).to(key.dtype)
            hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            mask = creg_maps[hidden_states.size(-2)].unsqueeze(1).repeat_interleave(repeats=self.heads, dim=1) #.repeat(self.heads,1,1)
            hidden_states = amplify_feature_topk(hidden_states, value, mask, N=FEAT_NUM, alpha=FEAT_V, \
                                             high_noun_indices=HIGH_NOUN, max_token=len(DECODED_PROMPTS)) 

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        COUNT_DUAL += 1

        return hidden_states


def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
    residual = hidden_states
    # print(hidden_states.shape)

    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    # print(self)
    # if type(encoder_hidden_states)==tuple: pdb.set_trace()
    batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)

    global sreg, creg, COUNT, COUNT_DUAL, creg_maps, sreg_maps, reg_sizes, text_cond, dense_step, alpha
    
    sa_ = True if encoder_hidden_states is None else False
    encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
        
    if self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    if sa_:
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
    else:
        key = self.head_to_batch_dim(key)
        # key[key.shape[0]//2:] *= PROMPT_MASK.unsqueeze(0).to(key.dtype)
        value = self.head_to_batch_dim(value)

    if COUNT/DENSE_LAYERS < dense_step:
        #  32 = (16 self + 16 cross), no more than 15 steps
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                        dtype=query.dtype, device=query.device),
                            query, key.transpose(-1, -2), beta=0, alpha=self.scale)
        # out=β×input+α×(batch1×batch2), self.scale=0.15811388300841897, torch.Size([16, 4096, 4096])
        
        # treg = 1
        treg = torch.pow(timesteps[COUNT//DENSE_LAYERS]/1000, 5)

        ## reg at self-attn
        if sa_:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1) # torch.Size([8, 4096, 1])
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1) # torch.Size([8, 4096, 1])
            mask = sreg_maps[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
            # sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
            
        ## reg at cross-attn
        else:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = creg_maps[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)

            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)

        attention_probs = sim.softmax(dim=-1)
        # if not sa_: 
            # attention_probs = torch.where(attention_probs > 0.1, torch.tensor(1.0), attention_probs)
        attention_probs = attention_probs.to(dtype)
            
    else:
        attention_probs = self.get_attention_scores(query, key, attention_mask)

    if sa_:
        hidden_states = torch.bmm(attention_probs, value)
    else:
        hidden_states = torch.bmm(attention_probs, value)
        mask = creg_maps[hidden_states.size(-2)].repeat_interleave(repeats=self.heads, dim=0) #.repeat(self.heads,1,1)
        hidden_states = amplify_feature_topk(hidden_states, value, mask, N=FEAT_NUM, alpha=FEAT_V, \
                                             high_noun_indices=HIGH_NOUN, max_token=len(DECODED_PROMPTS)) 

    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    COUNT += 1
    COUNT_DUAL += 1

    return hidden_states

#################################################
#################################################

layer_dense, layer_control, layer_dual = 0, 0, 0

for name, _module in pipe.unet.named_modules():
    if name in Layers_Enabled_ALL:
        if _module.attn1.__class__.__name__ == "Attention":
            _module.attn1.forward = types.MethodType(mod_forward_dual, _module.attn1)
            layer_dual += 1
        if _module.attn2.__class__.__name__ == "Attention":
            _module.attn2.forward = types.MethodType(mod_forward_dual, _module.attn2)
            layer_dual += 1

for name, _module in pipe.controlnet.named_modules():
    if name in Layers_Enabled_ALL:
        if name in Layers_Enabled_Control:
            if _module.attn1.__class__.__name__ == "Attention":
                _module.attn1.forward = types.MethodType(mod_forward, _module.attn1)
                layer_control += 1
            if _module.attn2.__class__.__name__ == "Attention":
                _module.attn2.forward = types.MethodType(mod_forward, _module.attn2)
                layer_control += 1
        else:
            if _module.attn1.__class__.__name__ == "Attention":
                _module.attn1.forward = types.MethodType(mod_forward_dual, _module.attn1)
                layer_dual += 1
            if _module.attn2.__class__.__name__ == "Attention":
                _module.attn2.forward = types.MethodType(mod_forward_dual, _module.attn2)
                layer_dual += 1


DENSE_LAYERS = layer_dense+layer_control
print(f" Modified: {DENSE_LAYERS}, dense: {layer_dense}, control: {layer_control}; dual: {layer_dual}")

# pipe.prepare_latents = types.MethodType(prepare_latents, pipe)


#################################################
def process_generation(image_sketch, layout_path, binary_matrixes, seed, creg_, sreg_, sizereg_, dense_step_, bsz, \
                       master_prompt, negative_prompt, feat_value, feat_num, s_prompts, switch_prompt, switch_balance, \
                    control_scale_, guidance_, infer_steps_, control_start_, control_end_, convert_to_gray, *prompts):

    global creg, sreg, sizereg, dense_step, timesteps, num_inference_global, \
        DECODED_PROMPTS, PROMPT_MASK, HIGH_NOUN, FEAT_V, FEAT_NUM
    creg, sreg, sizereg, dense_step = creg_, sreg_, sizereg_, dense_step_,
    S_PROMPT, FEAT_V, FEAT_NUM = s_prompts, feat_value, feat_num
    num_inference_global = infer_steps_

    pipe.scheduler.set_timesteps(infer_steps_)
    timesteps = pipe.scheduler.timesteps

    clipped_prompts = prompts[:len(binary_matrixes)]
    # master_prompt + Background prompt + N entities prompts general_prompt_dual
    # prompts = [master_prompt] + list(clipped_prompts) + [general_prompt_dual]
    prompts = [master_prompt] + list(clipped_prompts) + [master_prompt]
    # layouts torch.Size([5, 1, 64, 64]) Background + N sub prompts
    # binary_matrixes 5 (512, 512)
    layouts = torch.cat([preprocess_mask(mask_, sp_sz, sp_sz, device) for mask_ in binary_matrixes])

    # shape: torch.Size([6, 77, 2048]), torch.Size([6, 77, 2048]), torch.Size([6, 1280]), torch.Size([6, 1280)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        device=device,
        num_images_per_prompt=1,
        negative_prompt=[negative_prompt]*(len(prompts)-1)+[negative_prompt], # empty_prompt
        negative_prompt_2=[negative_prompt]*(len(prompts)-1)+[negative_prompt], # empty_prompt
    )

    text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt",)
    # text_input_2 = pipe.tokenizer_2(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
    #                             max_length=pipe.tokenizer_2.model_max_length, truncation=True, return_tensors="pt",)
    input_ids = text_input['input_ids'][0]
    DECODED_PROMPTS = [pipe.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in input_ids if ids!=49407]
    PROMPT_MASK = return_mask_with_spacy(DECODED_PROMPTS, beta=S_PROMPT).to(device)
    Prompt_Dict = {string: strength for string, strength in zip(DECODED_PROMPTS, list(PROMPT_MASK.squeeze(-1).cpu().numpy()))}
    prompt_embeds_word, _, _, _, = pipe.encode_prompt(prompt=["single"]+DECODED_PROMPTS[1:], prompt_2=["single"]+DECODED_PROMPTS[1:], device=device)
    HIGH_NOUN = return_noun_indices(DECODED_PROMPTS)
    print(f"Decoded prompts {len(DECODED_PROMPTS)}: {DECODED_PROMPTS}")

    ###########################
    ###### prep for creg ######
    ###########################
    pww_maps = torch.zeros(1,77,sp_sz,sp_sz).to(device)
       
    uncond_embeddings = negative_prompt_embeds[:1]
    cond_embeddings = prompt_embeds.detach().clone()
    # pdb.set_trace()
    for i in range(1,len(prompts)-1):
        # Except for the first prompt, the rest of the prompts are the segment prompts.
        # Then extract the valid segment text_input by ignoring the start and end tokens.
        # if i==len(prompts)-2: pdb.set_trace()
        wlen = text_input['length'][i] - 2 # ignore the start and end tokens
        widx = text_input['input_ids'][i][1:1+wlen] # valid tokens
        for j in range(77):
            try:
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    # use sub prompt as the condition to match the main prompts, and make the masks matched to 1.
                    # activate the valid tokens and conda_embeddings.
                    pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]

                    if switch_prompt==2:
                        cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen] if i >1 else cond_embeddings[i][1:1+wlen]
                    # cond_embeddings[0][j:j+wlen] = PROMPT_MASK[j:j+wlen]*cond_embeddings[i][1:1+wlen] if i >1 else cond_embeddings[i][1:1+wlen]
                    elif switch_prompt==1:
                        for idx in HIGH_NOUN:
                            if idx in range(j, j+wlen):
                                cond_embeddings[0][idx] = prompt_embeds_word[idx][1]
                            # cond_embeddings[0][idx] = PROMPT_MASK[idx]*prompt_embeds_word[idx][1]
                    else:
                        pass
                    if i==1: j1_background = list(range(j, j+wlen))
                    break
            except:
                # raise gr.Error("Please check whether every segment prompt is included in the full text !")
                print(f"Please check whether every segment {prompts[i]} is included in the full text {prompts[0]}!")

    pww_maps[0][j1_background] += 1 - torch.clamp(pww_maps.sum(1), max=1.0, min=0.)
    # pww_maps[0, 1:7, ...] = pww_maps[0][j1_background[0]]
    # pdb.set_trace()

    global creg_maps
    creg_maps = {}
    for r in range(4):
        res = int(sp_sz/np.power(2,r))
        layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1)
        creg_maps[np.power(res, 2)] = layout_c

    ###########################
    ###### prep for sreg ###### 
    ###########################
    global sreg_maps, reg_sizes
    sreg_maps = {} # {4096: torch.Size([1, 4096, 4096]), 1024:xxx, 256:xxx, 64:xxx}, segment maps
    reg_sizes = {} # 1-nomalize Layouts in the last dimension.
    

    for r in range(4):
        # pdb.set_trace()
        res = int(sp_sz/np.power(2,r))
        layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
        layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0)
        layouts_s = layouts_s
        # sizereg, The degree of mask-area adaptive adjustment
        reg_sizes[np.power(res, 2)] = 1-sizereg*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
        sreg_maps[np.power(res, 2)] = layouts_s

    
    ###########################    
    #### prep for text_emb ####
    ###########################
    global text_cond
    text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1]])    
    
    global COUNT, COUNT_DUAL
    COUNT, COUNT_DUAL = 0, 0
    

    set_seed_default(seed)
    
    cond_embeddings_first = cond_embeddings[0:1].clone()
    if switch_balance: 
        cond_embeddings_first = blance_text_embeddings(cond_embeddings_first, DECODED_PROMPTS, HIGH_NOUN, beta=S_PROMPT, switch=switch_balance)
    
    # pdb.set_trace()
    if convert_to_gray:
        image_sketch = image_sketch.convert("L")
    image = pipe(
                # prompts[0:1] + prompts[0:1], 
                # prompt_2=prompts[:1]*bsz,
                # latents=latents,
                # image=image_sketch_dual,
                image=[image_sketch],
                controlnet_conditioning_scale=float(control_scale_),
                control_guidance_start=float(control_start_/num_inference_global),
                control_guidance_end=float(control_end_/num_inference_global),
                guidance_scale=float(guidance_),
                num_inference_steps=num_inference_global,
                prompt_embeds=cond_embeddings_first,
                pooled_prompt_embeds=pooled_prompt_embeds[0:1],
                negative_prompt=[negative_prompt],
                negative_prompt_2=[negative_prompt],
                width=out_width, height=out_height,
                # guess_mode = True,
    ).images

    return [image_sketch], image


#################################################
#################################################
### define the interface

    
with gr.Blocks(css=css) as demo:
    binary_matrixes = gr.State([])
    color_layout = gr.State([])
    gr.Markdown('''## T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation''')
    gr.Markdown('''
    #### 😺 Instruction to generate images 😺 <br>
    (1) Create the image layout. <br>
    (2) Label each segment with a text prompt. <br>
    (3) Adjust the full text. The default full text is automatically concatenated from each segment's text. The default one works well, but refineing the full text will further improve the result. <br>
    (4) Check the generated images, and tune the hyperparameters if needed. <br>
    ''')
    
    with gr.Column():
        with gr.Group(elem_id="main-image"):
            canvas_data = gr.JSON(value={}, visible=False)
            with gr.Row():
                canvas = gr.HTML(canvas_html)
            image_sketch = gr.State(None)
            button_run = gr.Button("(1) I've finished my sketch ! 😺", elem_id="main_button", interactive=True)
      
            prompts = []
            colors = []
            color_row = [None] * MAX_COLORS
            with gr.Row():
                with gr.Column(visible=False) as post_sketch:
                    for n in range(MAX_COLORS):
                        if n == 0 :
                            with gr.Row(visible=False) as color_row[n]:
                                colors.append(gr.Image(label="background", type="pil", image_mode="RGB"))
                                prompts.append(gr.Textbox(label="Prompt for the background (white region)", value=""))
                        else:
                            with gr.Row(visible=False) as color_row[n]:
                                colors.append(gr.Image(label="segment "+str(n), type="pil", image_mode="RGB"))
                                prompts.append(gr.Textbox(label="Prompt for the segment "+str(n)))
                        
                    get_genprompt_run = gr.Button("(2) I've finished segment labeling ! 😺", elem_id="prompt_button", interactive=True)

                with gr.Column(visible=False) as gen_prompt_vis:
                    general_prompt = gr.Textbox(value='', label="(3) Textual Description for the entire image", interactive=True)
                    negative_prompt = gr.Textbox(value='distorted, low res', \
                                                label="Negative prompt for the entire image: distorted, low res", interactive=True)
                    seed_ = gr.Slider(label="Seed: Use -1 for a random seed", minimum=-1, maximum=999999999, value=-1, step=1)
                    convert_to_gray = gr.Slider(label="0 for colorful sketch, 1 for gray sketch", minimum=0, maximum=1, value=0, step=1)
                    # bridge, house, artifacts, human, tree, waterfall, ugly, disproportionate, distorted, low res, rundown
                    with gr.Accordion("Characteristics Prominence Parameters", open=True):
                        with gr.Row():
                            feat_value = gr.Slider(label="Strength to enhance the feature", minimum=0, maximum=10, value=2, step=0.1)
                            feat_num = gr.Slider(label="Top K to control the feature", minimum=0, maximum=20, value=2, step=1)
                    with gr.Accordion("Prompt Balance Parameters", open=True):
                        label = gr.Markdown("Switch to replace tokens of prompt, 0:original, 1 for single or few-word subprompts, 2 for complex subprompts with adjective words.")
                        switch_prompt = gr.Slider(label="Switch:0/1/2", minimum=0, maximum=3, value=1, step=1)
                        with gr.Row(visible=False):
                            s_prompt = gr.Slider(label="Strength to control the prompt", minimum=0, maximum=3, value=1, step=0.1)
                            switch_balance = gr.Slider(label="switch the balance of the prompt, 0, u1, u2, u3", minimum=0, maximum=3, value=2, step=1)
                    with gr.Accordion("Dense Control Parameters", open=True):
                        with gr.Row():
                            creg_ = gr.Slider(label="Dense-Control: w\u1D9C (The degree of attention modulation at cross-attention layers) ", minimum=0, maximum=5., value=.1, step=0.01)
                            sreg_ = gr.Slider(label="Dense-Control:  w \u02E2 (The degree of attention modulation at self-attention layers) ", minimum=0, maximum=5., value=.1, step=0.01)

                    with gr.Accordion("(4) Tune the hyperparameters", open=False):
                        with gr.Row():
                            sizereg_ = gr.Slider(label="Dense-Control: The degree of mask-area adaptive adjustment", minimum=0, maximum=1., value=0., step=0.1)
                            dense_step_ = gr.Slider(label="Dense-Control: The process ratio of the valid inference steps", minimum=0, maximum=100, value=32, step=1)
                        with gr.Row():
                            control_start_ = gr.Slider(label="ControlNet: The start of controlnet", minimum=0, maximum=100, value=0, step=1)
                            control_end_ = gr.Slider(label="ControlNet: The end of controlnet", minimum=0, maximum=100, value=32, step=1)
                        with gr.Row():
                            control_scale_ = gr.Slider(label="ControlNet: The strength of controlnet", minimum=0, maximum=10., value=0.5, step=0.01)
                            bsz_ = gr.Slider(label="Number of Samples to generate", minimum=1, maximum=5, value=1, step=1)
                        with gr.Row():
                            infer_steps_ = gr.Slider(label="Inference steps ", minimum=0, maximum=100, value=32, step=1)
                            guidance_ = gr.Slider(label="Inference guidance", minimum=0, maximum=10., value=9., step=0.1)
                        
                    final_run_btn = gr.Button("Generate ! 😺")
                    
                    layout_path = gr.Textbox(label="layout_path", visible=False)
                    all_prompts = gr.Textbox(label="all_prompts", visible=False)
                
                    # image_sketch_control = gr.Image(label="Sketch for Controlnet", type="pil", image_mode="RGB", height='auto')
                    out_image = gr.Gallery(label="Result", columns=1, height='auto')
                    image_sketch_control = gr.Gallery(label="Sketch for Controlnet", columns=1, height='auto')
    

    button_run.click(process_sketch, inputs=[canvas_data], outputs=[image_sketch, post_sketch, binary_matrixes, *color_row, *colors], js=get_js_colors, queue=False)
    
    get_genprompt_run.click(process_prompts, inputs=[binary_matrixes, *prompts], outputs=[gen_prompt_vis, general_prompt], queue=False)
    
    final_run_btn.click(process_generation, inputs=[image_sketch, layout_path, binary_matrixes, seed_, creg_, sreg_, sizereg_, dense_step_, bsz_, \
                                                    general_prompt, negative_prompt, feat_value, feat_num, s_prompt, switch_prompt, switch_balance, \
        control_scale_, guidance_, infer_steps_, control_start_, control_end_, convert_to_gray, *prompts], outputs=[image_sketch_control, out_image])

    gr.Examples(
        examples=generate_examples(val_layout, val_prompt),
        inputs=[layout_path, all_prompts, seed_, switch_prompt],
        outputs=[post_sketch, binary_matrixes, *color_row, *colors, *prompts, gen_prompt_vis, general_prompt, seed_, image_sketch, switch_prompt],
        fn=process_example,
        run_on_click=True,
        label='😺 Examples 😺',
    )

    demo.load(None, None, None, js=load_js)
    
demo.launch(server_name="0.0.0.0", server_port=80)