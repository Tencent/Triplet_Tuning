## T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation

> #### Authors &emsp;&emsp; Zhenhong Sun, Yifu Wang, Yonhon Ng, Yunfei Duan, Daoyi Dong, Hongdong Li, Pan Ji 

> #### Abstract
Scene generation is crucial to many computer graphics applications. Recent advances in generative AI have streamlined sketch-to-image workflows, easing the workload for artists and designers in creating scene concept art. However, these methods often struggle for complex scenes with multiple detailed objects, sometimes missing small or uncommon instances. In this paper, we propose a Training-free Triplet Tuning for Sketch-to-Scene (T3-S2S) generation after reviewing the entire cross-attention mechanism. This scheme revitalizes the existing ControlNet model, enabling effective handling of multi-instance generations, involving prompt balance, characteristics prominence, and dense tuning. Specifically, this approach enhances keyword representation via the prompt balance module, reducing the risk of missing critical instances. It also includes a characteristics prominence module that highlights TopK indices in each channel, ensuring essential features are better represented based on token sketches. Additionally, it employs dense tuning to refine contour details in the attention map, compensating for instance-related regions. Experiments validate that our triplet tuning approach substantially improves the performance of existing sketch-to-image models. It consistently generates detailed, multi-instance 2D images, closely adhering to the input prompts and enhancing visual quality in complex multi-instance scenes.


> #### Environment and Models
```shell
# Prepare the environment with conda.
conda create -n t3 python=3.10
conda activate t3
# Please manually install torch==2.0.1/torchvision==0.15.2
pip install pytorch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt
# If you meet some conflicts on packages, please repair it manually.
# Download the base models from huggingface.
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
git clone https://huggingface.co/xinsir/controlnet-union-sdxl-1.0
```

### How to launch a web interface

- Run the Gradio app.
```shell
python gradio_triplet.py
# use the examples for the evaluation.
# for the complex colored cases, please adjust hyperparameters.
```

----


#### BibTeX
```
@journal{sun2024T3S2S,
      title={T$^3$-S2S: Training-free Triplet Tuning for Sketch to Scene Generation}, 
      author={Zhenhong Sun and Yifu Wang and Yonhon Ng and Yunfei Duan and Daoyi Dong and Hongdong Li and Pan Ji},
      year={2024},
      eprint={2412.13486},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.13486}, 
}
```

---

#### Acknowledgment
The demo was developed referencing this [source code](https://github.com/naver-ai/DenseDiffusion). Thanks for the inspiring work! 🙏 

