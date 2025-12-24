import random
import os

import numpy as np
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from src.data.transforms import ImageTransform
from src.data.data_utils import add_special_tokens
from src.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from src.modeling.qwen2 import Qwen2Tokenizer
from src.modeling.autoencoder import load_ae
from src.inferencer import InterleaveInferencer

def load_model(model_path):
    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model      = SiglipVisionModel(vit_config)
        model          = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Model Loading and Multi GPU Infernece Preparing
    max_mem_per_gpu = "40GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    print(device_map)

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    # Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
    )

    model = model.eval()
    return model, vae_model

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gen(inferencer, prompt):
    inference_hyper=dict(
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )
    output_dict = inferencer(text=prompt, **inference_hyper)
    img = output_dict['image']
    return img

def think_gen(inferencer, prompt):
    inference_hyper=dict(
        max_think_token_n=1000,
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )

    output_dict = inferencer(text=prompt, think=True, **inference_hyper)
    print(output_dict['text'])
    img = output_dict['image']
    return img

def interleaved_gen(inferencer, prompt):
    print("-" * 20)
    print("Interleave Inference")
    print("-" * 20)
    num_interleaves = 3  # Specify the number of interleave iterations
    current_image = None
    
    # Hyperparameters for generation
    inference_hyper_interleave = dict(
        do_sample=False,
        cfg_text_scale=4.0,
        cfg_img_scale=1.0,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
    )

    for i in range(num_interleaves):
        print(f"\\nInterleave Step {i+1}/{num_interleaves}")

        output_dict = inferencer(image=current_image, text=prompt, **inference_hyper_interleave)

        current_image = output_dict['image']

    return output_dict['image']


if __name__ == "__main__":    
    set_random_seed(0)
    model_path = "ckpts/BAGEL-7B-MoT"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
    model, vae_model = load_model(model_path)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    # inference preparing
    inferencer = InterleaveInferencer(
        model=model, 
        vae_model=vae_model, 
        tokenizer=tokenizer, 
        vae_transform=vae_transform, 
        vit_transform=vit_transform, 
        new_token_ids=new_token_ids
    )

    keys = ['glass', 'rabbit', 'duck', 'car', 'monkey']
    prompts = {
        'glass': "1 wine glass, 2 bottles of wine, 3 cans of beer",
        'rabbit': "A white rabbit in blue jogging clothes, a turtle wearing a red tank top",
        'duck': "Duck is playing with a mobile phone",
        'car': "A car made of small cars",
        'monkey': "A monkey making latte art"
    }

    for k in keys:
        prompt = prompts[k]
        print("prompt:", prompt)
        print("Generating...")
        
        img = gen(inferencer, prompt)
        img_save_path = f"results/{k}.png"
        img.save(img_save_path)

        img = think_gen(inferencer, prompt)
        img_save_path = f"results/think_{k}.png"
        img.save(img_save_path)

        img = interleaved_gen(inferencer, prompt)
        img_save_path = f"results/interleaved_{k}.png"
        img.save(img_save_path)
