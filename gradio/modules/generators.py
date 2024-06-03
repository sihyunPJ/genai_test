##############################################################################
# 시스템 환경변수 추가
# import os
# os.getcwd()
# os.chdir('./gradio/modules')
# import sys
# sys.path.append('C:\\Users\\kim_sihyeon02\\test\\gradio\\modules')
##############################################################################

# pipeline.load_lora_weights(".", weight_name="howls_moving_castle.safetensors")
import gradio as gr
from modules.model_pipeline import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers import LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
# from modules.scheduler import lms, dpm, euler
from modules.prompt_engineering import gen
import torch

# lms = lms()
# dpm = dpm()
# euler = euler()

lms = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )

dpm = DPMSolverMultistepScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

euler = EulerAncestralDiscreteScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear"
)

# t2i_pipe = DiffusionPipeline()
# i2i_pipe = StableDiffusionXLImg2ImgPipeline()
# ip_pipe = StableDiffusionInpaintPipeline()

def t2i_generator(prompt, negative, scheduler, num_inference_steps, width, height, cfg_scale, batch_count, seed, is_chatgpt):
  t2i_pipe = DiffusionPipeline()
  prompt = gen(prompt) if is_chatgpt else prompt
  # prompt = gen(prompt)
  # prompt = prompt
  t2i_pipe.scheduler = lms if scheduler == "lms" else dpm if scheduler == "dpm" else euler

  # 고정 프롬프트 설정
  fixed_prompt = "realistic, best quality, masterpiece, "
  fixed_negative = "worst, bad quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly, "
  input_prompt = fixed_prompt + prompt

  img = t2i_pipe(
    prompt = input_prompt,
    negative_prompt=fixed_negative + negative,
    num_inference_steps=num_inference_steps,
    guidance_scale = cfg_scale,
    width=width,
    height=height,
    generator=torch.Generator(device="cuda").manual_seed(seed),
    num_images_per_prompt=batch_count
  ).images[0]

  return input_prompt, img

def i2i_generator(prompt, negative, input_img, num_inference_steps, width, height, cfg_scale, denoising_strength, batch_count, seed, scheduler, is_chatgpt):
  i2i_pipe = StableDiffusionXLImg2ImgPipeline()
  prompt = gen(prompt) if is_chatgpt else prompt
  # prompt = gen(prompt)
  # prompt = prompt
  i2i_pipe.scheduler = lms if scheduler == "lms" else dpm if scheduler == "dpm" else euler

  # 고정 프롬프트 설정
  fixed_prompt = "realistic, best quality, masterpiece, "
  fixed_negative = "worst, bad quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly, "
  input_prompt = fixed_prompt + prompt
  negative_prompt = fixed_negative + negative

  img = i2i_pipe(
    prompt = input_prompt,
    negative_prompt=fixed_negative + negative,
    image=input_img,
    num_inference_steps=num_inference_steps,
    width=width,
    height=height,
    guidance_scale = cfg_scale,
    strength=denoising_strength,
    num_images_per_prompt=batch_count,
    generator=torch.Generator(device="cuda").manual_seed(seed)
  ).images[0]

  return input_prompt, img

def ip_generator(prompt, negative, input_img, mask_img, num_inference_steps, width, height, cfg_scale, denoising_strength, batch_count, seed, scheduler, is_chatgpt):
  ip_pipe = StableDiffusionInpaintPipeline()
  fixed_prompt = "realistic, best quality, masterpiece, "
  fixed_negative = "worst, bad quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly, "
  input_prompt = fixed_prompt + prompt
  negative_prompt = fixed_negative + negative

  img = ip_pipe(
    prompt = input_prompt,
    negative_prompt=fixed_negative + negative,
    image=input_img,
    mask_image=mask_img,
    num_inference_steps=num_inference_steps,
    width=width,
    height=height,
    guidance_scale = cfg_scale,
    strength=denoising_strength,
    num_images_per_prompt=batch_count,
    generator=torch.Generator(device="cuda").manual_seed(seed)
  ).images[0]

  return input_prompt, img