import os
import json
from pathlib import Path
import openai
# BASE_DIR = Path().resolve()

# with open(BASE_DIR/'secrets.json') as f:
#     secrets = json.loads(f.read())
    
from dotenv import load_dotenv

# .env 파일 활성화
load_dotenv()

SECRET_KEY = os.getenv('OPENAI_KEY')

def gen(x):
  gpt_prompt = [{
      "role": "system",
      "content": ("You're an artificial intelligence chatbot with a lot of imagination."
                  "Look at the words you're presented with and imagine what you look like and describe them in detail.\n\n"
                  "Example:\n"
                  "Input: 귀여운 아기 공룡"
                  "Output: bady dinosaur, adorable, pink, spotted, short neck, four legs, two small wings"),
  }]

  gpt_prompt.append({
      "role":"user",
      "content": ("Imagine the word below and describe their appearance in English in about 20 words,"
                  f"using mainly none and adjectives, separted by commas:\n\n{x}")
  })

  gpt_response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=gpt_prompt
  )
  return gpt_response["choices"][0]["message"]["content"]


from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, DiffusionPipeline  # PM++ 2M Karras, Euler a
import torch

# model_id = "dreamlike-art/dreamlike-photoreal-2.0"
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "runwayml/stable-diffusion-v1-5"

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

# SDXL - Text to Image
t2i_pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")

# dreamlike-photoreal-2.0
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16)
t2i_pipe = t2i_pipe.to("cuda")


# SDXL - Image to image
# i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16")

# i2i_pipe = i2i_pipe.to("cuda")

# ip_model_id = "runwayml/stable-diffusion-inpainting"
# ip_pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     ip_model_id,
#     variant="fp16",
#     use_safetensors=True,
#     torch_dtype=torch.float16,
# )
# ip_pipe = ip_pipe.to("cuda")

print("Model loaded!")


# pipeline.load_lora_weights(".", weight_name="howls_moving_castle.safetensors")
import gradio as gr

def t2i_generator(prompt, negative, scheduler, num_inference_steps, width, height, cfg_scale, batch_count, seed, is_chatgpt):
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

# def i2i_generator(prompt, negative, input_img, num_inference_steps, width, height, cfg_scale, denoising_strength, batch_count, seed, scheduler, is_chatgpt):
#   prompt = gen(prompt) if is_chatgpt else prompt
#   # prompt = gen(prompt)
#   # prompt = prompt
#   i2i_pipe.scheduler = lms if scheduler == "lms" else dpm if scheduler == "dpm" else euler

#   # 고정 프롬프트 설정
#   fixed_prompt = "realistic, best quality, masterpiece, "
#   fixed_negative = "worst, bad quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly, "
#   input_prompt = fixed_prompt + prompt
#   negative_prompt = fixed_negative + negative

#   img = i2i_pipe(
#     prompt = input_prompt,
#     negative_prompt=fixed_negative + negative,
#     image=input_img,
#     num_inference_steps=num_inference_steps,
#     width=width,
#     height=height,
#     guidance_scale = cfg_scale,
#     strength=denoising_strength,
#     num_images_per_prompt=batch_count,
#     generator=torch.Generator(device="cuda").manual_seed(seed)
#   ).images[0]

#   return input_prompt, img

# def ip_generator(prompt, negative, input_img, mask_img, num_inference_steps, width, height, cfg_scale, denoising_strength, batch_count, seed, scheduler, is_chatgpt):
#   fixed_prompt = "realistic, best quality, masterpiece, "
#   fixed_negative = "worst, bad quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, ugly, "
#   input_prompt = fixed_prompt + prompt
#   negative_prompt = fixed_negative + negative

#   img = ip_pipe(
#     prompt = input_prompt,
#     negative_prompt=fixed_negative + negative,
#     image=input_img,
#     mask_image=mask_img,
#     num_inference_steps=num_inference_steps,
#     width=width,
#     height=height,
#     guidance_scale = cfg_scale,
#     strength=denoising_strength,
#     num_images_per_prompt=batch_count,
#     generator=torch.Generator(device="cuda").manual_seed(seed)
#   ).images[0]

#   return input_prompt, img



t2i_generator_tab = gr.Interface(fn=t2i_generator,
                                 inputs=["textbox",
                                          "textbox",
                                          gr.Dropdown(["lms", "dpm", "euler"], label="Sampling method", value="lms"),
                                          gr.Slider(1, 150, step=1, value=20, label="Sampling Steps", show_label=True ),
                                          gr.Slider(64, 2048, step=8, value=512, label="Img Width", show_label=True ),
                                          gr.Slider(64, 2048, step=8, value=512, label="Img Height", show_label=True),
                                          gr.Slider(1, 30, step=0.5, value=7, label="CFG Scale", show_label=True),
                                          gr.Slider(1, 10, step=1, value=1, label="Batch count", show_label=True),
                                          gr.Slider(1, 99999, step=1, value=1, label="Seed", show_label=True),
                                          "checkbox"
                                          ]
                    , outputs=["text", "image"]
                    , title = "Innople AI Studio")

# i2i_generator_tab = gr.Interface(fn=i2i_generator,
#                                  inputs=["textbox",
#                                          "textbox",
#                                           gr.Image(value=None, type="pil"),
#                                           gr.Slider(1, 150, step=1, value=20, label="Sampling Steps", show_label=True ),
#                                           gr.Slider(64, 2048, step=8, value=512, label="Img Width", show_label=True ),
#                                           gr.Slider(64, 2048, step=8, value=512, label="Img Height", show_label=True),
#                                           gr.Slider(1, 30, step=0.5, value=7, label="CFG Scale", show_label=True),
#                                           gr.Slider(0, 1, step=0.01, value=0.75, label="Denoising Strength", show_label=True ),
#                                           gr.Slider(1, 10, step=1, value=1, label="Batch count", show_label=True),
#                                           gr.Slider(1, 99999, step=1, value=1, label="Seed", show_label=True),
#                                           gr.Dropdown(["lms", "dpm", "euler"], label="Sampling method", value="lms"),
#                                           "checkbox"
#                                           ]
#                     , outputs=["text", "image"]
#                     , title = "Innople AI Studio")

# ip_generator_tab = gr.Interface(fn=ip_generator,
#                                  inputs=["textbox",
#                                          "textbox",
#                                           gr.Image(value=None, type="pil"),
#                                           gr.Image(value=None, type="pil"),
#                                           gr.Slider(1, 150, step=1, value=20, label="Sampling Steps", show_label=True ),
#                                           gr.Slider(64, 2048, step=8, value=512, label="Img Width", show_label=True ),
#                                           gr.Slider(64, 2048, step=8, value=512, label="Img Height", show_label=True),
#                                           gr.Slider(1, 30, step=0.5, value=7, label="CFG Scale", show_label=True),
#                                           gr.Slider(0, 1, step=0.01, value=0.75, label="Denoising Strength", show_label=True ),
#                                           gr.Slider(1, 10, step=1, value=1, label="Batch count", show_label=True),
#                                           gr.Slider(1, 99999, step=1, value=1, label="Seed", show_label=True),
#                                           gr.Dropdown(["lms", "dpm", "euler"], label="Sampling method", value="lms"),
#                                           "checkbox"
#                                           ]
#                     , outputs=["text", "image"]
#                     , title = "Innople AI Studio")



# demo = gr.TabbedInterface([t2i_generator_tab, i2i_generator_tab, ip_generator_tab], ["Text to Image", "Image to Image", "Inpainting"])
# demo = gr.TabbedInterface([t2i_generator_tab, i2i_generator_tab], ["Text to Image", "Image to Image"])
demo = t2i_generator_tab



if __name__ == "__main__":
    demo.launch(debug=True, share=False, server_port=22)
