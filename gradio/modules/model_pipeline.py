from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionInpaintPipeline,DiffusionPipeline
import torch

# dreamlike-photoreal-2.0
# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16)

t2i_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
i2i_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
ip_model_id = "runwayml/stable-diffusion-inpainting"

class pipeline:
    def DiffusionPipeline():
        # SDXL - Text to Image
        t2i_pipe = DiffusionPipeline.from_pretrained(
            t2i_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
            )
        t2i_pipe = t2i_pipe.to("cuda")
        return t2i_pipe

    def StableDiffusionXLImg2ImgPipeline():
        # SDXL - Image to image
        i2i_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            i2i_model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
            )
        i2i_pipe = i2i_pipe.to("cuda")
        return i2i_pipe

    def StableDiffusionInpaintPipeline():
        ip_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            ip_model_id,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        )
        ip_pipe = ip_pipe.to("cuda")
        return ip_pipe

print("Model loaded!")