import torch
import os, sys
from PIL import Image
import types
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256_inpainting import (
    StableDiffusionXLInpaintPipeline,
)
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer


def infer(
    image_path="test.png",
    mask_path="tm.png",
    prompt="change the shirt color to red",
    negative_prompt="",
    guidance_scale=6.0,
    num_inference_steps=25,
    strength=0.999,
    seed=603,
    return_image=False
):
    # Hard-coded image paths and prompt will be overridden by function parameters
    ckpt_dir = "Kolors/weights/Kolors-Inpainting"
    text_encoder = ChatGLMModel.from_pretrained(
        f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
    ).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")

    # Fix the tokenizer padding function to handle overflow error
    original_pad = tokenizer._pad

    def patched_pad(*args, **kwargs):
        # Remove padding_side if present
        if "padding_side" in kwargs:
            del kwargs["padding_side"]
        
        try:
            return original_pad(*args, **kwargs)
        except OverflowError:
            # Handle the overflow case with a simpler approach
            encoded_inputs = args[0]
            max_length = kwargs.get("max_length", 256)
            
            # Just truncate rather than pad if we encounter overflow
            if "input_ids" in encoded_inputs:
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"][-max_length:]
            if "attention_mask" in encoded_inputs:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"][-max_length:]
                
            return encoded_inputs

    tokenizer._pad = patched_pad

    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        f"{ckpt_dir}/unet", revision=None
    ).half()

    pipe = StableDiffusionXLInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    pipe.to("cuda")
    pipe.enable_attention_slicing()
    
    # Create a completely new patched pipeline to avoid dimension issues
    original_call = pipe.__call__
    
    def patched_call(
        self,
        prompt=None,
        image=None,
        mask_image=None,
        height=None,
        width=None,
        strength=0.999,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        **kwargs
    ):
        device = self._execution_device
        
        # Process height and width
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # Process images
        if isinstance(image, (list, tuple)):
            image = image[0]
            
        if isinstance(mask_image, (list, tuple)):
            mask_image = mask_image[0]
            
        image = self.image_processor.preprocess(image, height=height, width=width)
        mask_image = self.mask_processor.preprocess(mask_image, height=height, width=width)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # VAE encode
        latents = self.vae.encode(image.to(device=device, dtype=self.vae.dtype)).latent_dist.sample()
        latents = self.vae.config.scaling_factor * latents
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, timesteps[:1])
        
        # Process mask
        mask = torch.from_numpy(mask_image).to(device=device, dtype=torch.float32) / 255.0
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=(height // 8, width // 8)
        )
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)
        
        # Process prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # Create text embeddings
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_embeds = self.text_encoder(text_inputs.input_ids)[0]
        
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
                
            uncond_tokens = self.tokenizer(negative_prompt, return_tensors="pt").to(device)
            negative_prompt_embeds = self.text_encoder(uncond_tokens.input_ids)[0]
            
            # Concatenate for classifier free guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # Set initial latents
        init_latents = latents
        
        # Loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict the noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"mask": mask},
            ).sample
            
            # Apply guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # Compute the previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Apply mask (keep original content in the unmasked areas)
            init_latents_proper = self.scheduler.add_noise(init_latents, noise, t)
            latents = (1 - mask) * init_latents_proper + mask * latents
        
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        images = self.image_processor.postprocess(image, output_type="pil")
        
        return type(self).ImagePipelineOutput(images=images)
        
    # Replace the call method
    pipe.__call__ = types.MethodType(patched_call, pipe)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    basename = image_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    try:
        print(f"Starting inpainting process...")
        print(f"Loading images...")
        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")

        print(f"Processing image: {image_path}")
        print(f"With mask: {mask_path}")
        print(f"Using prompt: '{prompt}'")

        print(f"Running inference...")
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=1024,
            width=768,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            strength=strength,
        ).images[0]

        if return_image:
            return result
        else:
            # Create output directory if it doesn't exist
            output_dir = f"{root_dir}/scripts/outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = f"{output_dir}/sample_inpainting_{basename}.jpg"
            result.save(output_path)
            print(f"Output saved to: {output_path}")
            return output_path

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    infer()
