import os

import torch

from datetime import datetime
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


SAVE_PATH = './'
cache_dir = "./Models/.cache"

generic_models = {}
refiners = {}

def save_img(img, username):
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    i = 0
    while os.path.exists(SAVE_PATH + "users/" + username + "/images/" + current_datetime_str + "_" + str(i) + ".jpg"):
        i += 1
    img_path = SAVE_PATH + "users/" + username + "/images/" + current_datetime_str + "_" + str(i) + ".jpg"
    img.save(img_path)
    return img_path

def save_images(images, username):
    paths = []
    for image in images:
        paths.append(save_img(image, username))

    return paths


def check_len(prompt, max_length=200):
    if len(prompt) > max_length:
        print(f"Prompt is longer than {max_length} characters.")
        last_space_index = prompt.rfind(' ', 0, max_length)

        truncated_string = prompt[:last_space_index] if last_space_index != -1 else prompt[:max_length]
        return truncated_string
    else:
        return prompt


def get_models():
    if not os.path.exists('./Models'):
        os.makedirs('./Models')

    subdirs = ['.cache', 'Generic', 'Refiner', 'UserLoras']

    for subdir in subdirs:
        subdir_path = os.path.join('./Models', subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    generic_models_dir = os.path.join('./Models', 'Generic')
    refiners_dir = os.path.join('./Models', 'Refiners')

    if os.path.exists(generic_models_dir):
        for subdir in os.listdir(generic_models_dir):
            subdir_path = os.path.join(generic_models_dir, subdir)
            if os.path.isdir(subdir_path):
                generic_models[subdir] = subdir_path

    if os.path.exists(refiners_dir):
        for subdir in os.listdir(refiners_dir):
            subdir_path = os.path.join(refiners_dir, subdir)
            if os.path.isdir(subdir_path):
                refiners[subdir] = subdir_path


# def get_first_file_path(directory):
#     # Get a list of all files in the directory
#     files = os.listdir(directory)
#
#     # Filter out directories and get the first file
#     first_file = next((file for file in files if os.path.isfile(os.path.join(directory, file))), None)
#
#     # If a file is found, return its full path
#     if first_file:
#         return os.path.join(directory, first_file)
#     else:
#         return None


def load_model_text2img(MODEL_PATH):

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, safety_checker=None
    ).to("cuda")

    return pipe

def load_model_img2img(MODEL_PATH):

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_PATH,torch_dtype=torch.float16, variant="fp16", use_safetensors=True, safety_checker=None
    ).to("cuda")

    return pipe


def get_available_generic_models():
    return list(generic_models.keys())

def get_available_refiners():
    return list(refiners.keys())

def count_images(directory):
    return len([filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))])

# def lightning(prompt, negative = None, guidance_scale = 0, refiner=False, username="test", num_images_per_prompt=1):
#     base = "stabilityai/stable-diffusion-xl-base-1.0"
#     repo = "ByteDance/SDXL-Lightning"
#     ckpt = "sdxl_lightning_8step_unet.safetensors"
#
#     # Load model.
#     unet = UNet2DConditionModel.load_config(base, subfolder="unet")
#     unet = UNet2DConditionModel.from_config(unet).to("cuda", torch.float16)
#     unet.load_state_dict(load_file(hf_hub_download(repo, ckpt, cache_dir=cache_dir), device="cuda"))
#     pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
#
#     pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
#
#
#     image = pipe(prompt=prompt, negative_prompt=negative, num_inference_steps=8, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt).images[0]
#
#
#     #images[0].save("./test/output.png")
#
#     img_path = save_img(image, username)
#     return img_path


def txt2img(prompt, negative, guidance_scale, model, refiner="none", username="test", personalized=True):

    # if model == "Lightning":
    #     return lightning(prompt, negative, guidance_scale, refiner, username)

    model_path = generic_models[model]


    prompt = check_len(prompt)
    pipe_t2i = load_model_text2img(model_path)

    generator = torch.Generator(device="cuda").manual_seed(0)

    if os.path.exists("./Models/userLoras/"+username+"/"+model) and personalized == True:

        prompt = username + "xxxxixxxi123, " + prompt
        pipe_t2i.load_lora_weights("./Models/userLoras/"+username+"/"+model, weight_name="pytorch_lora_weights.safetensors")

    if refiner != "none":
        refiner_path = refiners[refiner]
        image_iti = pipe_t2i(prompt=prompt, negative_prompt=negative, guidance_scale=guidance_scale, output_type="latent", generator=generator).images[0]

        pipe_i2i = load_model_img2img(refiner_path)

        image_out = pipe_i2i(prompt=prompt, negative_prompt=negative, image=image_iti, guidance_scale=guidance_scale, generator=generator).images[0]

    else:
        image_out = pipe_t2i(prompt=prompt, negative_prompt=negative, guidance_scale=guidance_scale, generator=generator).images[0]

    img_path = save_img(image_out, username)

    return img_path


def img2img(prompt, negative, image, guidance_scale, strength, model, refiner=False, username="test", personalized=True):

    model_path = generic_models[model]

    prompt = check_len(prompt)
    pipe_i2i = load_model_img2img(model_path)

    generator = torch.Generator(device="cuda").manual_seed(0)
    print("********************************************HERE********************************************")
    print(personalized == True)
    if os.path.exists("./Models/userLoras/"+username+"/"+model) and personalized == True:
        print("Loading personalized model")
        prompt = username + "xxxxixxxi123, " + prompt
        pipe_i2i.load_lora_weights("./Models/userLoras/"+username+"/"+model, weight_name="pytorch_lora_weights.safetensors")

    if refiner == True:
        refiner_path = refiners[refiner]
        images = pipe_i2i(prompt=prompt, negative_prompt=negative, image=image, strength=strength, guidance_scale=guidance_scale, output_type="latent", generator=generator).images

        pipe_i2i = load_model_img2img(refiner_path)

        images = pipe_i2i(prompt=prompt, negative_prompt=negative, image=images, strength=strength, guidance_scale=guidance_scale, generator=generator).images
    else:
        images = pipe_i2i(prompt=prompt, negative_prompt=negative, image=image, strength=strength, guidance_scale=guidance_scale, generator=generator).images

    img_paths = save_images(images, username)

    return img_paths



def create_rgb_image(images):
    min_size = min((img.size for img in images), key=lambda size: size[0]*size[1])
    resized_images = [img.resize(min_size, Image.Resampling.LANCZOS) for img in images]

    np_images = [np.array(img) for img in resized_images]

    new_image = np.zeros_like(np_images[0])

    for i in range(len(np_images)):
        if i % 3 == 0:  # Red component
            new_image[..., 0] += np_images[i][..., 0]
        elif i % 3 == 1:  # Green component
            new_image[..., 1] += np_images[i][..., 1]
        elif i % 3 == 2:  # Blue component
            new_image[..., 2] += np_images[i][..., 2]

    new_image[..., 0] //= (len(np_images) // 3 + (len(np_images) % 3 > 0))  # Red
    new_image[..., 1] //= (len(np_images) // 3 + (len(np_images) % 3 > 1))  # Green
    new_image[..., 2] //= (len(np_images) // 3)  # Blue

    new_image = Image.fromarray(new_image.astype('uint8'))

    return new_image


def merge_images(images):
    min_size = min((img.size for img in images), key=lambda size: size[0]*size[1])
    resized_images = [img.resize(min_size, Image.Resampling.LANCZOS) for img in images]

    np_images = [np.array(img) for img in resized_images]

    avg_image = np.mean(np_images, axis=0).astype(np.uint8)

    merged_image = Image.fromarray(avg_image)

    return merged_image

def merge_images_gray(images):
    bw_images = [img.convert('L') for img in images]

    min_size = min((img.size for img in bw_images), key=lambda size: size[0]*size[1])
    resized_images = [img.resize(min_size, Image.Resampling.LANCZOS) for img in bw_images]

    np_images = [np.array(img) for img in resized_images]

    avg_image = np.mean(np_images, axis=0).astype(np.uint8)

    merged_image = Image.fromarray(avg_image, 'L')

    return merged_image


def test():
    # image_array = []
    # directory = "./NewBaseImg"
    # # Iterate over all files in the directory
    # # for filename in os.listdir(directory):
    # #     if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more file extensions if needed
    # #         # Construct the full file path
    # #         file_path = os.path.join(directory, filename)
    # #
    # #         image = Image.open(file_path)
    # #
    # #         image_array.append(image)
    # #
    # # for image in image_array:
    # #     test_img2img("abstract nature", "", image, strength=0.8, guidance_scale=10, model="sdxl", refiner=False, num_images_per_prompt = 1)
    # image = Image.open("./NewBaseImg/IMG-20240405-WA0001.jpg")
    #
    # print(img2img("dog", "", image, 0,0.8, username="test" ,model="sdxl", refiner=False, num_images_per_prompt=2))
    #
    # #     lightning("Eye", guidance_scale=0, num_images_per_prompt=2)
    # #lightning("Cubist portrait of a dog", "blue, yellow", 0, num_images_per_prompt=1)

    # get_models()
    # print(generic_models)
    # print(refiners)

    pass

get_models()

if __name__ == "__main__":
    test()