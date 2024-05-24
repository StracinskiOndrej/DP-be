import shutil
import subprocess
import threading
import random

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers.utils import load_image
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import argparse



def load_config(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

config = load_config("config.json")

BLIP_PATH = config["blip_path"]
v1_5_models = config["v1_5_models"]
xl_models = config["xl_models"]


@torch.no_grad()
def generate_caption(images, capton_generator, cpation_processsor, user):
    text = user+"xxxxixxxi123, "

    inputs = cpation_processsor(images, text, return_tensors="pt").to(device="cuda", dtype=capton_generator.dtype)
    capton_generator.to("cuda")
    outputs = capton_generator.generate(**inputs, max_new_tokens=128)

    capton_generator.to("cpu")
    caption = cpation_processsor.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption


def delete_files_in_dir(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def generate_captions(img, user):
    processor = BlipProcessor.from_pretrained(BLIP_PATH)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH)

    return generate_caption(img, model, processor, user)


def get_image_names(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg')))


def read_files_in_order(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    files.sort()

    contents = []
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            contents.append(f.read())

    return contents


def create_metadata(images_dir, captions_dir):

    image_filenames = get_image_names(images_dir)
    captions = read_files_in_order(captions_dir)

    assert len(image_filenames) == len(captions), "Mismatch between number of filenames and captions"

    with open(images_dir + '/metadata.jsonl', 'w') as f:
        for filename, caption in zip(image_filenames, captions):
            data = {
                "file_name": filename,
                "image": "./" + images_dir + "/" +filename,
                "text": caption
            }

            f.write(json.dumps(data) + '\n')


def load_all_images(directory):
    image_files = sorted(f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg')))
    images = {file: load_image(os.path.join(directory, file)) for file in image_files}
    return images


def save_strings_to_files(captions_dict, directory):
    for image_name, caption in captions_dict.items():
        file_name = os.path.splitext(image_name)[0] + '.txt'
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'w') as f:
            f.write(caption)


def create_captions(img_dir, user, caption_dir):
    delete_files_in_dir(caption_dir)
    images = load_all_images(img_dir)
    captions = {}

    for image_name, image in tqdm(images.items(), desc="Generating captions"):
        captions[image_name] = generate_captions(image, user)

    save_strings_to_files(captions, caption_dir)


def read_output(pipe):
    for line in iter(pipe.readline, ''):
        print(line, end='')


def get_models(dir_path):
    items = [item for item in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, item))]
    if "stable-diffusion-v1-5" in items:
        items.remove("stable-diffusion-v1-5")
        items.insert(0, "stable-diffusion-v1-5")
    return items


def train_for_all_models(user, img_dir, model_path):

    models = get_models(model_path)

    for model in models:
        if os.path.basename(model) in v1_5_models or os.path.basename(model) in xl_models:
            if os.path.basename(model) in v1_5_models:
                script = "train_text_to_image_lora.py"
            else:
                script = "train_text_to_image_lora_sdxl.py"

            delete_files_in_dir("./Models/UserLoras/"+user+"/")
            command = [
                'accelerate', 'launch', '--mixed_precision=bf16', script,
                '--pretrained_model_name_or_path=./Models/Generic/'+ os.path.basename(model),
                '--train_data_dir=./'+img_dir, '--random_flip',
                '--output_dir=./Models/UserLoras/'+user+"/" + os.path.basename(model), '--image_column=image', '--caption_column=text',
                '--max_train_steps=1500', '--train_batch_size=1', '--learning_rate=1e-04'
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, text=True)

            # Create and start threads to read stdout and stderr
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout,))
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr,))

            stdout_thread.start()
            stderr_thread.start()

            process.wait()
            stdout_thread.join()
            stderr_thread.join()


def copy_images(source_dir, destination_dir, num_latest=80, num_random=20):
    delete_files_in_dir(destination_dir)
    image_files = sorted((os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.jpg')),
                         key=lambda x: os.path.getmtime(x), reverse=True)

    total_images = len(image_files)

    if total_images < 100:
        for img in image_files:
            shutil.copy(img, destination_dir)
    else:
        for i in range(min(total_images, num_latest)):
            shutil.copy(image_files[i], destination_dir)

        remaining_images = [img for img in image_files[num_latest:]]

        random_images = random.sample(remaining_images, min(num_random, len(remaining_images)))
        for img in random_images:
            shutil.copy(img, destination_dir)


def train_models(username):
    gallery_dir = "./users/"+username+"/gallery"
    train_dir = "./users/"+username+"/train/images"
    cap_dir = "./users/"+username+"/train/captions"

    copy_images(gallery_dir, train_dir)

    create_captions(train_dir, username, cap_dir)

    create_metadata(train_dir, cap_dir)

    train_for_all_models(username, train_dir, "./Models/Generic")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default="test", help="The username")
    args = parser.parse_args()

    train_models(args.username)




    #
    #create_captions(img_dir, user, caption_dir)
    #create_metadata(img_dir, caption_dir)


