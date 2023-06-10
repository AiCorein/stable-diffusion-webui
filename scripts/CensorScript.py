import launch

if not launch.is_installed("diffusers"):
    launch.run_pip(f"install diffusers", "diffusers")

import torch
import os
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
import numpy as np
from modules import scripts, shared
from modules.processing import process_images
import gradio as gr

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None

def pil_to_numpy(images):
    n_images = [np.array(image,dtype=float) for image in images]
    return n_images
# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(x_image, return_tensors="pt")
    _, has_nsfw_concepts = safety_checker(images=pil_to_numpy(x_image), clip_input=safety_checker_input.pixel_values)
    return has_nsfw_concepts

def get_ban_img(img: Image.Image, banimg: Image.Image):
    s, m = img.size, img.mode
    if s[0]==s[1]:
        new_banimg = banimg.resize(s, Image.NEAREST)
    elif s[0]>s[1]:
        new_banimg = Image.new(m, (s[0], s[1]), (255, 255, 255))
        new_banimg.paste(banimg.resize((s[1], s[1]), Image.NEAREST), ((s[0]-s[1])//2, 0, (s[0]-s[1])//2+s[1], s[1]))
    else:
        new_banimg = Image.new(m, (s[0], s[1]), (255, 255, 255))
        new_banimg.paste(banimg.resize((s[0], s[0]), Image.NEAREST), (0, (s[1]-s[0])//2, s[0], (s[1]-s[0])//2+s[0]))
    return new_banimg

class CensorScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.instead_img: Image = Image.open(
            os.sep.join(os.path.join(__file__).split(os.sep)[:-1]) + os.sep + 'warning.png'
        )

    def title(self):
        return "CensorScript"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        nsfw_check = gr.Checkbox(False, label="NSFW check")
        nsfw_ban = gr.Checkbox(False, label="NSFW with ban")
        return [nsfw_check,nsfw_ban]

    def run(self,p,nsfw_check,nsfw_ban):
        print("NSFW check:",nsfw_check," ban:",nsfw_ban)
        proc = process_images(p)
        if not nsfw_check:
            return proc
        #nsfw check
        has_nsfw_concepts = check_safety(proc.images)
        print("NSFW results:",has_nsfw_concepts)
        proc.extra_generation_params['nsfw'] = has_nsfw_concepts
        for index,nsfw in enumerate(has_nsfw_concepts):
            if nsfw and nsfw_ban:
                proc.images[index] = get_ban_img(proc.images[index], self.instead_img)
        return proc