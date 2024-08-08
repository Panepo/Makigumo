import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pipeline import pipe
from diffusers.utils import load_image

negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

## Parameter setting
num_steps = 20
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
  start_merge_step = 30

input_id_images = []
original_image = load_image("photos/scarletthead_woman/scarlett_0.jpg")
input_id_images.append(original_image)

def visualize_results(orig_img: Image.Image, output_img: Image.Image):
    """
    Helper function for pose estimationresults visualization

    Parameters:
       orig_img (Image.Image): original image
       output_img (Image.Image): processed image with PhotoMaker
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure
    """
    orig_img = orig_img.resize(output_img.size)
    orig_title = "Original image"
    output_title = "Output image"
    im_w, im_h = orig_img.size
    is_horizontal = im_h < im_w
    fig, axs = plt.subplots(
        2 if is_horizontal else 1,
        1 if is_horizontal else 2,
        sharex="all",
        sharey="all",
    )
    fig.suptitle(f"Prompt: '{prompt}'", fontweight="bold")
    fig.patch.set_facecolor("white")
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(output_img))
    list_axes[0].set_title(orig_title, fontsize=15)
    list_axes[1].set_title(output_title, fontsize=15)
    fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
    fig.tight_layout()
    return fig

while 1:
  print("================================================")
  prompt = input("Please say something: ")
  generator = torch.Generator("cpu").manual_seed(random.randint(1, 99))

  images = pipe(
    prompt=prompt,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=start_merge_step,
    generator=generator,
  ).images

  fig = visualize_results(original_image, images[0])
  fig.show()
