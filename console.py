import torch
import random
from pipeline import pipe
from diffusers.utils import load_image
from figure import visualize_results
from prompt import prompt_converter

negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

## Parameter setting
num_steps = 20
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
  start_merge_step = 30

while 1:
  print("================================================")
  text = input("Please say something: ")
  if (text == 'exit'): break
  else:
    prompt, imagePath = prompt_converter(text)

    input_id_images = []
    original_image = load_image(imagePath)
    input_id_images.append(original_image)

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

    fig = visualize_results(original_image, images[0], prompt)
    fig.show()
