
male_prompt = "closeup portrait photo of a man img "
female_prompt = "closeup portrait photo of a woman img "

male_image_path = "photos/newton_man/newton_0.jpg"
female_image_path = "photos/scarletthead_woman/scarlett_0.jpg"

def prompt_converter(prompt: str):
  if prompt.find("man"):
    output = male_prompt + ' ' + prompt
    image = male_image_path
  elif prompt.find("woman"):
    output = female_prompt + ' ' + prompt
    image = female_image_path
  else:
    output = male_prompt + ' ' + prompt
    image = male_image_path

  return output, image
