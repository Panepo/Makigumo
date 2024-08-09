import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_results(orig_img: Image.Image, output_img: Image.Image, prompt: str):
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
