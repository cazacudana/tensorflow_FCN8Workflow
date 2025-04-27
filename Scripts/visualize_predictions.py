import os
from PIL import Image
import sys
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from Run_settings import modelparams

# Determine project root (parent of this Scripts folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(script_dir)
#IMAGE_DIR =  cwd + "/sampleRun/images/
IMAGE_DIR =   cwd + "/sampleRun/images/trainimagesold/"

#PRED_DIR  = cwd + "/sampleRun/results/"
PRED_DIR  = cwd + "/sampleRun/images/" + "GTinfo_old/"
pred_dir   = PRED_DIR  # .mat masks are directly in this folder

# Create folder to save unlabeled prediction visualizations
output_dir = os.path.join(pred_dir, "PREDICT_UNLABELED_RESULTS")
os.makedirs(output_dir, exist_ok=True)

for fn in sorted(os.listdir(pred_dir)):
    if not fn.lower().endswith('.mat'):
        continue
    # Load predicted mask from .mat
    mat = scipy.io.loadmat(os.path.join(pred_dir, fn))
    # Extract the first non-metadata variable
    for key,val in mat.items():
        if not key.startswith('__'):
            image_data = val
            break
    # Convert one-hot or single-channel to label map
    if image_data.ndim == 3 and image_data.shape[2] > 1:
        label_map = np.argmax(image_data, axis=2)
    else:
        label_map = image_data
    # Create a PIL mask image (mode 'L')
    mask = Image.fromarray(label_map.astype(np.uint8), mode='L')

    # Determine corresponding original image filename (.png)
    core = fn.split('.')[0]
    # Strip off any coordinate suffix (e.g., '_rowmin...')
    if '_rowmin' in core:
        core = core[:core.index('_rowmin')]
    img_fname = core + '.png'
    img_path = os.path.join(IMAGE_DIR, img_fname)
    if not os.path.exists(img_path):
        print(f"Original image not found for mask: {fn}")
        continue
    img = Image.open(img_path).convert("RGBA")

    # Create a class‐colored overlay
    # Define RGBA colors per class index
    # Adjust these RGBA tuples to your colormap if needed
    class_colors = {
        1: (255,   0,   0, 100),  # class 1: tumor (red)
        2: (  0,   0, 255, 100),  # class 2: stroma (blue)
        3: (255, 255,   0, 100),  # class 3: necrosis (yellow)
        4: (  0, 255,   0, 100),  # class 4: inflammatory (green)
        # skip 0 (background/exclude)
    }
    # Start from the original image as RGBA
    overlay = img.copy()
    # Convert label_map to numpy array once
    label_arr = np.array(label_map, dtype=np.uint8)
    for cls_val, rgba in class_colors.items():
        # Mask for this class
        cls_mask = (label_arr == cls_val).astype(np.uint8) * 255
        if cls_mask.max() == 0:
            continue
        mask_img = Image.fromarray(cls_mask, mode='L')
        # Create solid color layer
        color_layer = Image.new("RGBA", img.size, rgba)
        # Apply this class mask as alpha
        color_layer.putalpha(mask_img)
        # Composite over the current overlay
        overlay = Image.alpha_composite(overlay, color_layer)
    
    fig, ax = plt.subplots(1,2, figsize=(8,4))
    ax[0].imshow(Image.open(img_path))
    ax[0].set_title("Original")
    ax[1].imshow(overlay)
    ax[1].set_title("Overlay Pred")
    ax[0].axis("off"); ax[1].axis("off")
    # Save the comparison figure
    out_fname = f"comparison_{core}.png"
    fig.savefig(os.path.join(output_dir, out_fname), bbox_inches='tight')
    plt.close(fig)