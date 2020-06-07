# Data handling
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import resize


def img2tensor_func():
    return transforms.Compose([transforms.ToTensor()])


def image_loader(image_name, imsize, device='cpu'):
    image = Image.open(image_name)
    loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])
    image = loader(image)
    image = image.unsqueeze(0)  # gets right dimension at particular size
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clone the tensor, prevent pass by ref
    image = torch.squeeze(image)  # index 0 is the batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if not title:
        plt.title(title)
    plt.pause(10)  # pausing to update the plots


def extract_masks(segment):
    """
    Extracts the segmentation masks from the segmentated image.
    Allowed colors are:
        blue, green, black, white, red,
        yellow, grey, light_blue, purple.
    """
    extracted_colors = []

    # BLUE
    mask_r = segment[..., 0] < 0.1
    mask_g = segment[..., 1] < 0.1
    mask_b = segment[..., 2] > 0.9
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # GREEN
    mask_r = segment[..., 0] < 0.1
    mask_g = segment[..., 1] > 0.9
    mask_b = segment[..., 2] < 0.1
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # BLACK
    mask_r = segment[..., 0] < 0.1
    mask_g = segment[..., 1] < 0.1
    mask_b = segment[..., 2] < 0.1
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # WHITE
    mask_r = segment[..., 0] > 0.9
    mask_g = segment[..., 1] > 0.9
    mask_b = segment[..., 2] > 0.9
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # RED
    mask_r = segment[..., 0] > 0.9
    mask_g = segment[..., 1] < 0.1
    mask_b = segment[..., 2] < 0.1
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # YELLOW
    mask_r = segment[..., 0] > 0.9
    mask_g = segment[..., 1] > 0.9
    mask_b = segment[..., 2] < 0.1
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # GREY
    mask_r = (segment[..., 0] > 0.4) & (segment[..., 0] < 0.6)
    mask_g = (segment[..., 1] > 0.4) & (segment[..., 1] < 0.6)
    mask_b = (segment[..., 2] > 0.4) & (segment[..., 2] < 0.6)
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # LIGHT_BLUE
    mask_r = segment[..., 0] < 0.1
    mask_g = segment[..., 1] > 0.9
    mask_b = segment[..., 2] > 0.9
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    # PURPLE
    mask_r = segment[..., 0] > 0.9
    mask_g = segment[..., 1] < 0.1
    mask_b = segment[..., 2] > 0.9
    mask = mask_r & mask_g & mask_b
    extracted_colors.append(mask)

    return extracted_colors


def get_all_masks(path):
    """
    Returns the segmentation masks from the segmentated image.
    """
    image = Image.open(path)
    np_image = np.array(image, dtype=np.float) / 255
    return extract_masks(np_image)


def is_nonzero(mask, thrs=0.01):
    """
    Checks segmentation mask is dense.
    """
    return np.sum(mask) / mask.size > thrs


def get_masks(path_style, path_content):
    """
    Returns the meaningful segmentation masks.
    Avoides "orphan semantic labels" problem.
    """
    masks_style = get_all_masks(path_style)
    masks_content = get_all_masks(path_content)

    non_zero_masks = [
        is_nonzero(mask_c) and is_nonzero(mask_s)
        for mask_c, mask_s in zip(masks_content, masks_style)
    ]

    masks_style = [mask for mask, cond in zip(masks_style, non_zero_masks) if cond]
    masks_content = [mask for mask, cond in zip(masks_content, non_zero_masks) if cond]

    return masks_style, masks_content


def resize_masks(masks_style, masks_content, size):
    """
    Resizes masks to given size.
    """
    resize_mask = lambda mask: resize(mask, (size, size))

    masks_style = [resize_mask(mask) for mask in masks_style]
    masks_content = [resize_mask(mask) for mask in masks_content]

    return masks_style, masks_content


def masks_to_tensor(masks_style, masks_content):
    """
    Transforms masks to torch.Tensor from np.array.
    """
    tensor = img2tensor_func()
    masks_style = [tensor(mask) for mask in masks_style]
    masks_content = [tensor(mask) for mask in masks_content]

    return masks_style, masks_content


def masks_loader(path_style, path_content, size):
    """
    Loads masks.
    """
    style_masks, content_masks = get_masks(path_style, path_content)
    style_masks, content_masks = resize_masks(style_masks, content_masks, size)
    style_masks, content_masks = masks_to_tensor(style_masks, content_masks)

    return style_masks, content_masks
