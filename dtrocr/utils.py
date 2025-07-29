import torch

from PIL import Image

from dataclasses import asdict
from dtrocr.data import DTrOCRProcessorOutput


def resize_and_pad(image, target_size, alignment='left', color=(255, 255, 255)):
    target_width, target_height = target_size
    original_width, original_height = image.size

    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    padded_image = Image.new("RGB", target_size, color)

    paste_y = (target_height - new_height) // 2

    if alignment == 'left':
        paste_x = 0
    elif alignment == 'right':
        paste_x = target_width - new_width
    else:
        paste_x = (target_width - new_width) // 2

    padded_image.paste(resized_image, (paste_x, paste_y))

    return padded_image


def send_inputs_to_device(dictionary, device):
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in dictionary.items()
    }


def send_processor_output_to_device(inputs, device):
    inputs = send_inputs_to_device(asdict(inputs), device)
    return DTrOCRProcessorOutput(**inputs)
