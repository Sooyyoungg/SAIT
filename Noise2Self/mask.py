#from https://github.com/czbiohub/noise2self/blob/master/mask.py

import numpy as np
import torch

class Masker:
    """Object for masking and demasking"""

    def __init__(self, width=3, mode="zero", infer_single_pass=False, include_mask_as_input=False):
        self.grid_size = width # mask <- width x width
        self.n_masks = width ** 2 # number of masks

        self.mode = mode
        self.infer_single_pass = infer_single_pass
        self.include_mask_as_input = include_mask_as_input

    def mask(self, image, i):

        phasex = i % self.grid_size
        phasey = (i // self.grid_size) % self.grid_size
        if len(image.shape) > 2:
            mask = pixel_grid_mask(image[0, 0].shape, self.grid_size, phasex, phasey)
        else:
            mask = pixel_grid_mask(image.shape, self.grid_size, phasex, phasey)
        mask = mask.to(image.device)

        mask_inv = torch.ones(mask.shape).to(image.device) - mask # mask_inv = 1 - mask

        if self.mode == "interpolate":
            masked = interpolate_mask(image, mask, mask_inv)
        elif self.mode == "zero":
            masked = image * mask_inv
        else:
            raise NotImplementedError

        if self.include_mask_as_input:
            net_input = torch.cat((masked, mask.repeat(image.shape[0], 1, 1, 1)), dim=1)
        else:
            net_input = masked

        return net_input, mask

    def __len__(self):
        return self.n_masks

    def infer_full_image(self, image, model):

        if self.infer_single_pass:
            if self.include_mask_as_input:
                net_input = torch.cat(
                    (image, torch.zeros(image[:,0:1].shape).to(image.device)), dim=1
                )
            else:
                net_input = image
            net_output = model(net_input)
            return net_output

        else:
            net_input, mask = self.mask(image, 0)
            net_output = model(net_input)

            acc_tensor = torch.zeros(net_output.shape).to(image.device)

            for i in range(self.n_masks):
                net_input, mask = self.mask(image, i)
                net_output = model(net_input)
                acc_tensor = acc_tensor + (net_output * mask).to(image.device)

            return acc_tensor

def pixel_grid_mask(shape, patch_size, phase_x, phase_y):
    A = torch.zeros(shape[-2:])
    for i in range(shape[-2]):
        for j in range(shape[-1]):
            if i % patch_size == phase_x and j % patch_size == phase_y:
                A[i, j] = 1
    return torch.Tensor(A)

def interpolate_mask(tensor, mask, mask_inv):
    device = tensor.device

    mask = mask.to(device)

    kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)]) ## kernel model
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(tensor, kernel, stride=1, padding=1)

    return filtered_tensor * mask + tensor * mask_inv