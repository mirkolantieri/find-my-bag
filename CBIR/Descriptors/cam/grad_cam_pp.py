import cv2
import numpy as np
import torch
from src.cam.base_cam import BaseCAM

class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAMPlusPlus, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor, 
                              target_category, 
                              activations, 
                              grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2*grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (2*grads_power_2 + sum_activations[:, None, None]*grads_power_3 + eps)

        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0)*aij
        weights = np.sum(weights, axis=(1, 2))
        return weights