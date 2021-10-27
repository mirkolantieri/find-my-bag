def deprocess_image(img):
    """
    `deprocess_image` : converts the float array to an image frame  

    args:
        img: the float array
    """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

    
def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    """ 
    `preprocess_image`: preprocess an image frame by converting it to a torch tensor

    args:
        img: the image provided as nd.array
        mean=None: the list of mean values to be used for processing
        std: standard deviation values to be used for the processing
    """
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    return preprocessing(img.copy()).unsqueeze(0)

    
    
def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    `show_cam_on_image` : returns the class activation maps of an image
    
    args: 
        img: the image provided as an nd.array
        mask: the mask filter as an nd.array
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

    
import os
import torch
import argparse
import cv2
import numpy as np
from torchvision import models
from cam.ablation_cam import AblationCAM
from cam.grad_cam import GradCAM
from cam.grad_cam_pp import GradCAMPlusPlus
from cam.guided_backprop import GuidedBackpropReLUModel
from cam.score_cam import ScoreCAM
from cam.xgrad_cam import XGradCAM
from torchvision.transforms import Compose, Normalize, Resize, ToTensor



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image', type=str, default=False, 
                        help='Input image path')
    parser.add_argument('--model', type=str, default=False, help='path of the stored model')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python activation.py --image <path_to_image> --path <path_pretrained_model> --case <no_case> --plane <axial or sagittal or coronal>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    # Select one the available method from the parsed arguments

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    #pretrained = torch.load(args.model) # utilizzare nel caso si dispongo i pesi di una rete preaddestrata
    model = models.resnet18(pretrained=True)
    #model.load_state_dict(pretrained, strict=False)
    
    """
    Choose the target layer you want to compute the visualization for.
    Usually this will be the last convolutional layer in the model.
    Some common choices can be:
        Resnet18 and 50: model.layer4[-1]
        VGG, densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        You can print the model to help chose the layer
        print(model) 
    """

    target_layer = model.layer4[-1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=args.cuda)

    rgb_img = cv2.imread(args.image, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(
        model=model, use_cuda=args.cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge(
        [grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'./images/activations/image_{args.method}.jpg', cam_image)