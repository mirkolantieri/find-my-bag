import os
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .u2net import utils, model

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

root = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = f"{root}/u2net/saved_models"


class U2NET():
    def __init__(self):

        # Check gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = model.U2NETP(3, 1)
        net.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'u2netp.pth'),
                            map_location=self.device))

        net.eval()
        self.net = net
        print("Pre-trained U2Net loaded.")


    def _norm_pred(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)

        return dn


    def _preprocess(self, image):
        label_3 = np.zeros(image.shape)
        label = np.zeros(label_3.shape[0:2])

        if 3 == len(label_3.shape):
            label = label_3[:, :, 0]
        elif 2 == len(label_3.shape):
            label = label_3

        if 3 == len(image.shape) and 2 == len(label.shape):
            label = label[:, :, np.newaxis]
        elif 2 == len(image.shape) and 2 == len(label.shape):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        transform = transforms.Compose([utils.RescaleT(320), utils.ToTensorLab(flag=0)])
        sample = transform({"imidx": np.array([0]), "image": image, "label": label})

        return sample


    def _predict(self, net, item):
        sample = self._preprocess(item)

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs_test = torch.cuda.FloatTensor(sample["image"].unsqueeze(0).float())
            else:
                inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

            pred = d1[:, 0, :, :]
            predict = self._norm_pred(pred)

            predict = predict.squeeze()
            predict_np = predict.cpu().detach().numpy()

            del d1, d2, d3, d4, d5, d6, d7, pred, predict, inputs_test, sample
            return predict_np


    def segment(self, img):
        output = self._predict(self.net, np.array(img))

        output_img = Image.fromarray(output * 255).convert("L")
        output_img = output_img.resize((img.size), resample=Image.BILINEAR)

        empty_img = Image.new("RGBA", (img.size), 0)
        #masked = Image.composite(img, empty_img, output_img)
        #masked.show()

        np_output = np.asarray(output_img)
        binary_mask = np.uint8((np_output > 127) * 255)
        masked_binary = Image.composite(img, empty_img, Image.fromarray(binary_mask))
        
        return masked_binary, binary_mask
        

if __name__ == "__main__":
    filename = '/41494.jpg'
    image = Image.open(filename)

    u2net = U2NET()

    new_img, binary_mask = u2net.segment(image)
    import pdb; pdb.set_trace()