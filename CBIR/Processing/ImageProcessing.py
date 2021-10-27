import cv2
from skimage import feature
import numpy as np
from PIL import Image

from . import PreProcessing
from . import FileUtils as utils
from . import groupClassifier as grpc

class ImageProcessing:
    """ Entrypoint class for image processing.
        If debug is True, store all intermediate images in file and send to Telegram chat debug information.
    """

    def __init__(self, bot=None, chat_id=None, img_path=None, img=None, grpChecker=None, debug=False):

        self.bot = bot
        self.chat_id = chat_id
        self.debug = debug

        if img_path is not None:
            self.input_img_path = img_path                                    # input image path
            self.input_img = cv2.imread(self.input_img_path)                  # load image
        elif img is not None:
            self.input_img = img
            self.input_img_path = None

        self.bw_img = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)    # convert image to bw
        if self.debug and self.input_img_path:
            utils.save_image(self.bw_img, self.input_img_path, suffix='_bw')

        self.preprocessing = PreProcessing.PreProcessing(
            bot, chat_id, self.input_img, self.bw_img, self.debug)

        self.group_checker = grpChecker
        self.enhanced_img = None
        self.enanched = False

    def check_quality(self):
        return self.preprocessing.check_image_quality()


    def enhance(self):
        self.enhanced_img, self.enanched = self.preprocessing.enhance_image()
        return self.enhanced_img, self.enanched#, utils.save_image(self.enhanced_img, self.input_img_path, "_enhanced")


    def check_object_presence(self):
        if self.enanched:
            img = Image.fromarray(cv2.cvtColor(self.enhanced_img, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB))
        return self.group_checker._predict_content(img)


    def remove_bg(self, img=None):
        """ Handcrafted method to performs the removal of the background from an image """
        if img is not None:
            input = img
        else:
            input = self.input_img

        # HSV conversion to take intensity channel
        input = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2HSV)[:,:,2]

        # Negative
        neg = cv2.bitwise_not(input)
        sobelxy = cv2.Sobel(src=neg, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        blur = cv2.GaussianBlur(sobelxy, (5,5),0)

        _, thresh = cv2.threshold(np.uint8(blur), 0, 255, cv2.THRESH_OTSU)

        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 4)

        blur_opening = cv2.GaussianBlur(opening, (5,5),4)

        mask = np.zeros(self.input_img.shape[:2], np.uint8)
        for x in [0,blur_opening.shape[0]-1]:
            for y in [0,blur_opening.shape[1]-1]:
                if thresh[x,y] == 0:
                    cv2.floodFill(blur_opening, None, (y,x), 127)

        mask[blur_opening!=127] = 255 

        ## Old method
        # blur and threshold
        #_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        #kernel = np.ones((3,3),np.uint8)
        #opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 4)
        # find image contours
        #contours = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # sort the contours
        #contours = sorted(contours, key=cv2.contourArea)
        # create the mask
        #mask = np.zeros(self.input_img.shape[:2], np.uint8)
        #max_area = (mask.shape[0]*mask.shape[1])/4
        #check_exist = False
        #for i in contours:
            #if cv2.contourArea(i) > max_area:
            #    check_exist = True
            #cv2.drawContours(mask, [i], -1, 255, -1)
            #    cv2.fillPoly(mask, [i], (255))
            #else:
            #    cv2.fillPoly(mask, [i], (0))
            
        #if not check_exist:
        #    cv2.fillPoly(mask, contours[-1], (255))
       
        #new_img = cv2.bitwise_and(input, input, mask=mask)

        old_input = Image.fromarray(cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB))

        empty_img = Image.new("RGBA", (old_input.size), 0)
        masked = Image.composite(old_input, empty_img, Image.fromarray(mask).convert("L"))

        if self.debug and self.input_img_path:
            masked.save(self.input_img_path + "_no_bg")

        return masked, mask


    def color_transfer(self, image):
        """ Method perform the color transfer by shifting the hue gradient """

        # load the images
        img = self.input_img
        img2 = cv2.imread(image, cv2.COLOR_BGR2LAB)

        # extract the mean and std of the two images
        s_mean, s_std = cv2.meanStdDev(img)
        t_mean, t_std = cv2.meanStdDev(img2)

        s_mean = np.hstack(np.around(s_mean, 2))
        t_mean = np.hstack(np.around(t_mean, 2))

        # Split the channels of the image
        h, w, c = cv2.split(img)

        for i in range(0, h):
            for j in range(0, w):
                for k in range(0, c):
                    x = img[i, j, k]
                    x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                    x = round(x)
                    x = 0 if x < 0 else x
                    x = 255 if x > 255 else x
                    img[i, j, k] = x

        # return the new image
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        if self.debug:
            utils.save_image(img, self.input_img_path, "_col_transf")
        return img