import cv2
import numpy as np
from skimage.exposure import is_low_contrast
from bisect import bisect_left, insort

STD_THRESHOLD = 20

VARIANCE_OF_LAPLACIAN_THRESHOLD_LOW = 80
VARIANCE_OF_LAPLACIAN_THRESHOLD_HIGH = 120

SAP_WINDOW = 5
SAP_NOISE_THRESHOLD = 0.02 # 2% of salt and pepper

CONTRAST_THRESHOLD = 0.15

class PreProcessing:
    def __init__(self, bot, chat_id, img, bw_img, debug):
        self.bot = bot
        self.chat_id = chat_id
        self.img = img          # colored input image
        self.bw_img = bw_img    # bw input image

        self.image_ok = True    # state variable that keeps track of image quality
        self.msg_problem = None
        self.variance_of_laplacian = None   # value to check in enhancement
        self.is_noisy = None
        self.denoised = None
        self.low_contrast = None
        self.is_empty = None
        self.is_blur = None

        self.debug = debug

    def reset_attributes(self):
        self.image_ok = True
        self.msg_problem = None
        self.variance_of_laplacian = None
        self.is_noisy = None
        self.denoised = None
        self.low_contrast = None
        self.is_empty = None
        self.is_blur = None

    #### control functions ####
    def check_image_quality(self):
        self.reset_attributes()

        self.check_uniform()
        #import pdb; pdb.set_trace()
        
        if not self.is_empty:
            self.check_blur()
            if not self.is_blur:
                self.check_sap_noise()
                self.check_low_contrast()
            else:
                self.image_ok = False
        else:
            self.image_ok = False


        if self.msg_problem is not None:
            # problem found -> notify
            if self.bot is not None:
                self.bot.sendMessage(self.chat_id, self.msg_problem)
            print(self.msg_problem)
        else:
            if self.bot is not None:
                self.bot.sendMessage(self.chat_id, "Immagine di buona qualità.")
                #print("Immagine di buona qualità.")

        return self.image_ok #, self.is_blur, self.is_empty, self.is_noisy, self.low_contrast


    #### WORKER FUNCTIONS ####
    # quality #
    def check_uniform(self):
        """ Check if the image is all white or black or with low variance"""

        mean = np.mean(self.img)
        if mean < 2 or mean > 253:
            self.msg_problem = "Hai inviato un'immagine priva di contenuto."
            self.is_empty = True
        else:
            self.is_empty = False

        _, std = cv2.meanStdDev(self.img)
        if self.debug:
            #self.bot.sendMessage(self.chat_id, f"[CHECK UNIFORM] std: {std}")
            print("[CHECK UNIFORM] std: {}".format(str(std)))
        for el in std:
            if el[0] < STD_THRESHOLD:
                self.msg_problem = "Hai inviato un'immagine priva di contenuto."
                self.is_empty = True
            else:
                self.is_empty = False

    def check_blur(self):
        """ 
        Check for blurry images using variance of laplacian.
        Since the Laplacian uses the gradient of images, 
        it calls internally the Sobel operator to perform its computation.
        """

        self.variance_of_laplacian = cv2.Laplacian(self.bw_img, cv2.CV_64F).var()

        if self.debug:
            #self.bot.sendMessage(self.chat_id, f"[CHECK BLUR] Varianza del Laplaciano: {self.variance_of_laplacian}")
            print("[CHECK BLUR] Varianza del Laplaciano: {self.variance_of_laplacian}")

        if self.variance_of_laplacian < VARIANCE_OF_LAPLACIAN_THRESHOLD_LOW:
            self.msg_problem = "La tua immagine è sfocata, per favore scattane una migliore."
            self.is_blur = True
        elif self.variance_of_laplacian >= VARIANCE_OF_LAPLACIAN_THRESHOLD_LOW and self.variance_of_laplacian < VARIANCE_OF_LAPLACIAN_THRESHOLD_HIGH:
            self.msg_problem = "L'immagine sembra essere leggermente sfocata, continuo con l'elaborazione..."
            self.is_blur = False
        else:
            self.is_blur = False

    def median_filter(self, data, filter_size=SAP_WINDOW):
        temp = []
        indexer = filter_size // 2
        data_final = []
        data_final = np.zeros((len(data),len(data[0])))

        pixel_of_interest = 0

        for i in range(len(data)):

            for j in range(len(data[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(data[i + z - indexer][j + k - indexer])

                temp.sort()
                data_final[i][j] = temp[len(temp) // 2]
                temp = []
                # end of median filter step

                # count pixel of interest
                if data[i][j] == 0 and data_final[i][j] != 0:
                    pixel_of_interest += 1
                if data[i][j] == 255 and data_final[i][j] != 255:
                    pixel_of_interest += 1

        # stats
        ratio = pixel_of_interest / (data.shape[0] * data.shape[1])
        is_noisy = ratio >= SAP_NOISE_THRESHOLD
        if self.debug:
            print(f"Changed {pixel_of_interest} pixels of interest")
            print("S&P noise ratio: {0:.3f}, noisy: {1}".format(ratio, is_noisy))

        return data_final, is_noisy

    def faster_median_filter(self, data, width=SAP_WINDOW):
        shape = data.shape
        data = data.flatten()
        pixel_of_interest = 0

        l = list(data[0].repeat(width))
        mididx = (width - 1) // 2
        result = np.empty_like(data)

        for idx, new_elem in enumerate(data):
            old_elem = data[max(0, idx - width)]
            del l[bisect_left(l, old_elem)]
            insort(l, new_elem)
            result[idx] = l[mididx]

            # count pixel of interest
            if old_elem == 0 and result[idx] != 0:
                pixel_of_interest += 1
            if old_elem == 255 and result[idx] != 255:
                pixel_of_interest += 1

        # stats
        ratio = pixel_of_interest / len(result)
        is_noisy = ratio >= SAP_NOISE_THRESHOLD
        if self.debug:
            print('[CHECK S&P NOISE]')
            print(f"Changed {pixel_of_interest} pixels of interest")
            print("S&P noise ratio: {0:.3f}, noisy: {1}".format(ratio, is_noisy))
        return np.reshape(result, shape), is_noisy

    def check_sap_noise(self):
        """ Salt and pepper noise removal """
        r, g, b = cv2.split(self.img)
        
        denoised_r, is_noisy_r = self.faster_median_filter(r)
        denoised_g, is_noisy_g = self.faster_median_filter(g)
        denoised_b, is_noisy_b = self.faster_median_filter(b)

        self.is_noisy = is_noisy_r or is_noisy_g or is_noisy_b          

        if self.is_noisy:
            self.msg_problem = "Sembra che la tua immagine sià affetta da rumore di tipo Sale e Pepe, correggo..."
            # self.denoised = cv2.merge([denoised_r, denoised_g, denoised_b])
            self.denoised = cv2.medianBlur(self.img, SAP_WINDOW)
            # debug save
            # utils.save_image(self.denoised, "query_img/test.jpg", "denoised_sap")


    def check_low_contrast(self):
        """
        Check image contrast.
        An image is considered low-contrast when its range of brightness spans less
        than this fraction (THRESHOLD) of its data type’s full range.
        """

        low_contrast = is_low_contrast(self.bw_img, fraction_threshold=CONTRAST_THRESHOLD)
        if low_contrast:
            self.msg_problem = "Sembra che il contrasto della tua immagine sia basso, continuo il processing..."
            self.low_contrast = True
        else:
            self.low_contrast = False


    # enhancement #
    def enhance_image(self):

        # starts with denoised image if necessary
        img = self.img
        if self.is_noisy:
            return self.denoised.copy(), True

        if self.low_contrast:
            # convert from RGB color-space to YCrCb
            ycrcb_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
            
            # Contrast Limited Adaptive Histogram Equalization
            # https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])

            # convert back to RGB color-space from YCrCb
            equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

            # remove gaussian noise
            # https://docs.opencv.org/3.4/d1/d79/group__photo__denoise.html#ga4c6b0031f56ea3f98f768881279ffe93
            # http://www.ipol.im/pub/art/2011/bcm_nlm/
            equalized_denoised_img = cv2.fastNlMeansDenoising(equalized_img,None,10,7,21)

            return equalized_denoised_img, True

        return img, False