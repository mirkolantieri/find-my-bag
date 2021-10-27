import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from skimage.feature import local_binary_pattern


def colorHSV(image, mask):
    bins = [8,12,3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_features = []
    for index, channel in enumerate(cv2.split(image)):
        if index == 0:
            range = [0, 180]
        else:
            range = [0, 256]
        hist = cv2.calcHist([channel], [0], mask, [bins[index]], range)
        hist = normalize(hist, norm='max', axis=0)
        color_features.append(hist[:,0])

    features = np.hstack(color_features)
    
    #hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    #import pdb; pdb.set_trace()
    #dst = np.zeros(hist.shape[:0], dtype="float")
    #hist = cv2.normalize(hist, dst)
    #features = hist.flatten()
    return features


def texture(image, mask):
    """ LBP histograms """
    # https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=local%20binary#skimage.feature.local_binary_pattern
    # https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patterns = local_binary_pattern(gray, 8, 1, method="uniform")
    n_bins = int(patterns.max() + 1)
    #plt.figure()
    #plt.imshow(patterns, cmap="gray")
    #plt.show()
    #hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    #hist_masked2, _ = np.histogram(patterns[mask>0], bins=np.arange(2**8 + 1), density=True)
    hist_masked = cv2.calcHist([np.uint8(patterns)], [0], mask, [n_bins], [0,n_bins])

    # normalize the histogram
    #hist = hist.astype("float")
    #hist /= (hist.sum() + 1e-7)
    #hist_masked = hist_masked.astype("float")
    #hist_masked /= (hist_masked.sum() + 1e-7)
    hist_masked_norm = normalize(hist_masked, norm='max', axis=0)

    features = np.hstack(hist_masked_norm)

    # plt.plot(hist)
    # plt.show()
    #print(f"LBP shape: {hist.shape}")
    return features


def huMoments(mask):
    features = cv2.HuMoments(cv2.moments(mask)).flatten()
    return features



### Not used ###
def raw_moments(image):
    """ default moments """
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.moments
    img = image.copy()
    M = cv2.moments(img, order=3)
    area = M[0, 0]
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    # print(area)
    # print(centroid)
    return area, centroid

def sift(image):
    """ SIFT descriptor """
    import pandas as pd
    # probably too big output shape ???
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    cv2.imwrite('img_sift_kp.jpg', img)
    print(f'{pd.DataFrame(des)}')
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.close()
    print(f"SIFT descriptor shape: {des.shape}")
    return des

def corners(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    #for i in corners:
    #    x, y = i.ravel()
    #    cv2.circle(img, (x, y), 30, 255, -1)
    #cv2.imwrite('img_corner.jpg', img)
    print(f"Corners shape: {corners.shape}")
    return corners

def ColorHSVlocal(image):
    # https://github.com/rasyadh/cbir-color/blob/master/cbir/colordescriptor.py
    bins = [8,12,3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    # Grab dimension and center of image
    (h, w) = image.shape[:2]
    (cX, cY) = (int(w * 0.5), int(h * 0.5))
    # Divide image to four regions
    # (top-left, top-right, bottom-left, bottom-right)
    regions = [(cX, 0, 0, cY), (cX, 0, w, cY), (0, cY, cX, h), (cX, cY, w, h)]
    # Loop over the regions
    for (startX, startY, endX, endY) in regions:
        regionMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(regionMask, (startX, startY), (endX, endY), 255, -1)
        # Extract color histogram from image
        hist = cv2.calcHist([image], [0, 1, 2], regionMask, bins, [0, 180, 0, 256, 0, 256])
        dst = np.zeros(hist.shape[:0], dtype="float")
        hist = cv2.normalize(hist, dst, norm_type=cv2.NORM_MINMAX)
        hist = hist.flatten()
        # Update feature
        features.append(hist)
    return features

def color(image, mask):
    """ Extracts color histogram """
    img = image.copy()
    color_features = np.empty((3, 256))
    for index, channel in enumerate(cv2.split(img)):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        color_features[index] = hist[:,0]
    #    plt.plot(hist)
    #    plt.xlim([0,256])
    #plt.show()
    print(f"Color features shape: {color_features.shape}")
    return color_features