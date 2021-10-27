from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}

IMG_TARGET_SIZE = 512 	# used for resize images


def resize_image(img, size, interpolation=Image.BILINEAR):
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh), interpolation)
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh), interpolation)


def load_image(filename):
    """
    Load an image converting from grayscale or alpha as needed.
    Parameters
    ----------
    filename : string (path) or IOstream
    Returns
    -------
    image : a Image of size (H x W x 3) in RGB.
    """
    # Image load with PIL
    _img = Image.open(filename).convert('RGBA')
    _img.load()  # needed for split()
    img = Image.new('RGB', _img.size, (255,255,255))
    img.paste(_img, mask=_img.split()[-1])  # -1 is the alpha channel

    # Check exif for orientation
    if hasattr(img, '_getexif'):
        exif = img._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            if orientation in ORIENTATIONS:
                for method in ORIENTATIONS[orientation]:
                    img = img.transpose(method)

    # Resize image
    img = resize_image(img, IMG_TARGET_SIZE)
    return img


def show_images(query, best_imgs, distances):
    num = len(best_imgs)
    
    fig, axs = plt.subplots(2, num)
    [axs[0,t].axis('off') for t in range(num)]
    axs[0,2].imshow(query)
    axs[0,2].title.set_text('Query image')
    for i, img in enumerate(best_imgs):
        axs[1,i].imshow(img)
        axs[1,i].title.set_text('Top {}\n ({})'.format(str(i+1), str(distances[i])))
        axs[1,i].axis('off')

    plt.show()

def show_images_linear(query, query_class, query_group, best_imgs, idxs, idxs_group, distances, count):
    num = len(best_imgs)

    plt.figure(figsize=(20,10))
    columns = 6
    plt.subplot(len(best_imgs) / columns +1, columns, 0 +1)
    plt.imshow(query)
    plt.title('Query image\n (class: {}) \n (group: {}) \n (distance: {})'.format(str(query_class), 
                                                                                    str(query_group), 0))
    plt.axis('off')
    for i, image in enumerate(best_imgs):
        plt.subplot(len(best_imgs) / columns + 1, columns, i+1 + 1)
        plt.imshow(image)
        plt.title('Top {}\n (class: {}) \n (group: {}) \n(distance: {})'.format(str(i+1), str(idxs[i]),  
                                                                                str(idxs_group[i]), 
                                                                                str(np.around(distances[i],2))))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def pil_to_opencv(input):
    """ Converts PIL image into openCV2 image """
    if isinstance(input, str):
        pil_image = Image.open(input)
    else:
        pil_image = input

    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencvImage

def opencv_to_pil(input):
    """ Converts openCV2 image into PIL image """
    pilImage = Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
    return pilImage
    

def show_cv2(input):
    cv2.imshow('img', input)
    cv2.waitKey(0) 



#### Evaluation Metrics ####
def AP(label, results):
    ''' Average precision of a single query'''
    precision = []
    hit = 0
    for i, result in enumerate(results):
        if result == label[0]:
            hit += 1
            precision.append(hit / (i+1.))
    if hit == 0:
        return 0.
    return np.mean(precision)


def apk(actual, predicted, k=10, all=False, tot_relevant=1000):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py#L25-L39

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if all:
        k = len(predicted)
    else:
        if len(predicted)>k:
            predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual.any():
        return 0.0

    #return score / min(len(actual), k)
    return score / min(k, tot_relevant)

def mapk(actual, predicted, tot_relevant, k=100, all=False):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k, all, tot_rel) for a,p,tot_rel in zip(actual, predicted, tot_relevant)])


#### Distances ####
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))