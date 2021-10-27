import os
import cv2

def fileparts(full_string=None):
    """
    Splits a full string image path into three components:
    directory, filename and extension.
    """
    (dir_name, file_name) = os.path.split(full_string)
    (file_base_name, file_extension) = os.path.splitext(file_name)
    return dir_name, file_base_name, file_extension


def save_image(img=None, source_img_path=None, suffix=None):
    """
    Saves a new image by appending a suffix to the provided source image path
    Returns img path.
    """
    dir_name, file_base_name, file_extension = fileparts(source_img_path)
    new_path = os.path.join(dir_name, file_base_name + suffix + file_extension)

    cv2.imwrite(new_path, img)
    return new_path