import numpy as np
from PIL import Image
from i008.files import list_files


def load_img(path, grayscale=False, target_size=None):
    """
    Load an image into PIL format.

    # Arguments
    path: path to image file
    grayscale: boolean
    target_size: None (default to original size)
    or (img_height, img_width)
    """

    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def load_image_keras(image_path, dim_ordering='default', gray=False, target_size=(300, 300)):
    from keras.preprocessing import image
    img = load_img(image_path, grayscale=gray, target_size=target_size)
    return np.expand_dims(image.img_to_array(img, dim_ordering=dim_ordering), axis=0)


def load_images_keras(images_path_list):
    return np.concatenate(
        [load_image_keras(p) for p in images_path_list],
        axis=0
    )


def load_image_keras_imagenet_compatible(image_path, gray=False, target_size=(224, 224)):
    from keras.preprocessing import image
    from keras.applications.vgg19 import preprocess_input

    im = image.load_img(image_path, grayscale=gray, target_size=target_size)
    imarray = image.img_to_array(im)
    imarray = np.expand_dims(imarray, axis=0)
    imarray = preprocess_input(imarray)
    return imarray


def list_images(base_path, contains=None):
    # return the set of files that are valid
    return list_files(base_path, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)
