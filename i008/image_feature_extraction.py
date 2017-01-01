import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn_theano.feature_extraction.overfeat import SMALL_NETWORK_FILTER_SHAPES
from i008.images import load_image_keras_imagenet_compatible


def build_batch(paths, fixed_size):
    # load the images from disk, prepare them for extraction, and convert
    # the list to a NumPy array
    images = [prepare_image(cv2.imread(p), fixed_size) for p in paths]
    images = np.array(images, dtype="float")

    # extract the labels from the image paths
    labels = [":".join(p.split("/")[-2:]) for p in paths]

    # return the labels and images
    return labels, images


def prepare_image(image, fixed_size):
    # convert the image from BGR to RGB, then resize it to a fixed size,
    # ignoring aspect ratio
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, tuple(fixed_size))

    # return the image
    return image


class OverfeatExtractor:
    def __init__(self, layer_num):
        # store the layer number and initialize the Overfeat transformer
        self.layer_num = layer_num
        self.of = OverfeatTransformer(output_layers=[layer_num])

    def describe(self, data):
        # apply the Overfeat transfrom to the images
        return self.of.transform(data)

    def getFeatureDim(self):
        # return the feature dimensionality from the supplied layer
        return SMALL_NETWORK_FILTER_SHAPES[self.layer_num][0]

        # im_paths = list(list_images('/home/i008/cars_train/'))[:2]
        # ofe = OverfeatExtractor(-3)
        # labels, images = build_batch(im_paths, (231, 231))
        # f = ofe.describe(images)


class ImageNetExtractor:
    def __init__(self, architecture='vgg19', include_top=False):
        self.architecture = architecture
        self.include_top = include_top

        if architecture == 'vgg19':
            self.model = VGG19(include_top=include_top)
        elif architecture == 'resnet':
            self.model = ResNet50(include_top=include_top)
        else:
            raise ValueError('unsupported architecture')

    def describe(self, array_of_images):
        shape = array_of_images.shape

        if not len(shape) == 4:
            raise ValueError('Keras required imnages to be passed as 4 dim array, for example shape = (n, 3, 222,222)')
        if not shape[1:] in [(3, 224, 224), (224, 224, 3)]:
            raise ValueError(
                'Imagenet requires images with shape (n, 3, 224, 224) / (n, 224, 224, 3) \n {}'.format(shape))

        return self.model.predict(array_of_images).flatten()

    def describe_from_path(self, list_of_image_pahts):
        array_of_images = np.concatenate([load_image_keras_imagenet_compatible(p) for p in list_of_image_pahts], axis=0)
        return self.describe(array_of_images)


if __name__ == '__main__':
    # from keras import backend as K
    # K.set_image_dim_ordering('tf')
    ex = ImageNetExtractor('resnet')
    sh = ex.describe_from_path(
        ['/home/i008/00019.jpg']
    )

    print(sh.shape)
    # ex.model(sh)
