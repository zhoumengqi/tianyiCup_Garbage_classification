'''Load data and build generators.'''

from data_preprocesser import normalize_image, random_crop_image, center_crop_image
from data_preprocesser import resize_image, horizontal_flip_image
# from data_preprocesser import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
import random,cv2
import numpy as np
def train_preprocessing(x, size_target=(244, 244)):
    '''Preprocessing for train dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return normalize_image(
        center_crop_image(
            horizontal_flip_image(
                resize_image(
                    x,
                    size_target=size_target,
                    flg_keep_aspect=True
                )
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )

def valid_preprocessing(x, size_target=(244, 244)):
    '''Preprocessing for validation dataset image.

    Args:
        x: input image.
        size_target: a tuple (height, width) of the target size.

    Returns:
        Preprocessed image.
    '''
    return normalize_image(
        center_crop_image(
            resize_image(
                x,
                size_target=size_target,
                flg_keep_aspect=True
            )
        ),
        mean=[123.82988033, 127.3509729, 110.25606303]
    )
def random_crop(img,w = 244,h = 244):
    #随机生成多边形并填充
    n = random.randint(5, 10)  # n个点
    # img = np.zeros((w, h, 3), np.int32)
    # img = np.asarray(img, np.int32)
    point = np.zeros((n, 2))
    for i in range(n):
        x = random.randint(int(0), int(w  ))
        y = random.randint(int(0 ), int(h  ))
        point[i][0] = x
        point[i][1] = y
        # print(x, y)
    point = np.asarray(point, np.int32)
    # plt.imshow(img)
    color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    res=cv2.fillPoly(img, [point], color)
    img = np.asarray(res.get(), np.int32)
    # print(res)

    # plt.imshow(img)
    # plt.show()
    # sys.exit(0)
    return res.get()
def build_generator(
        train_dir=None,
        valid_dir=None,
        batch_size=128
    ):
    '''Build train and validation dataset generators.

    Args:
        train_dir: train dataset directory.
        valid_dir: validation dataset directory.
        batch_size: batch size.

    Returns:
        Train generator and validation generator.
    '''

    results = []
    if train_dir:
        train_datagen = ImageDataGenerator(
            preprocessing_function=random_crop,
            rescale = 1./255,
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=100,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.4,
            zoom_range=0.4,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0
        )
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [train_generator]

    if valid_dir:
        valid_datagen = ImageDataGenerator(
            preprocessing_function=random_crop,
            rescale = 1./255,
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=100,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.4,
            zoom_range=0.4,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0
        )
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(244, 244),
            batch_size=batch_size,
            class_mode='categorical'
        )
        results += [valid_generator]

    return results


if __name__ == "__main__":
    pass
