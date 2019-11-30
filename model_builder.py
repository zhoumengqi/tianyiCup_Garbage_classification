'''Build the Bilinear CNN model.'''

from keras import backend as keras_backend
from keras.layers import Input, Reshape, Dense, Lambda, Activation
from keras.optimizers import adam, RMSprop, SGD
from keras.models import Model
from keras.regularizers import l2
from keras.applications.vgg16 import VGG16
from keras.initializers import *
from keras.applications.densenet import DenseNet121
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
# from tensorflow.python.keras.applications.resnet import ResNet152
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import  Xception
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model
def _outer_product(x):
    '''Calculate outer-products of two tensors.

    Args:
        x: a list of two tensors.
        Assume that each tensor has shape = (size_minibatch, total_pixels, size_filter)

    Returns:
        Outer-products of two tensors.
    '''
    return keras_backend.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]

def _signed_sqrt(x):
    '''Calculate element-wise signed square-root.

    Args:
        x: input tensor.

    Returns:
        Element-wise signed square-root tensor.
    '''
    return keras_backend.sign(x) * keras_backend.sqrt(keras_backend.abs(x) + 1e-9)

def _l2_normalize(x, axis=-1):
    '''Calculate L2 normalization.

    Args:
        x: input tensor.
        axis: axis for narmalization.

    Returns:
        L2 normalized tensor.
    '''
    return keras_backend.l2_normalize(x, axis=axis)

def buil_bcnn(
        all_trainable=False,

        size_height=224,
        size_width=224,
        no_class=200,
        no_last_layer_backbone=-1,

        name_optimizer='sgd',
        learning_rate=1.0,
        decay_learning_rate=0.0,
        decay_weight_rate=0.0,

        name_initializer='glorot_normal',
        name_activation='softmax',
        name_loss='binary_crossentropy'
    ):
    '''Build Bilinear CNN.

    Detector and extractor are both VGG16.

    Args:
        all_trainable: fix or unfix VGG16 layers.
        size_height: default 224.
        size_width: default 224.
        no_class: number of prediction classes.
        no_last_layer_backbone: number of VGG16 backbone layer.
        name_optimizer: optimizer method.
        learning_rate: learning rate.
        decay_learning_rate: learning rate decay.
        decay_weight_rate: l2 normalization decay rate.
        name_initializer: initializer method.
        name_activation: activation method.
        name_loss: loss function.

    Returns:
        Bilinear CNN model.
    '''
    ##########################
    # Load pre-trained model #
    ##########################

    # Load model
    input_tensor = Input(shape=[size_height, size_width, 3])
    pre_train_model = DenseNet121(
        input_tensor=input_tensor,
        include_top=False,
        weights='imagenet')

    # Pre-trained weights
    for layer in pre_train_model.layers:
        layer.trainable = all_trainable


    ######################
    # Combine two models #
    ######################

    # Extract features form detecotr
    model_detector = pre_train_model
    output_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape

    # Extract features from extractor
    model_extractor = pre_train_model
    output_extractor = model_extractor.layers[no_last_layer_backbone].output
    shape_extractor = model_extractor.layers[no_last_layer_backbone].output_shape

    # Reshape tensor to (minibatch_size, total_pixels, filter_size)
    output_detector = Reshape(
        [shape_detector[1]*shape_detector[2], shape_detector[-1]])(output_detector)
    output_extractor = Reshape(
        [shape_extractor[1]*shape_extractor[2], shape_extractor[-1]])(output_extractor)

    # Outer-products
    x = Lambda(_outer_product)([output_detector, output_extractor])
    # Reshape tensor to (minibatch_size, filter_size_detector*filter_size_extractor)
    x = Reshape([shape_detector[-1]*shape_extractor[-1]])(x)
    # Signed square-root
    x = Lambda(_signed_sqrt)(x)
    # L2 normalization
    x = Lambda(_l2_normalize)(x)


    ###############################
    # Attach full-connected layer #
    ###############################

    if name_initializer is not None:
        name_initializer = eval(name_initializer+'()')

    # FC layer
    x = Dense(
        units=no_class,
        kernel_initializer=name_initializer,
        kernel_regularizer=l2(decay_weight_rate))(x)
    # x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    output_tensor = Activation(name_activation)(x)


    #################
    # Compile model #
    #################

    model_bcnn = Model(inputs=[input_tensor], outputs=[output_tensor])

    # Optimizer
    if name_optimizer == 'adam':
        optimizer = adam(lr=learning_rate, decay=decay_learning_rate)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(lr=learning_rate, decay=decay_learning_rate)
    elif name_optimizer == 'sgd':
        optimizer = SGD(lr=learning_rate, decay=decay_learning_rate, momentum=0.9, nesterov=None)
    else:
        raise RuntimeError('Optimizer should be one of Adam, RMSprop and SGD.')

    # Compile
    model_bcnn.compile(loss=name_loss, optimizer=optimizer, metrics=['accuracy'])
    # parralel_model = multi_gpu_model(model_bcnn, 3)
    # parralel_model.compile(loss=name_loss, optimizer=optimizer, metrics=['accuracy'])
    # print('-------- Mode summary --------')
    # print(model_bcnn.summary())
    # print('------------------------------')

    return model_bcnn

def save_model(
        size_height=224,
        size_width=224,
        no_class=200
    ):
    '''Save Bilinear CNN to current directory.

    The model will be saved as `model.json`.

    Args:
        size_height: default 224.
        size_width: default 224.
        no_class: number of prediction classes.

    Returns:
        Bilinear CNN model.
    '''
    model = buil_bcnn(
        size_height=size_height,
        size_width=size_width,
        no_class=no_class)

    # Save model json
    model_json = model.to_json()
    with open('./model.json', 'w') as f:
        f.write(model_json)

    print('Model is saved to ./model.json')

    return True


if __name__ == '__main__':
    pass
