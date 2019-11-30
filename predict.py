from keras.models import load_model
import os,sys
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import layers
from sklearn.metrics import roc_auc_score
from keras import backend as K
import tensorflow as tf
import os

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.backend import clear_session

from model_builder import buil_bcnn
from data_loader import build_generator

img_width = 224
img_height = 224
batch_size = 32
nbr_test_samples = 5339

FishNames = ['0', '1']

root_path = './'
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

weights_path = os.path.join(root_path, 'checkpoints/Xception_model_19-0.969.h5')

test_data_dir = os.path.join('../rubbish_classification/Test_A/')
#test_data_dir = os.path.join(r"C:\data\智慧环保初赛测试集及提交样例\智慧环保初赛测试集及提交样例\Test_A")
# test data generator for prediction
test_datagen = ImageDataGenerator(
    rescale=1./255
    )

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

test_image_list = test_generator.filenames
#print(test_image_list[:10])
#sys.exit(0)
print('Loading model and weights from training process ...')
model = buil_bcnn(
    all_trainable=False,
    no_class=2,
    name_optimizer='sgd',
    learning_rate=0.0001,
    decay_learning_rate=1e-8)

model.load_weights(weights_path)

print('Begin to predict for testing data ...')
predictions = model.predict_generator(test_generator, nbr_test_samples)

np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)


print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('pic_id,pred\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % predictions[i, 1]]#只去第1列，也就是预测为1的概率
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('pic_%s,%s\n' % (os.path.basename(image_name.split(".")[0]), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
