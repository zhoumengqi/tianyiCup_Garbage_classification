'''Main script.'''

import os
from model_trainer import train_model

def transfer_learning(
        train_dir,
        valid_dir,
        no_class=200,
        epoch=200,
        batch_size=128,
        tensorboard_dir=None,
        checkpoint_dir=None
    ):
    '''For transfer learning.

    Build the model and fix all VGG16 layers to untrainable.
    Transfer the FC layer to the target classes.
    '''

    return train_model(
        name_optimizer='adam',
        learning_rate=0.001,
        decay_learning_rate=1e-9,
        train_dir=train_dir,
        valid_dir=valid_dir,
        no_class=no_class,
        epoch=epoch,
        batch_size=batch_size,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir
    )

def fine_tuning(
        train_dir,
        valid_dir,
        model_weights_path,
        all_trainable=True,
        no_class=200,
        epoch=100,
        batch_size=64,
        tensorboard_dir=None,
        checkpoint_dir=None
    ):
    '''For fine tuning.

    Load a model and make all layers trainbale.
    Fine tune the whole model.
    '''

    return train_model(
        name_optimizer='sgd',
        learning_rate=0.0001,
        decay_learning_rate=1e-8,
        train_dir=train_dir,
        valid_dir=valid_dir,
        all_trainable=all_trainable,
        model_weights_path=model_weights_path,
        no_class=no_class,
        epoch=epoch,
        batch_size=batch_size,
        tensorboard_dir=tensorboard_dir,
        checkpoint_dir=checkpoint_dir
    )


if __name__ == "__main__":
    import os
    TRAIN_DIR = '../rubbish_classification/all_train_split'
    VALID_DIR = '../rubbish_classification/all_val_split'

    # Start transfer learning
    TENSORBOARD_DIR = './logs_tl'
    CHECKPOINT_DIR = './checkpoints'

    if not os.path.exists(TENSORBOARD_DIR):
        os.makedirs(TENSORBOARD_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Transfer the model to 4 classes, images and labels of
    # each class should be well prepared in train and validataion directory.
    transfer_learning(
        TRAIN_DIR,
        VALID_DIR,
        no_class=2,
        batch_size=64)


    # Start fine-tuning
    MODEL_WEIGHTS_PATH = './new_model_weights.h5'
    TENSORBOARD_DIR = './logs_ft'
    CHECKPOINT_DIR = './checkpoints'

    if not os.path.exists(TENSORBOARD_DIR):
        os.makedirs(TENSORBOARD_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Load the generated weights from transer learning,
    # then fine tune all layers.
    fine_tuning(
        TRAIN_DIR,
        VALID_DIR,
        model_weights_path=MODEL_WEIGHTS_PATH,
        no_class=2,
        batch_size=16,
        tensorboard_dir=TENSORBOARD_DIR,
        checkpoint_dir=CHECKPOINT_DIR)
