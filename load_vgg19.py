
import tensorflow as tf
from tensorflow import keras

#TODO: Change the path to your groups personal folder
#####################################################
your_groups_path = "/beegfs/work/fpds05/"
#####################################################

def save_model(path_name=your_groups_path):
    """
    Save the vgg19 model.

    :return: None
    :rtype: None
    """
    print("Save model...")
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.save(f"{path_name}vgg19_imagenet_no_top.h5")
    print("Model saved...")
    vgg.summary()

def load_model(path_name=your_groups_path, show_summary=False):
    """
    Load the previously saved model.

    :param path_name: path to your model
    :type path_name: str
    :param show_summary: If True, shows model summary after loading
    :type show_summary: bool
    :return: tensorflow model
    :rtype: tf.model
    """
    print("Load Model...")
    model = keras.models.load_model(f"{path_name}vgg19_imagenet_no_top.h5")
    print("Model loaded...")
    if show_summary:
        model.summary()
    return model


if __name__ == "__main__":
    save_model()
    loaded_vgg_model = load_model(show_summary=True)