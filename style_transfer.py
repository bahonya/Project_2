import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import PIL
from typing import List

tf.keras.backend.set_floatx('float64')

def load_img(path_to_img, max_dim):
  """tf.image.decode_image already has 'dtype' option, so f.image.convert_image_dtype can be omitted"""
  img = tf.image.decode_image(tf.io.read_file(path_to_img), dtype=tf.float64)
  img = tf.image.resize(img, size=[max_dim, max_dim], preserve_aspect_ratio=True)
  return tf.cast(tf.expand_dims(img, axis=0), dtype=tf.float64)

def vgg_layers(layer_names: List[str]) -> tf.keras.models.Model:
    vgg19 = tf.keras.models.load_model("vgg19_imagenet_no_top.h5")
    model = tf.keras.Model(inputs = vgg19.input, outputs = [vgg19.get_layer(layer_name).output for layer_name in layer_names])
    return model

def gram_matrix(input_tensor):
    summation = tf.linalg.einsum('aijb,aijc->abc', input_tensor, input_tensor)
    shape= tf.cast(tf.shape(input_tensor), dtype= tf.float64)
    return summation/(shape[1]*shape[2]*shape[3])

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=-0, clip_value_max=1)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_sum((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= style_weight /(4 * len(style_outputs))
    content_loss = tf.add_n([tf.reduce_sum((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= content_weight /(2 * len(content_outputs))
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image, opt, extractor, style_targets, content_targets, style_weight, content_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)
    grad = tape.gradient(loss, image)    
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss

class StyleContentModel(tf.keras.models.Model):  
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
        
    def call(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs*255)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

def apply_style_transfer(max_dim, content_image_path, style_image_path, result_image_path, parameters_dict):
    content_image = load_img(content_image_path, max_dim=max_dim)
    style_image = load_img(style_image_path, max_dim=max_dim)
    extractor = StyleContentModel(parameters_dict["style_layers"], parameters_dict["content_layers"])
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]
    image = tf.Variable(content_image)
    opt = parameters_dict["opt"]

    file_writer = tf.summary.create_file_writer('logs' + f'/stw{parameters_dict["style_weight"]}_cow{parameters_dict["content_weight"]}')
    file_writer.set_as_default()
    for loop in range(parameters_dict["loops"]):
        tf.summary.image('image', data=image, step=loop * parameters_dict["iters_per_loop"])
        for it in range(parameters_dict["iters_per_loop"]):
            # YOUR CODE
            loss = train_step(image, opt, extractor, style_targets, content_targets, parameters_dict["style_weight"], parameters_dict["content_weight"])
            tf.summary.scalar('loss', data=loss, step=loop * parameters_dict["iters_per_loop"] + it)
    tensor_to_image(image).save(result_image_path)


# class StyleTransfer:
#     def __init__(self, max_dim, content_image, style_image, style_layers, 
#     content_layers, content_weight, style_weight, tv_weight, optimizer):
#         self.max_dim = max_dim
#         self.extractor = StyleContentModel(style_layers, content_layers)
#         self.style_targets = self.extractor(load_img(style_image, max_dim=max_dim))["style"]
#         self.content_targets = self.extractor(load_img(content_image, max_dim=max_dim))["style"]
#         self.style_weight = style_weight
#         self.content_weight = content_weight
#         self.image = tf.Variable(load_img(content_image, max_dim=max_dim))
#         self.optimizer = optimizer
#         self.tv_weight = tv_weight

#     def train(self, loops=100, iters_per_loop=10):
#         # YOUR CODE
#         opt = self.optimizer

#         file_writer = tf.summary.create_file_writer('logs' + f'/stw{self.style_weight}_cow{self.content_weight}')
#         file_writer.set_as_default()

#         for loop in range(loops):
#             tf.summary.image('image', data=self.image, step=loop * iters_per_loop)
#             for it in range(iters_per_loop):
#                 # YOUR CODE
#                 loss = train_step(self.image, self.optimizer, self.extractor, self.style_targets, self.content_targets, self.style_weight, self.content_weight)
#                 tf.summary.scalar('loss', data=loss, step=loop * iters_per_loop + it)