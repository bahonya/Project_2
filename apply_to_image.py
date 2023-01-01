from style_transfer import apply_style_transfer
import tensorflow as tf
import PIL

# content_image = "images/img1_content.jpg"
# style_image = "images/img1_style.jpg"

# content_layers = ["block4_conv2"]
# style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]

# content_weight = 1e-3
# style_weight = 1
# tv_weight = 1e3
# opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)


# try_1 = StyleTransfer(128, content_image, style_image, style_layers=style_layers, content_layers=content_layers, content_weight=content_weight, style_weight=style_weight, tv_weight=tv_weight, opt)
# try_1.train()
# tensor_to_image(try_1.image).save("img1.png")
parameters_dict = {
    "content_layers": ["block4_conv2"],
    "style_layers": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
    "content_weight": 1e-3,
    "style_weight": 1,
    "opt": tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1),
    "loops": 100,
    "iters_per_loop": 10
}

apply_style_transfer(512, content_image_path="images/img3_content.jpg", style_image_path="images/img3_style.jpg", result_image_path="images/img3_result.jpg",parameters_dict=parameters_dict)