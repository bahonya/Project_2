from style_transfer import apply_style_transfer
import tensorflow as tf
import PIL


parameters_dict = {
    "content_layers": ["block4_conv2"],
    "style_layers": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
    "content_weight": 1e-3,
    "style_weight": 1,
    "opt": tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1),
    "loops": 100,
    "iters_per_loop": 10
}

apply_style_transfer(1024, content_image_path="images/img7_content.jpg", style_image_path="images/img7_style.jpg", result_image_path="images/img7_result.jpg",parameters_dict=parameters_dict)