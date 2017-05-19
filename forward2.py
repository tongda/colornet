import numpy as np
import skimage.transform
import tensorflow as tf
from skimage.io import imsave, imread

from train import rgb2yuv, yuv2rgb, phase_train


def fake_vgg_batch_norm(layer, weights, pool_index):
    exp = 0.001
    beta = weights['pool{}_beta'.format(pool_index)]
    gamma = weights['pool{}_gamma'.format(pool_index)]
    mean = weights['pool{}_mean'.format(pool_index)]
    var = weights['pool{}_variance'.format(pool_index)]
    return gamma * (layer - mean) / tf.sqrt(var + exp) + beta


def colornet2(_tensors):
    """
    Network architecture http://tinyclouds.org/colorize/residual_encoder.png
    """
    with tf.variable_scope('colornet'):
        with tf.variable_scope('conv1'):
            # Bx28x28x512 -> batch norm -> 1x1 conv = Bx28x28x256
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(fake_vgg_batch_norm(_tensors["conv4_3"],weights, 4),
                                                           _tensors["weights"]["wc1"], [1, 1, 1, 1], 'SAME'),
                                              _tensors["weights"]["bias1"]))
            # upscale to 56x56x256
            conv1 = tf.image.resize_bilinear(conv1, (56, 56))
            conv1 = tf.add(conv1, fake_vgg_batch_norm(_tensors["conv3_3"], weights, 3))

        with tf.variable_scope('conv2'):
            # Bx56x56x256-> 3x3 conv = Bx56x56x128
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, _tensors["weights"]['wc2'], [1, 1, 1, 1], 'SAME'),
                                              _tensors["weights"]["bias2"]))
            # upscale to 112x112x128
            conv2 = tf.image.resize_bilinear(conv2, (112, 112))
            conv2 = tf.add(conv2, fake_vgg_batch_norm(_tensors["conv2_2"], weights, 2))

        with tf.variable_scope('conv3'):
            # Bx112x112x128 -> 3x3 conv = Bx112x112x64
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, _tensors["weights"]['wc3'], [1, 1, 1, 1], 'SAME'),
                                              _tensors["weights"]["bias3"]))
            # upscale to Bx224x224x64
            conv3 = tf.image.resize_bilinear(conv3, (224, 224))
            conv3 = tf.add(conv3, fake_vgg_batch_norm(_tensors["conv1_2"], weights, 1))

        with tf.variable_scope('conv4'):
            # Bx224x224x64 -> 3x3 conv = Bx224x224x3
            conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, _tensors["weights"]['wc4'], [1, 1, 1, 1], 'SAME'),
                                              _tensors["weights"]["bias5"]))
            conv4 = tf.add(conv4, _tensors["grayscale"])

        with tf.variable_scope('conv5_6'):
            # Bx224x224x3 -> 3x3 conv = Bx224x224x3
            conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, _tensors["weights"]['wc5'], [1, 1, 1, 1], 'SAME'),
                                              _tensors["weights"]["bias5"]))
            # Bx224x224x3 -> 3x3 conv = Bx224x224x2
            conv6 = tf.nn.sigmoid(tf.nn.bias_add(tf.nn.conv2d(conv5, _tensors["weights"]['wc6'], [1, 1, 1, 1], 'SAME'),
                                              _tensors["weights"]["bias6"]))

    return conv6


def load_image(path):
    img = imread(path)
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    img = skimage.transform.resize(crop_img, (224, 224))
    # desaturate image
    return (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0


with open("vgg/tensorflow-vgg16/vgg16-20160129.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

with tf.variable_scope('colornet'):
    # Store layers weight
    weights = {
        # 1x1 conv, 512 inputs, 256 outputs
        'wc1': tf.Variable(
            np.transpose(np.fromfile('./weights/color4_filter', dtype=np.float32).reshape(256, 1, 1, 512),
                         [1, 2, 3, 0]), name="wc1"),
        'bias1': tf.Variable(np.fromfile('./weights/color4_biases', dtype=np.float32), name="bias1"),
        # 3x3 conv, 512 inputs, 128 outputs
        'wc2': tf.Variable(
            np.transpose(np.fromfile('./weights/color3_filter', dtype=np.float32).reshape(128, 3, 3, 256),
                         [1, 2, 3, 0]), name="wc2"),
        'bias2': tf.Variable(np.fromfile('./weights/color3_biases', dtype=np.float32), name="bias2"),
        # 3x3 conv, 256 inputs, 64 outputs
        'wc3': tf.Variable(
            np.transpose(np.fromfile('./weights/color2_filter', dtype=np.float32).reshape(64, 3, 3, 128), [1, 2, 3, 0]),
            name="wc3"),
        'bias3': tf.Variable(np.fromfile('./weights/color2_biases', dtype=np.float32), name="bias3"),
        # 3x3 conv, 128 inputs, 3 outputs
        'wc4': tf.Variable(
            np.transpose(np.fromfile('./weights/color1_filter', dtype=np.float32).reshape(3, 3, 3, 64), [1, 2, 3, 0]),
            name="wc4"),
        'bias4': tf.Variable(np.fromfile('./weights/color1_biases', dtype=np.float32), name="bias4"),
        # 3x3 conv, 6 inputs, 3 outputs
        'wc5': tf.Variable(
            np.transpose(np.fromfile('./weights/color0_filter', dtype=np.float32).reshape(3, 3, 3, 3), [1, 2, 3, 0]),
            name="wc5"),
        'bias5': tf.Variable(np.fromfile('./weights/color0_biases', dtype=np.float32), name="bias5"),
        # 3x3 conv, 3 inputs, 2 outputs
        'wc6': tf.Variable(
            np.transpose(np.fromfile('./weights/uv_filter', dtype=np.float32).reshape(2, 3, 3, 3), [1, 2, 3, 0]),
            name="wc6"),
        'bias6': tf.Variable(np.fromfile('./weights/uv_biases', dtype=np.float32), name="bias6"),
    }

    for i in range(1, 5):
        weights['pool{}_beta'.format(i)] = tf.Variable(
            np.fromfile('./weights/pool{}_bn_beta'.format(i), dtype=np.float32), name="pool{}_beta".format(i))
        weights['pool{}_gamma'.format(i)] = tf.Variable(
            np.fromfile('./weights/pool{}_bn_gamma'.format(i), dtype=np.float32), name="pool{}_gamma".format(i))
        weights['pool{}_mean'.format(i)] = tf.Variable(
            np.fromfile('./weights/pool{}_bn_mean'.format(i), dtype=np.float32), name="pool{}_mean".format(i))
        weights['pool{}_variance'.format(i)] = tf.Variable(
            np.fromfile('./weights/pool{}_bn_variance'.format(i), dtype=np.float32), name="pool{}_variance".format(i))

grayscale = tf.placeholder(tf.float32, (1, 224, 224, 1))
grayscale_rgb = tf.image.grayscale_to_rgb(grayscale)
grayscale_yuv = rgb2yuv(grayscale_rgb)
grayscale_1 = tf.concat([grayscale, grayscale, grayscale], 3)

tf.import_graph_def(graph_def, input_map={"images": grayscale_1})

graph = tf.get_default_graph()

with tf.variable_scope('vgg'):
    conv1_2 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
    conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
    conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
    conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")

tensors = {
    "conv1_2": conv1_2,
    "conv2_2": conv2_2,
    "conv3_3": conv3_3,
    "conv4_3": conv4_3,
    "grayscale": grayscale,
    "weights": weights
}

shark_gray = load_image("shark.jpg").reshape(1, 224, 224, 1)

pred = colornet2(tensors)
pred_yuv = tf.concat([tf.split(grayscale_yuv, 3, 3)[0], pred], 3)
pred_rgb = yuv2rgb(pred_yuv)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inferred_batch = sess.run(pred_rgb, feed_dict={grayscale: shark_gray, phase_train: False})
    imsave("shark-color2.jpg", inferred_batch[0])
    print("saved shark-color.jpg")
