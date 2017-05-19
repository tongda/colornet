import tensorflow as tf
import skimage.transform
from skimage.io import imsave, imread
import numpy as np


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


shark_gray = load_image("shark.jpg").reshape(1, 224, 224, 1)
print(np.max(shark_gray))

with open("colorize.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
grayscale = tf.placeholder("float", [1, 224, 224, 1])
tf.import_graph_def(graph_def, input_map={"grayscale": grayscale}, name="colorize")

with tf.Session() as sess:
    inferred_rgb = sess.graph.get_tensor_by_name("colorize/inferred_rgb:0")
    uv_ = sess.graph.get_tensor_by_name("colorize/uv/Sigmoid:0")
    color0_ = sess.graph.get_tensor_by_name("colorize/color0/color0_bn/BatchNormWithGlobalNormalization:0")
    vgg4_3_ = sess.graph.get_tensor_by_name("colorize/vgg16/conv4_3/Conv2D:0")
    inferred_batch, uv, color0, vgg4_3 = sess.run([inferred_rgb, uv_, color0_, vgg4_3_], feed_dict={grayscale: shark_gray})
    imsave("shark-color.jpg", inferred_batch[0])
    print("saved shark-color.jpg")
    print(uv.shape)
    np.set_printoptions(precision=3, threshold=np.nan)
    with open("test.txt", "w") as f:
        f.write(np.array_str(np.transpose(vgg4_3[0], (2, 0, 1))))
    # print(uv[0][:, :, 0] * 255)
    # print(uv[0][:, :, 1] * 255)
    image = (np.concatenate([uv.reshape(224, 224, 2), np.zeros((224, 224, 1))], axis=2) * 255).astype(np.uint8)
    imsave("test.jpg", image)
    print(np.max(color0.reshape(224, 224, 3).astype(np.int8)))
    print(np.min(color0.reshape(224, 224, 3).astype(np.int8)))
    color0[color0 > 1] = 1
    color0[color0 < 0] = 0
    imsave("color0.jpg", color0.reshape(224, 224, 3))
