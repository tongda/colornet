from array import array

import tensorflow as tf
from itertools import chain
import re
import os
import numpy as np


with open("colorize.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)

tf.import_graph_def(graph_def)

print(tf.get_default_graph().as_graph_def().version)
print(tf.get_default_graph().as_graph_def().node[10].name)
print(type(tf.get_default_graph().as_graph_def().node[10].attr['value']))
print(type(tf.get_default_graph().get_operations()))
print(type(tf.get_default_graph().get_operation_by_name(
    "import/vgg16/conv3_1/filter").get_attr("value")))
# print([n.attr for n in tf.get_default_graph().as_graph_def().node])

with tf.Session() as sess:
    tensors = chain(
        *[op.outputs for op in tf.get_default_graph().get_operations()])
    patterns = [re.compile(r"import/vgg16/conv\d_\d/filter"),
                re.compile(r"import/vgg16/conv\d_\d/biases"),
                re.compile(r"import/color\d/filter"),
                re.compile(r"import/color\d/biases"),
                re.compile(r"import/color\d/color\d_bn/beta"),
                re.compile(r"import/color\d/color\d_bn/gamma"),
                re.compile(r"import/color\d/color\d_bn/mean"),
                re.compile(r"import/color\d/color\d_bn/variance")]
    tensors = [t for t in tensors if any(p.match(t.name) is not None for p in patterns)]
    results = sess.run(tensors)
    
    if not os.path.exists('weights'):
            os.mkdir('weights')

    print([t.name for t in tensors])

    param_mapping = dict(zip([t.name for t in tensors], results))
    exp = 0.001
    for i in range(5):
        weight = param_mapping["import/color{}/filter:0".format(i)]
        gamma = param_mapping["import/color{0}/color{0}_bn/gamma:0".format(i)]
        beta = param_mapping["import/color{0}/color{0}_bn/beta:0".format(i)]
        mean = param_mapping["import/color{0}/color{0}_bn/mean:0".format(i)]
        var = param_mapping["import/color{0}/color{0}_bn/variance:0".format(i)]
        a = gamma / np.sqrt(var + exp)
        bias = -a * mean
        weight = a * weight
        param_mapping["import/color{}/filter:0".format(i)] = weight
        param_mapping["import/color{}/biases:0".format(i)] = bias

    for t, r in param_mapping.items():
        name = '_'.join(t.split(":")[0].split("/")[1:])
        
        if 'filter' in t:
            r = np.transpose(r, [3, 0, 1, 2])

        with open('weights/' + name, 'wb') as f:
            r.tofile(f)
            print("{} with shape {} write finished.".format(name, np.shape(r)))
