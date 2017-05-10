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
    for t, r in zip(tensors, results):
        name = '_'.join(t.name.split(":")[0].split("/")[1:])
        
        if 'filter' in t.name:
            r = np.transpose(r, [3, 0, 1, 2])

        with open('weights/' + name, 'wb') as f:
            r.tofile(f)
            print("{} write finished.".format(name))
