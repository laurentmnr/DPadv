import tensorflow as tf
import re
import os
from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from differential_privacy.dp_sgd.dp_optimizer import dp_pca
from differential_privacy.dp_sgd.dp_optimizer import sanitizer
from differential_privacy.dp_sgd.dp_optimizer import utils
from differential_privacy.privacy_accountant.tf import accountant
from dp_mnist import MnistInput


NUM_TRAINING_IMAGES = 60000
NUM_TESTING_IMAGES = 10000
IMAGE_SIZE = 28

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
network_parameters = utils.NetworkParameters()

# If the ASCII proto isn't specified, then construct a config protobuf based
import mnist
import numpy as np
train_images = mnist.train_images().reshape([-1,784])/255.
train_labels = mnist.train_labels()

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True


sess = tf.InteractiveSession(config=run_config)
load_model("/Users/laurent/Desktop/DP/models/")
images = tf.get_default_graph().get_tensor_by_name("images:0")
logits = tf.get_default_graph().get_tensor_by_name("logits:0")



grad= tf.concat([tf.reshape(tf.gradients(logits[:,i],images)[0],(1,-1,28,28)) for i in range(10)],axis=0)[train_labels[0]]
grad
g = -grad.eval(session=sess,feed_dict={images:train_images[:1,]})
image=train_images[0].reshape((-1,28,28))+0.03*np.sign(g)

import matplotlib.pyplot as plt


plt.imshow(image.reshape((28, 28)), cmap='gray')
