'''
From https://github.com/tsc2017/Inception-Score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_inception_score(images, splits=10)
Args:
    images: A numpy array with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. A dtype of np.uint8 is recommended to save CPU memory.
    splits: The number of splits of the images, default is 10.
Returns:
    Mean and standard deviation of the Inception Score across the splits.
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
# pip install tensorflow-gan
import tensorflow_gan as tfgan
session=tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces TPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 1000
INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
FIRST_RUN=[True]
# Run images through Inception.
inception_images =[None] 
image_iterator_init=[None]
inception_size = 299
input_size=[32]
def inception_logits(images):
    images = tf.transpose(images, [0, 2, 3, 1])
    images = tf.compat.v1.image.resize_bilinear(images, [inception_size, inception_size])
    generated_images_list = array_ops.split(images, num_or_size_splits = 1)
    logits = tf.map_fn(
        fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_OUTPUT, True),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits

logits=[None]
def get_inception_probs(inps, session=None, strategy=None):
    if FIRST_RUN[0]:
        print('Running Inception for the first time, compiling...')
        with session.graph.as_default():
            inception_images[0]=tf.compat.v1.placeholder(tf.float32, [BATCH_SIZE, 3, input_size[0], input_size[0]], name = 'inception_images')
            image_dataset = tf.data.Dataset.from_tensor_slices((inception_images[0])).batch(BATCH_SIZE, drop_remainder=True)
            image_iterator = strategy.make_dataset_iterator(image_dataset)
            image_iterator_init[0] = image_iterator.initialize()
            logits[0]=tf.concat(strategy.experimental_run(inception_logits, image_iterator).values,0)
            FIRST_RUN[0]=False
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        session.run(image_iterator_init[0],{inception_images[0]: inp})
        preds[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(logits[0])[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10, session=None, strategy=None):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    input_size[0]=images.shape[3]
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time=time.time()
    preds = get_inception_probs(images, session, strategy)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.38 for 50000 CIFAR-10 training set images, or mean=11.31, std=0.10 if in 10 splits.
