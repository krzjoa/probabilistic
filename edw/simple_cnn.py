import numpy as np
from scipy.ndimage import imread
import os
import tensorflow as tf

# Params
IMAGES = "/home/somepath/Pulpit/Datasets/carvana/train"
GT = "/home/somepath/Pulpit/Datasets/carvana/train_masks"


class DataLoader(object):

    def __init__(self, images, ground_truth):
        # Root directories
        self.img_root = images
        self.gt_root = ground_truth

        # Subdirectories
        self.images = os.listdir(images)
        self.ground_truth = os.listdir(ground_truth)

    def serve(self):
        for i, gt in zip(self.images, self.ground_truth):
            img = imread(os.path.join(self.img_root, i))
            img_mask = imread(os.path.join(self.gt_root, gt))
            yield img.astype(np.float32), img_mask.astype(np.float32)




class ConvNN(object):

    def __init__(self):
        # Placeholders
        self.input = tf.placeholder(np.float32, [None, 1280, 1918, 3])

        model = tf.layers.conv2d(self.input, 32, [3, 3])
        model = tf.layers.conv2d(model, 32, [3, 3])

        self.model = model

    def forward(self, X):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # init variables
            sess.run(init)

            # Compute output
            output = sess.run(self.model, feed_dict={self.input: X})

            return output


    def fit(self, X , y):

        # Loss function




        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # init variables
            sess.run(init)

            # Compute output
            output = sess.run(self.model, feed_dict={self.input: X})

            return output


if __name__ == '__main__':
    dataloader = DataLoader(IMAGES, GT)

    cnn = ConvNN()

    for i, gt in dataloader.serve():
        # print i.shape, gt.shape

        i = i[np.newaxis, :, :, :]

        print i.shape

        print cnn.forward(i).shape
