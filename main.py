#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    image_input = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # scale layers 3 and 4 as it is done by the FCN paper authors in their recent update to the network architecture
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s-atonce/train.prototxt
    # this approch was mentioned in
    # https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name="layer3_out_scaled")
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name="layer4_out_scaled")

    # 1x1 convolution for layer 7
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out,        num_classes, (1, 1), strides=(1, 1), padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='conv_1x1_layer7')

    # transposed convolution for conv_1x1_layer7
    upsample_7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, (4, 4),strides=(2, 2), padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='upsample_7')

    # 1x1 convolution for scaled layer 4
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, (1, 1), strides=(1, 1), padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='conv_1x1_layer4')

    # fuse upsample_7 and conv_1x1_layer4
    skip_4 = tf.add(upsample_7, conv_1x1_layer4, name='skip_4')

    # transposed convolution for skip_4
    upsample_4 = tf.layers.conv2d_transpose(skip_4, num_classes, (4, 4), strides=(2, 2), padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='upsample_4')

    # 1x1 convolution for scaled layer 3
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, (1, 1), strides=(1, 1), padding='same',
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='conv_1x1_layer3')

    # fuse upsample_4 and conv_1x1_layer3
    skip_3 = tf.add(upsample_4, conv_1x1_layer3, name='skip_3')

    # transposed convolution for skip_3
    upsample_3 = tf.layers.conv2d_transpose(skip_3, num_classes, (16, 16), strides=(8, 8), padding='same',
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='upsample_3')

    return upsample_3


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    return None, None, None


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    pass


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'

    # Download Kitti Road dataset
    helper.maybe_download_kitti_road_dataset_from_yandex_disk(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Run tests to check that environment is ready to execute the semantic segmentation pipeline
    tests.test_for_kitti_dataset(data_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    #tests.test_train_nn(train_nn)


    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
