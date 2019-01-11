#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
import argparse
import scipy
import numpy as np
import shutil
import time
from glob import glob
from datetime import datetime
from distutils.version import LooseVersion
from moviepy.editor import VideoFileClip
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class Dataset:
    """ Class aimed to store information relevant to a particular dataset."""

    def __init__(self, name, num_classes, image_shape, archive_name, training_dir_name, testing_dir_name,
                 yandex_disk_url) -> None:
        self.name = name
        self.num_classes = num_classes
        self.image_shape = image_shape

        dataset_dir_name = archive_name.split('.')[0]

        self.data_root_dir = 'data'
        self.runs_root_dir = 'runs'
        self.models_root_dir = 'models'

        self.archive_path = os.path.join(self.data_root_dir, archive_name)

        self.data_dir = os.path.join(self.data_root_dir, dataset_dir_name)
        self.runs_dir = os.path.join(self.runs_root_dir, dataset_dir_name)
        self.models_dir = os.path.join(self.models_root_dir, dataset_dir_name)

        self.data_training_dir = os.path.join(self.data_dir, training_dir_name)
        self.data_testing_dir = os.path.join(self.data_dir, testing_dir_name)

        self.runs_training_dir = os.path.join(self.runs_dir, training_dir_name)
        self.runs_testing_dir = os.path.join(self.runs_dir, testing_dir_name)

        self.yandex_disk_url = yandex_disk_url


DATASETS = {
    'kitti_road': Dataset('kitti_road', 2, (160, 576), 'data_road.zip', 'training', 'testing',
                          'https://yadi.sk/d/-je27MB90LootQ'),
    'cityscapes': None    # TODO: implement logic for Cityscapes dataset
}


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
    # this approach was mentioned in
    # https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf
    vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name="layer3_out_scaled")
    vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name="layer4_out_scaled")

    # 1x1 convolution for layer 7
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out,        num_classes, (1, 1), strides=(1, 1), padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='conv_1x1_layer7')

    # transposed convolution for conv_1x1_layer7
    upsample_7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, (4, 4), strides=(2, 2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='upsample_7')

    # 1x1 convolution for scaled layer 4
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out_scaled, num_classes, (1, 1), strides=(1, 1), padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='conv_1x1_layer4')

    # fuse upsample_7 and conv_1x1_layer4
    skip_4 = tf.add(upsample_7, conv_1x1_layer4, name='skip_4')

    # transposed convolution for skip_4
    upsample_4 = tf.layers.conv2d_transpose(skip_4, num_classes, (4, 4), strides=(2, 2), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='upsample_4')

    # 1x1 convolution for scaled layer 3
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out_scaled, num_classes, (1, 1), strides=(1, 1), padding='same',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                       name='conv_1x1_layer3')

    # fuse upsample_4 and conv_1x1_layer3
    skip_3 = tf.add(upsample_4, conv_1x1_layer3, name='skip_3')

    # transposed convolution for skip_3
    upsample_3 = tf.layers.conv2d_transpose(skip_3, num_classes, (16, 16), strides=(8, 8), padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                            name='upsample_3')

    return upsample_3


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFlow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss, softmax)
    """
    # no need to reshape the nn_last_layer to 2d, tf.nn.softmax_cross_entropy_with_logits
    # accepts tensors of any shape and will apply the softmax function on the last axis of the tensor
    logits = tf.identity(nn_last_layer, name='logits')
    labels = correct_label
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    softmax = tf.nn.softmax(logits, name='softmax')

    return logits, train_op, cross_entropy_loss, softmax


def save_model(sess, model_name_prefix, dataset, **kwargs):
    """
    Save session to file.
    :param sess: TF Session
    :param model_name_prefix: name prefix for model file
    :param dataset: Dataset object
    :param kwargs: key-value pairs used as a meta-information about session
    """
    saver = tf.train.Saver()

    output_dir = os.path.join(dataset.models_dir, model_name_prefix + '-' + str(datetime.isoformat(datetime.now())))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(datetime.now(), "Saving model to", output_dir)

    model_file = os.path.join(output_dir, model_name_prefix)

    info_file = os.path.join(output_dir, "info.txt")
    info_file_content = '\n'.join('%s = %s' % (k, v) for k, v in kwargs.items())

    saver.save(sess, model_file)

    with open(info_file, 'w') as f:
        f.write(info_file_content)


def restore_model(sess, model_path):
    """
    Restore session, which was previously saved to file.
    :param sess: TF Session
    :param model_path: path to several files having model_path as their prefix
    """
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)


def mean_iou(softmax, labels, num_classes):
    """
    Creates IoU metrics tensors.
    :param logits: output of the network tensor
    :param labels: ground truth tensor
    :param num_classes: number of classes in the dataset
    :return: a tensor representing the mean intersection-over-union
             and an operation that increments the confusion matrix
    """
    predictions_argmax = tf.argmax(softmax, axis=-1)
    labels_argmax = tf.argmax(labels, axis=-1)
    iou, iou_op = tf.metrics.mean_iou(labels_argmax, predictions_argmax, num_classes)
    return iou, iou_op


def train_nn(sess, dataset, epochs, save_model_freq, batch_size, learning_rate, keep_prob,
             get_batches_fn, train_op_tensor, cross_entropy_loss_tensor, input_image_tensor,
             correct_label_tensor, keep_prob_tensor, learning_rate_tensor, iou_tensor, iou_op_tensor):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param dataset: Dataset object
    :param epochs: Number of epochs
    :param save_model_freq: Frequency to save models with (each save_model_freq epoch)
    :param batch_size: Batch size
    :param learning_rate: Learning rate
    :param keep_prob: keep probability for dropout layers during training
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op_tensor: TF Operation to train the neural network
    :param cross_entropy_loss_tensor: TF Tensor for the amount of loss
    :param input_image_tensor: TF Placeholder for input images
    :param correct_label_tensor: TF Placeholder for label images
    :param keep_prob_tensor: TF Placeholder for dropout keep probability
    :param learning_rate_tensor: TF Placeholder for learning rate
    :param iou_tensor: tensor representing the mean intersection-over-union
    :param iou_op_tensor: operation that increments the confusion matrix
    """
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(datetime.now(), "Started training with", epochs, "epochs,", batch_size, "batch size, {:.6f} learning rate."
                          .format(learning_rate))
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        train_mean_iou = 0.0
        print(datetime.now(), "\n===== Beg Epoch:", epoch, "=====")
        num_images = 0
        for images, gt_images in get_batches_fn(batch_size):
            _, batch_loss, _ = \
                sess.run([train_op_tensor, cross_entropy_loss_tensor, iou_op_tensor],
                         feed_dict={keep_prob_tensor: keep_prob,
                                    correct_label_tensor: gt_images,
                                    input_image_tensor: images,
                                    learning_rate_tensor: learning_rate})
            batch_iou = sess.run(iou_tensor)
            train_mean_iou += batch_iou * len(images)
            train_loss += batch_loss * len(images)
            num_images += len(images)
            print(datetime.now(), "Batch size: {}; Batch train loss: {:.6f}; Batch train mean IoU: {:.6f}"
                                  .format(len(images), batch_loss, batch_iou))
        train_loss /= num_images
        train_mean_iou /= num_images
        print(datetime.now(), "Train loss: {:.6f}; Train mean IoU: {:.6f}".format(train_loss, train_mean_iou))
        print(datetime.now(), "\n===== End Epoch:", epoch, "=====")

        if epoch % save_model_freq == 0:
            save_model(sess, 'fcn8-interim', dataset,
                       epoch=epoch, batch_size=batch_size, learning_rate=learning_rate,
                       keep_prob=keep_prob, train_loss=train_loss, train_mean_iou=train_mean_iou)

    print(datetime.now(), "Finished training.")


def train(epochs: int = None, save_model_freq: int = None, batch_size: int = None, learning_rate: float = None,
          keep_prob: float = None, dataset: str = None):
    """
    Performs the FCN training from begining to end, that is, downloads required datasets and pretrained models,
    constructs the FNC architecture, trains it, and saves the trained model.
    :param epochs: number of epochs for training
    :param save_model_freq: save model each save_model_freq epoch
    :param batch_size: batch size for training
    :param learning_rate: learning rate for training
    :param keep_prob: keep probability for dropout layers for training
    :param dataset: dataset name
    """
    if None in [epochs, save_model_freq, batch_size, learning_rate, keep_prob, dataset]:
        raise ValueError('some parameters were not specified for function "%s"' % train.__name__)

    dataset = DATASETS[dataset]

    # Download Kitti Road dataset
    helper.maybe_download_dataset_from_yandex_disk(dataset)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg_from_yandex_disk(dataset.data_root_dir)

    # Run tests to check that environment is ready to execute the semantic segmentation pipeline
    if dataset.name == 'kitti_road':
        tests.test_for_kitti_dataset(dataset.data_root_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn, dataset)

    # TODO: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    #  https://www.cityscapes-dataset.com/

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Path to vgg model
        vgg_path = os.path.join(dataset.data_root_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(dataset.data_training_dir, dataset.image_shape)

        # TODO: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        image_input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = \
            load_vgg(sess, vgg_path)
        output_layer_tensor = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor, dataset.num_classes)
        correct_label_tensor = tf.placeholder(tf.float32, (None, None, None, dataset.num_classes))
        learning_rate_tensor = tf.placeholder(tf.float32)
        logits_tensor, train_op_tensor, cross_entropy_loss_tensor, softmax_tensor = \
            optimize(output_layer_tensor, correct_label_tensor, learning_rate_tensor, dataset.num_classes)

        iou_tensor, iou_op_tensor = mean_iou(softmax_tensor, correct_label_tensor, dataset.num_classes)

        train_nn(sess, dataset, epochs, save_model_freq, batch_size, learning_rate, keep_prob,
                 get_batches_fn, train_op_tensor, cross_entropy_loss_tensor, image_input_tensor, correct_label_tensor,
                 keep_prob_tensor, learning_rate_tensor, iou_tensor, iou_op_tensor)

        save_model(sess, 'fcn8-final', dataset,
                   epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, keep_prob=learning_rate)


def get_process_image_function(sess, dataset, softmax, keep_prob, image_input):
    def process_image_fn(image):
        orig_shape = image.shape
        image = scipy.misc.imresize(image, dataset.image_shape)
        # Run inference
        im_softmax = sess.run([softmax], {keep_prob: 1.0, image_input: [image]})

        # reshape output back to image_shape
        im_softmax = im_softmax[0][0][:, :, 1]

        # If road softmax > 0.5, prediction is road
        segmentation = (im_softmax > 0.5).reshape(dataset.image_shape[0], dataset.image_shape[1], 1)
        # Create mask based on segmentation to apply to original image
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        return scipy.misc.imresize(np.array(street_im), orig_shape)
    return process_image_fn


def infer(source_type: str = None, source: str = None, model: str = None, dataset: str = None):
    """
    Performs semantic segmentation for the given input using the pre-trained FCN model.
    :param source_type: type of source
    :param source: path to source file or directory in the local file system
    :param model: path to pre-trained model in the local file system
    :param dataset: dataset name, on which the model was trained
    """
    if None in [source_type, source, model, dataset]:
        raise ValueError('some parameters were not specified for function "%s"' % infer.__name__)

    if not os.path.exists(source):
        raise FileNotFoundError(source_type + ' ' + source + ' does not exist')

    # get known Dataset object
    dataset = DATASETS[dataset]

    # create a new tensorflow session
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        restore_model(sess, model)

        # extract some tensors
        keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        image_input = sess.graph.get_tensor_by_name("image_input:0")
        softmax = sess.graph.get_tensor_by_name('softmax:0')
        # get function that applies inference on a single image
        process_image_fn = get_process_image_function(sess, dataset, softmax, keep_prob, image_input)

        # prepare output directory
        output_dir = os.path.join(dataset.runs_dir, str(datetime.isoformat(datetime.now())))
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # extract file's basename
        source_basename = os.path.basename(source)

        # output file name (for image and video source types)
        output_file = os.path.join(output_dir, source_basename)

        # based on source type, process the source
        if source_type == 'image_directory':
            for image_file in glob(os.path.join(source, '*.png')):
                print(datetime.now(), "Processing", image_file, "file. ", end='')
                start_t = time.time()
                result_image = process_image_fn(scipy.misc.imread(image_file))
                end_t = time.time()
                print("Processed time: {:.6f} sec. ".format(end_t - start_t), end='')
                image_file_basename = os.path.basename(image_file)
                image_output_file = os.path.join(output_dir, image_file_basename)
                scipy.misc.imsave(image_output_file, result_image)
                print("Result saved to %s." % image_output_file)

        elif source_type == 'video':
            print(datetime.now(), "Processing", source, "file.", )
            clip = VideoFileClip(source)
            start_t = time.time()
            result_clip = clip.fl_image(process_image_fn)
            # write processed video into file
            result_clip.write_videofile(output_file)
            end_t = time.time()
            print(datetime.now(), "Processed time: {:.6f} sec. ".format(end_t - start_t), end='')
            print("Result saved to %s." % output_file)

        elif source_type == 'image':
            print(datetime.now(), "Processing", source, "file. ", end='')
            start_t = time.time()
            result_image = process_image_fn(scipy.misc.imread(source))
            end_t = time.time()
            print("Processed time: {:.6f} sec. ".format(end_t - start_t), end='')
            scipy.misc.imsave(output_file, result_image)
            print("Result saved to %s." % output_file)

        else:
            raise ValueError("unknown value for attribute source_type: %s" % source_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='With this script you can train and apply the trained model '
                    'to perform semantic segmentation on images and videos.',
        add_help=True
    )
    subparsers = parser.add_subparsers(dest='command', title='commands')

    # train parser
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', action='store', type=int, default=200,
                              help='number of epochs for training (default: %(default)s)')
    train_parser.add_argument('--save_model_freq', action='store', type=int, default=10,
                              help='frequency of saving model during training (default: %(default)s)')
    train_parser.add_argument('--batch_size', action='store', metavar='BATCH_SIZE', type=int, default=14,
                              help='number of images to feed to the network simultaneously (default: %(default)s)')
    train_parser.add_argument('--learning_rate', action='store', metavar='LEARNING_RATE',
                              type=float,  default=0.000005,
                              help='learning rate to be used during training (default: %(default)s)')
    train_parser.add_argument('--keep_prob', action='store', metavar='KEEP_PROB', type=float, default=0.5,
                              help='probability of keeping the connection between '
                                   'network nodes for dropout layers (default: %(default)s)')
    train_parser.add_argument('--dataset', action='store',
                              choices=['kitti_road', 'cityscapes'], type=str, default='kitti_road',
                              help='dataset to be used for training (default: %(default)s)')

    # infer parser
    infer_parser = subparsers.add_parser('infer')
    infer_parser.add_argument('--source_type', action='store', type=str, required=True,
                              choices=['video', 'image', 'image_directory'],
                              help='type of input to apply inference on')
    infer_parser.add_argument('--source', action='store', type=str, required=True,
                              help='path to input to apply inference on')
    infer_parser.add_argument('--model', action='store', type=str, required=True,
                              help='path to model to use for inference')
    infer_parser.add_argument('--dataset', action='store',
                              choices=['kitti_road', 'cityscapes'], type=str, default='kitti_road',
                              help='dataset to be used to retrieve some parameters (default: %(default)s)')

    parsed_args = parser.parse_args()

    # get the function to execute and its arguments
    command = globals()[parsed_args.command]
    command_args = vars(parsed_args)
    command_args.pop('command')

    # run function with its arguments
    command(**command_args)
