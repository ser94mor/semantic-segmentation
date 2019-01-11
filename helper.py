"""
You should not edit helper.py as part of your submission.

This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
"""

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
from datetime import datetime
import urllib.request
import json
import tensorflow as tf
from glob import glob
from tqdm import tqdm


class DLProgress(tqdm):
    """
    Report download progress to the terminal.
    :param tqdm: Information fed to the tqdm library to estimate progress.
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        Store necessary information for tracking progress.
        :param block_num: current block of the download
        :param block_size: size of current block
        :param total_size: total download size, if known
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)  # Updates progress
        self.last_block = block_num


def retrieve_direct_yandex_disk_url(url):
    """
    The links Yandex.Disk provides after file sharing are not direct. To obtain a direct link, a special procedure
    should be performed.
    :param url: public non-direct link to file
    :return: the direct link to file
    """
    # getting the direct link
    request_direct_link = \
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}" \
        .format(url)
    with urllib.request.urlopen(request_direct_link) as resp:
        direct_link = json.loads(resp.read().decode('utf-8'))['href']
    return direct_link


def download_file(direct_url, file):
    """
    Downloads files.
    :param direct_url: direct link to file to download.
    :param file: file name in the local file system
    """
    print(datetime.now(),
          "Started downloading", file, "file using", direct_url, "link.")
    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
        urllib.request.urlretrieve(direct_url, file, pbar.hook)
    print(datetime.now(), "Finished downloading", file, "file.")


def extract_zip_archive(archive, directory):
    """
    Extracts ZIP archive.
    :param archive: archive path in the local file system
    :param directory: directory into which extract the archive
    """
    print(datetime.now(), "Started extracting", archive, "file.")
    with zipfile.ZipFile(archive, 'r') as zip_file:
        zip_file.extractall(directory)
    print(datetime.now(), "Finished extracting", archive, "file.")


def maybe_download_dataset_from_yandex_disk(dataset):
    """
    Downloads the selected dataset from Yandex.Disk.
    :param dataset: Dataset object
    """
    # download dataset if needed from Yandex.Disk
    if not (os.path.isdir(dataset.data_dir)):
        direct_link = retrieve_direct_yandex_disk_url(dataset.yandex_disk_url)
        download_file(direct_link, dataset.archive_path)
        extract_zip_archive(dataset.archive_path, dataset.data_root_dir)
        # Remove zip file to save space
        os.remove(dataset.archive_path)


def maybe_download_pretrained_vgg_from_yandex_disk(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    yandex_disk_url = "https://yadi.sk/d/aTzGJtuzdtYE3Q"
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_filename = os.path.join(data_dir, 'vgg.zip')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)

        direct_link = retrieve_direct_yandex_disk_url(yandex_disk_url)
        download_file(direct_link, vgg_filename)
        extract_zip_archive(vgg_filename, data_dir)
        # Remove zip file to save space
        os.remove(vgg_filename)


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        # Grab image and label paths
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        # Shuffle training data
        random.shuffle(image_paths)
        # Loop through batches and grab images, yielding each batch
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
                # Re-size to image_shape
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                # Create "one-hot-like" labels by class
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn
