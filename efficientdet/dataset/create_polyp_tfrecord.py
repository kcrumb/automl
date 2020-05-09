# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert CSV dataset to TFRecord.

Example usage:
    python create_polyp_tfrecord.py \
        --data_file=/polyp-datasets\train_polyp_bb.csv  \
        --base_dir=/polyp-datasets \
        --output_path=/polyp-datasets/annotations/polyp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os

import PIL.Image
import tensorflow.compat.v1 as tf

from dataset import tfrecord_util

flags = tf.app.flags
flags.DEFINE_string('data_file', '', 'CSV file of polyp dataset.')
flags.DEFINE_string('base_dir', '', 'Base dir of the images.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord and json.')
flags.DEFINE_string('label_map_json_path', None, 'Path to label map json file with a dictionary.')
flags.DEFINE_integer('num_shards', 100, 'Number of shards for output file.')
FLAGS = flags.FLAGS

polyp_label_map_dict = {
    'polyp': 0
}


def dict_to_tf_example(img_path: str,
                       base_dir: str,
                       bboxes: list,
                       label_map_dict
                       ):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      img_path: The path (or relative to base dir) of the image.
      base_dir: The directory where all data sets are stored.
      bboxes: A list of all bounding boxes in that image.
      label_map_dict: A map from string label names to integers ids.

    Returns:
      example: The converted tf.Example.
    """

    full_path = os.path.join(base_dir, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_img = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_img)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size
    key = hashlib.sha256(encoded_img).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for bb in bboxes:
        box = bb.split(',')
        xmin.append(float(box[0]) / width)
        ymin.append(float(box[1]) / height)
        xmax.append(float(box[2]) / width)
        ymax.append(float(box[3]) / height)
        classes_text.append(box[4].encode('utf8'))
        classes.append(label_map_dict[box[4]])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tfrecord_util.int64_feature(height),
        'image/width': tfrecord_util.int64_feature(width),
        'image/filename': tfrecord_util.bytes_feature(img_path.encode('utf8')),
        'image/key/sha256': tfrecord_util.bytes_feature(key.encode('utf8')),
        'image/encoded': tfrecord_util.bytes_feature(encoded_img),
        'image/object/bbox/xmin': tfrecord_util.float_list_feature(xmin),
        'image/object/bbox/xmax': tfrecord_util.float_list_feature(xmax),
        'image/object/bbox/ymin': tfrecord_util.float_list_feature(ymin),
        'image/object/bbox/ymax': tfrecord_util.float_list_feature(ymax),
        'image/object/class/text': tfrecord_util.bytes_list_feature(classes_text),
        'image/object/class/label': tfrecord_util.int64_list_feature(classes)
    }))
    return example


def read_file_csv(file: str) -> dict:
    logging.info('reading CSV file')
    img_anno = {}
    with open(file, mode='r') as f:
        lines = f.read().split('\n')
    line_split = [l.split(',', maxsplit=1) for l in lines if l]
    for line in line_split:
        key_img = line[0]

        if key_img in img_anno:
            img = img_anno[key_img]
            img_anno[key_img] = img + line[1:]
        else:
            img_anno[key_img] = line[1:]
    return img_anno


def main(_):
    if not FLAGS.output_path:
        raise ValueError('output_path cannot be empty.')

    data_file = FLAGS.data_file
    img_anno = read_file_csv(data_file)

    logging.info('writing to output path: %s', FLAGS.output_path)
    writers = [
        tf.python_io.TFRecordWriter(
            FLAGS.output_path + '-%05d-of-%05d.tfrecord' % (i, FLAGS.num_shards))
        for i in range(FLAGS.num_shards)
    ]

    if FLAGS.label_map_json_path:
        with tf.io.gfile.GFile(FLAGS.label_map_json_path, 'rb') as f:
            label_map_dict = json.load(f)
    else:
        label_map_dict = polyp_label_map_dict

    for idx, img in enumerate(img_anno):
        tf_example = dict_to_tf_example(img_path=img,
                                        base_dir=FLAGS.base_dir,
                                        bboxes=img_anno[img],
                                        label_map_dict=label_map_dict
                                        )
        writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

    for writer in writers:
        writer.close()


if __name__ == '__main__':
    tf.app.run()
