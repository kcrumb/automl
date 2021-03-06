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
    'background': 0, 'polyp': 1
}

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.


def get_image_id(filename):
    """Convert a string to a integer."""
    del filename
    global GLOBAL_IMG_ID
    GLOBAL_IMG_ID += 1
    return GLOBAL_IMG_ID


def get_ann_id():
    """Return unique annotation id across images."""
    global GLOBAL_ANN_ID
    GLOBAL_ANN_ID += 1
    return GLOBAL_ANN_ID


def dict_to_tf_example(img_path: str,
                       base_dir: str,
                       bboxes: list,
                       label_map_dict,
                       ann_json_dict=None
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

    image_id = get_image_id(img_path)

    if ann_json_dict:
        image = {
            'file_name': img_path,
            'height': height,
            'width': width,
            'id': image_id,
        }
        ann_json_dict['images'].append(image)

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

        if ann_json_dict:
            abs_xmin = int(box[0])
            abs_ymin = int(box[1])
            abs_xmax = int(box[2])
            abs_ymax = int(box[3])
            abs_width = abs_xmax - abs_xmin
            abs_height = abs_ymax - abs_ymin
            ann = {
                'area': abs_width * abs_height,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                'category_id': label_map_dict[box[4]],
                'id': get_ann_id(),
                'ignore': 0,
                'segmentation': [],
            }
            ann_json_dict['annotations'].append(ann)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tfrecord_util.int64_feature(height),
        'image/width': tfrecord_util.int64_feature(width),
        'image/filename': tfrecord_util.bytes_feature(img_path.encode('utf8')),
        'image/source_id': tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
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
    line_split_sorted = sorted(line_split, key=lambda x: int(x[0][x[0].rindex('/')+1 : x[0].rindex('.')]))
    for line in line_split_sorted:
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

    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }

    for class_name, class_id in label_map_dict.items():
      cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(cls)

    for idx, img in enumerate(img_anno):
        tf_example = dict_to_tf_example(img_path=img,
                                        base_dir=FLAGS.base_dir,
                                        bboxes=img_anno[img],
                                        label_map_dict=label_map_dict,
                                        ann_json_dict=ann_json_dict
                                        )
        writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

    for writer in writers:
        writer.close()

    json_file_path = os.path.join(
        os.path.dirname(FLAGS.output_path),
        'json_' + os.path.basename(FLAGS.output_path) + '.json')
    with tf.io.gfile.GFile(json_file_path, 'w') as f:
        json.dump(ann_json_dict, f)


if __name__ == '__main__':
    tf.app.run()
