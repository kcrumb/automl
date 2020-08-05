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
"""
A simple Grad-CAM script visualize the gradient of the last layer EfficientNet backbone.
Base on https://arxiv.org/abs/1610.02391
"""

import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf

import hparams_config
import inference
import utils
from keras import efficientdet_keras

import matplotlib.cm as cm

flags.DEFINE_string('image_path', None, 'Location of test image.')
flags.DEFINE_string('output_dir', None, 'Directory of annotated output images.')
flags.DEFINE_string('model_dir', None, 'Location of the checkpoint to run.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Model name to use.')
flags.DEFINE_string('hparams', '', 'Comma separated k=v pairs or a yaml file')
flags.DEFINE_bool('debug', False, 'If true, run function in eager for debug.')
flags.DEFINE_enum('gradient_type', 'cls', ['cls', 'box'], 'Gradient that should be visualized')
FLAGS = flags.FLAGS


def main(_):
  img = Image.open(FLAGS.image_path)
  imgs = [np.array(img)]
  # Create model config.
  config = hparams_config.get_efficientdet_config(FLAGS.model_name)
  config.is_training_bn = False
  # config.image_size = '640x640'
  config.nms_configs.score_thresh = 0.01
  config.nms_configs.max_output_size = 100
  config.override(FLAGS.hparams)

  # Use 'mixed_float16' if running on GPUs.
  policy = tf.keras.mixed_precision.experimental.Policy('float32')
  tf.keras.mixed_precision.experimental.set_policy(policy)
  tf.config.experimental_run_functions_eagerly(FLAGS.debug)

  # Create model
  model = efficientdet_keras.EfficientDetNet(config=config)
  target_size = utils.parse_image_size(config.image_size)
  target_size = target_size + (3,)
  model_inputs = tf.keras.Input(shape=target_size)
  model(model_inputs, False)
  model.summary()

  # output layers detailed
  # for i in model.layers:
  #   print(i.name, i.input, i.output)

  model.load_weights(tf.train.latest_checkpoint(FLAGS.model_dir))

  # create new model to access intermediate layers
  effdet_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(name='class_net').output,
                                                              model.get_layer(name='box_net').output,
                                                              model.backbone.layers[-3].output  # last layer
                                                              ])

  # is only used for pre- and post-processing methods
  effdet = efficientdet_keras.EfficientDetModel(config=config)

  # input image preprocessing
  inputs, scales = effdet._preprocessing(imgs, config.image_size, 'infer')

  with tf.GradientTape() as tape:
    # Compute activations of the last conv layer and make the tape watch it
    cls_output, box_output, efficientnet_last_layer = effdet_model(inputs, False)

  # save gradients
  grads = None
  if FLAGS.gradient_type == 'cls':
    grads = tape.gradient(cls_output, efficientnet_last_layer)
  elif FLAGS.gradient_type == 'box':
    grads = tape.gradient(box_output, efficientnet_last_layer)

  assert grads != None
  grad_cam(grads, efficientnet_last_layer[0], img, imgs[0], FLAGS.gradient_type)


  ### bounding box visualization ###
  boxes, scores, classes, valid_len = effdet._postprocess(cls_output, box_output, scales)

  # Visualize results.
  for i, img in enumerate(imgs):
    length = valid_len[i]
    img = inference.visualize_image(
        img,
        boxes[i].numpy()[:length],
        classes[i].numpy().astype(np.int)[:length],
        scores[i].numpy()[:length],
        min_score_thresh=config.nms_configs.score_thresh,
        max_boxes_to_draw=config.nms_configs.max_output_size)
    output_image_path = os.path.join(FLAGS.output_dir, str(i) + '.jpg')
    Image.fromarray(img).save(output_image_path)
    print('writing annotated image to ', output_image_path)


def grad_cam(grads, last_layer, input_img: Image, img, type: str):
  ### calculate grad-cam ###
  # neuron importance weights a^c_k
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  # liner combination
  weighted_features = tf.multiply(pooled_grads, last_layer)
  sumed_weighted_features = tf.reduce_sum(weighted_features, axis=-1)
  # relu operation
  heatmap = tf.maximum(sumed_weighted_features, 0)

  ### heatmap visualization ###
  # normalize
  heatmap = heatmap.numpy()
  heatmap = heatmap / np.max(heatmap)
  # rescale heatmap to a range 0-255
  heatmap = np.uint(255 * heatmap)
  # get color map
  cmap = cm.get_cmap("jet")
  # get colormap colors
  cmap_color = cmap(np.arange(cmap.N))
  # set alpha (zeros are transparent)
  alpha_channel = np.ones((cmap.N)) - 0.5
  alpha_channel[0] = 0
  cmap_color[:, -1] = alpha_channel
  # map heatmap values to RGBA value
  heatmap_colored = cmap_color[heatmap]
  # resize heatmap
  heatmap_img = tf.keras.preprocessing.image.array_to_img(heatmap_colored)
  heatmap_img = heatmap_img.resize((img.shape[1], img.shape[0]))
  # overlay heatmap and input image
  in_img = input_img.copy()
  in_img.paste(im=heatmap_img, box=(0, 0), mask=heatmap_img)

  # output image path
  img_name = FLAGS.image_path.replace('\\', '/')
  img_name = img_name[img_name.rindex('/') + 1:]
  img_name = img_name[:img_name.rindex('.')] + '-' + type + img_name[img_name.rindex('.'):]
  output_image_path = os.path.join(FLAGS.output_dir, img_name)

  in_img.save(output_image_path)
  print('Writing Grad-CAM image ({}) to {}'.format(type, output_image_path))


if __name__ == '__main__':
  flags.mark_flag_as_required('image_path')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('model_dir')
  logging.set_verbosity(logging.WARNING)
  app.run(main)
