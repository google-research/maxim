# Copyright 2022 Google LLC.
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

"""Run evaluation."""

import collections
import importlib
import io
import os

from absl import app
from absl import flags
import flax
import jax.numpy as jnp
import ml_collections
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'task', 'Denoising',
    ['Denoising', 'Deblurring', 'Deraining', 'Dehazing', 'Enhancement'],
    'Task to run.')
flags.DEFINE_string('ckpt_path', '', 'Path to checkpoint.')
flags.DEFINE_string('input_dir', '', 'Input dir to the test set.')
flags.DEFINE_string('output_dir', '', 'Output dir to store predicted images.')
flags.DEFINE_boolean('has_target', True, 'Whether has corresponding gt image.')
flags.DEFINE_boolean('save_images', True, 'Dump predicted images.')
flags.DEFINE_boolean('geometric_ensemble', False,
                     'Whether use ensemble infernce.')

_MODEL_FILENAME = 'maxim'

_MODEL_VARIANT_DICT = {
    'Denoising': 'S-3',
    'Deblurring': 'S-3',
    'Deraining': 'S-2',
    'Dehazing': 'S-2',
    'Enhancement': 'S-2',
}

_MODEL_CONFIGS = {
    'variant': '',
    'dropout_rate': 0.0,
    'num_outputs': 3,
    'use_bias': True,
    'num_supervision_scales': 3,
}


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.
  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.
  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def mod_padding_symmetric(image, factor=64):
  """Padding the image to be divided by factor."""
  height, width = image.shape[0], image.shape[1]
  height_pad, width_pad = ((height + factor) // factor) * factor, (
      (width + factor) // factor) * factor
  padh = height_pad - height if height % factor != 0 else 0
  padw = width_pad - width if width % factor != 0 else 0
  image = jnp.pad(
      image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)],
      mode='reflect')
  return image


def get_params(ckpt_path):
  """Get params checkpoint."""

  with tf.io.gfile.GFile(ckpt_path, 'rb') as f:
    data = f.read()
  values = np.load(io.BytesIO(data))
  params = recover_tree(*zip(*values.items()))
  params = params['opt']['target']

  return params


def calculate_psnr(img1, img2, crop_border, test_y_channel=False):
  """Calculate PSNR (Peak Signal-to-Noise Ratio).

  Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
  Args:
    img1 (ndarray): Images with range [0, 255].
    img2 (ndarray): Images with range [0, 255].
    crop_border (int): Cropped pixels in each edge of an image. These
        pixels are not involved in the PSNR calculation.
    test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
  Returns:
    float: psnr result.
  """
  assert img1.shape == img2.shape, (
      f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)

  if crop_border != 0:
    img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
    img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

  if test_y_channel:
    img1 = to_y_channel(img1)
    img2 = to_y_channel(img2)

  mse = np.mean((img1 - img2)**2)
  if mse == 0:
    return float('inf')
  return 20. * np.log10(255. / np.sqrt(mse))


def _convert_input_type_range(img):
  """Convert the type and range of the input image.

  It converts the input image to np.float32 type and range of [0, 1].
  It is mainly used for pre-processing the input image in colorspace
  convertion functions such as rgb2ycbcr and ycbcr2rgb.
  Args:
    img (ndarray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float32 type with range [0, 1].
  Returns:
      (ndarray): The converted image with type of np.float32 and range of
          [0, 1].
  """
  img_type = img.dtype
  img = img.astype(np.float32)
  if img_type == np.float32:
    pass
  elif img_type == np.uint8:
    img /= 255.
  else:
    raise TypeError('The img type should be np.float32 or np.uint8, '
                    f'but got {img_type}')
  return img


def _convert_output_type_range(img, dst_type):
  """Convert the type and range of the image according to dst_type.

  It converts the image to desired type and range. If `dst_type` is np.uint8,
  images will be converted to np.uint8 type with range [0, 255]. If
  `dst_type` is np.float32, it converts the image to np.float32 type with
  range [0, 1].
  It is mainly used for post-processing images in colorspace convertion
  functions such as rgb2ycbcr and ycbcr2rgb.
  Args:
    img (ndarray): The image to be converted with np.float32 type and
        range [0, 255].
    dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
        converts the image to np.uint8 type with range [0, 255]. If
        dst_type is np.float32, it converts the image to np.float32 type
        with range [0, 1].
  Returns:
    (ndarray): The converted image with desired type and range.
  """
  if dst_type not in (np.uint8, np.float32):
    raise TypeError('The dst_type should be np.float32 or np.uint8, '
                    f'but got {dst_type}')
  if dst_type == np.uint8:
    img = img.round()
  else:
    img /= 255.

  return img.astype(dst_type)


def rgb2ycbcr(img, y_only=False):
  """Convert a RGB image to YCbCr image.

  This function produces the same results as Matlab's `rgb2ycbcr` function.
  It implements the ITU-R BT.601 conversion for standard-definition
  television. See more details in
  https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
  It differs from a similar function in cv2.cvtColor: `RGB <-> YCrCb`.
  In OpenCV, it implements a JPEG conversion. See more details in
  https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

  Args:
    img (ndarray): The input image. It accepts:
        1. np.uint8 type with range [0, 255];
        2. np.float32 type with range [0, 1].
    y_only (bool): Whether to only return Y channel. Default: False.
  Returns:
    ndarray: The converted YCbCr image. The output image has the same type
        and range as input image.
  """
  img_type = img.dtype
  img = _convert_input_type_range(img)
  if y_only:
    out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
  else:
    out_img = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) + [16, 128, 128]
  out_img = _convert_output_type_range(out_img, img_type)
  return out_img


def to_y_channel(img):
  """Change to Y channel of YCbCr.

  Args:
    img (ndarray): Images with range [0, 255].
  Returns:
    (ndarray): Images with range [0, 255] (float type) without round.
  """
  img = img.astype(np.float32) / 255.
  if img.ndim == 3 and img.shape[2] == 3:
    img = rgb2ycbcr(img, y_only=True)
    img = img[..., None]
  return img * 255.


def augment_image(image, times=8):
  """Geometric augmentation."""
  if times == 4:  # only rotate image
    images = []
    for k in range(0, 4):
      images.append(np.rot90(image, k=k))
    images = np.stack(images, axis=0)
  elif times == 8:  # roate and flip image
    images = []
    for k in range(0, 4):
      images.append(np.rot90(image, k=k))
    image = np.fliplr(image)
    for k in range(0, 4):
      images.append(np.rot90(image, k=k))
    images = np.stack(images, axis=0)
  else:
    raise Exception(f'Error times: {times}')
  return images


def deaugment_image(images, times=8):
  """Reverse the geometric augmentation."""

  if times == 4:  # only rotate image
    image = []
    for k in range(0, 4):
      image.append(np.rot90(images[k], k=4-k))
    image = np.stack(image, axis=0)
    image = np.mean(image, axis=0)
  elif times == 8:  # roate and flip image
    image = []
    for k in range(0, 4):
      image.append(np.rot90(images[k], k=4-k))
    for k in range(0, 4):
      image.append(np.fliplr(np.rot90(images[4+k], k=4-k)))
    image = np.mean(image, axis=0)
  else:
    raise Exception(f'Error times: {times}')
  return image


def is_image_file(filename):
  """Check if it is an valid image file by extension."""
  return any(
      filename.endswith(extension)
      for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def save_img(img, pth):
  """Save an image to disk.

  Args:
    img: jnp.ndarry, [height, width, channels], img will be clipped to [0, 1]
      before saved to pth.
    pth: string, path to save the image to.
  """
  Image.fromarray(np.array(
      (np.clip(img, 0., 1.) * 255.).astype(jnp.uint8))).save(pth, 'PNG')


def make_shape_even(image):
  """Pad the image to have even shapes."""
  height, width = image.shape[0], image.shape[1]
  padh = 1 if height % 2 != 0 else 0
  padw = 1 if width % 2 != 0 else 0
  image = jnp.pad(image, [(0, padh), (0, padw), (0, 0)], mode='reflect')
  return image


def main(_):
  params = get_params(FLAGS.ckpt_path)

  if FLAGS.save_images:
    os.makedirs(FLAGS.output_dir, exist_ok=True)

  # sorted is important for continuning an inference job.
  filepath = sorted(os.listdir(os.path.join(FLAGS.input_dir, 'input')))
  input_filenames = [
      os.path.join(FLAGS.input_dir, 'input', x)
      for x in filepath
      if is_image_file(x)
  ]
  if FLAGS.has_target:
    target_filenames = [
        os.path.join(FLAGS.input_dir, 'target', x)
        for x in filepath
        if is_image_file(x)
    ]
  num_images = len(input_filenames)

  model_mod = importlib.import_module(f'maxim.models.{_MODEL_FILENAME}')
  model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
  model_configs.variant = _MODEL_VARIANT_DICT[FLAGS.task]
  model = model_mod.Model(**model_configs)

  psnr_all = []

  def _process_file(i):
    print(f'Processing {i + 1} / {num_images}...')
    input_file = input_filenames[i]
    input_img = np.asarray(Image.open(input_file).convert('RGB'),
                           np.float32) / 255.
    if FLAGS.has_target:
      target_file = target_filenames[i]
      target_img = np.asarray(Image.open(target_file).convert('RGB'),
                              np.float32) / 255.

    # Padding images to have even shapes
    height, width = input_img.shape[0], input_img.shape[1]
    input_img = make_shape_even(input_img)
    height_even, width_even = input_img.shape[0], input_img.shape[1]

    # padding images to be multiplies of 64
    input_img = mod_padding_symmetric(input_img, factor=64)

    if FLAGS.geometric_ensemble:
      input_img = augment_image(input_img, FLAGS.ensemble_times)
    else:
      input_img = np.expand_dims(input_img, axis=0)

    # handle multi-stage outputs, obtain the last scale output of last stage
    preds = model.apply({'params': flax.core.freeze(params)}, input_img)
    if isinstance(preds, list):
      preds = preds[-1]
      if isinstance(preds, list):
        preds = preds[-1]

    # De-ensemble by averaging inferenced results.
    if FLAGS.geometric_ensemble:
      preds = deaugment_image(preds, FLAGS.ensemble_times)
    else:
      preds = np.array(preds[0], np.float32)

    # unpad images to get the original resolution
    new_height, new_width = preds.shape[0], preds.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width
    preds = preds[h_start:h_end, w_start:w_end, :]

    # print PSNR scores
    if FLAGS.has_target:
      psnr = calculate_psnr(
          target_img * 255., preds * 255., crop_border=0, test_y_channel=False)
      print(f'{i}th image: psnr = {psnr:.4f}')
    else:
      psnr = -1

    # save files
    basename = os.path.basename(input_file)
    if FLAGS.save_images:
      save_pth = os.path.join(FLAGS.output_dir, basename)
      save_img(preds, save_pth)

    return psnr

  for i in range(num_images):
    psnr = _process_file(i)
    psnr_all.append(psnr)

  psnr_all = np.asarray(psnr_all)

  print(f'average psnr = {np.sum(psnr_all)/num_images:.4f}')
  print(f'std psnr = {np.std(psnr_all):.4f}')


if __name__ == '__main__':
  app.run(main)
