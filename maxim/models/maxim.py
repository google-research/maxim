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

"""Main file for the MAXIM model."""

import functools
from typing import Any, Sequence, Tuple

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp


Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))
Conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1))
ConvT_up = functools.partial(nn.ConvTranspose,
                             kernel_size=(2, 2),
                             strides=(2, 2))
Conv_down = functools.partial(nn.Conv,
                              kernel_size=(4, 4),
                              strides=(2, 2))

weight_initializer = nn.initializers.normal(stddev=2e-2)


class MlpBlock(nn.Module):
  """A 1-hidden-layer MLP block, applied over the last dimension."""
  mlp_dim: int
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, d = x.shape
    x = nn.Dense(self.mlp_dim, use_bias=self.use_bias,
                 kernel_init=weight_initializer)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn.Dense(d, use_bias=self.use_bias,
                 kernel_init=weight_initializer)(x)
    return x


def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


class UpSampleRatio(nn.Module):
  """Upsample features given a ratio > 0."""
  features: int
  ratio: float
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    n, h, w, c = x.shape
    x = jax.image.resize(
        x,
        shape=(n, int(h * self.ratio), int(w * self.ratio), c),
        method="bilinear")
    x = Conv1x1(features=self.features, use_bias=self.use_bias)(x)
    return x


class CALayer(nn.Module):
  """Squeeze-and-excitation block for channel attention.

  ref: https://arxiv.org/abs/1709.01507
  """
  features: int
  reduction: int = 4
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    # 2D global average pooling
    y = jnp.mean(x, axis=[1, 2], keepdims=True)
    # Squeeze (in Squeeze-Excitation)
    y = Conv1x1(self.features // self.reduction, use_bias=self.use_bias)(y)
    y = nn.relu(y)
    # Excitation (in Squeeze-Excitation)
    y = Conv1x1(self.features, use_bias=self.use_bias)(y)
    y = nn.sigmoid(y)
    return x * y


class RCAB(nn.Module):
  """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""
  features: int
  reduction: int = 4
  lrelu_slope: float = 0.2
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    shortcut = x
    x = nn.LayerNorm(name="LayerNorm")(x)
    x = Conv3x3(features=self.features, use_bias=self.use_bias, name="conv1")(x)
    x = nn.leaky_relu(x, negative_slope=self.lrelu_slope)
    x = Conv3x3(features=self.features, use_bias=self.use_bias, name="conv2")(x)
    x = CALayer(features=self.features, reduction=self.reduction,
                use_bias=self.use_bias, name="channel_attention")(x)
    return x + shortcut


class GridGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the second last.
  If applied on other dims, you should swapaxes first.
  """
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    u, v = jnp.split(x, 2, axis=-1)
    v = nn.LayerNorm(name="intermediate_layernorm")(v)
    n = x.shape[-3]   # get spatial dim
    v = jnp.swapaxes(v, -1, -3)
    v = nn.Dense(n, use_bias=self.use_bias, kernel_init=weight_initializer)(v)
    v = jnp.swapaxes(v, -1, -3)
    return u * (v + 1.)


class GridGmlpLayer(nn.Module):
  """Grid gMLP layer that performs global mixing of tokens."""
  grid_size: Sequence[int]
  use_bias: bool = True
  factor: int = 2
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, num_channels = x.shape
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = nn.Dense(num_channels * self.factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(y)
    y = nn.gelu(y)
    y = GridGatingUnit(use_bias=self.use_bias, name="GridGatingUnit")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x


class BlockGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.

  The 'spatial' dim is defined as the **second last**.
  If applied on other dims, you should swapaxes first.
  """
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    u, v = jnp.split(x, 2, axis=-1)
    v = nn.LayerNorm(name="intermediate_layernorm")(v)
    n = x.shape[-2]  # get spatial dim
    v = jnp.swapaxes(v, -1, -2)
    v = nn.Dense(n, use_bias=self.use_bias, kernel_init=weight_initializer)(v)
    v = jnp.swapaxes(v, -1, -2)
    return u * (v + 1.)


class BlockGmlpLayer(nn.Module):
  """Block gMLP layer that performs local mixing of tokens."""
  block_size: Sequence[int]
  use_bias: bool = True
  factor: int = 2
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    n, h, w, num_channels = x.shape
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    x = block_images_einops(x, patch_size=(fh, fw))
    # MLP2: Local (block) mixing part, provides within-block communication.
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = nn.Dense(num_channels * self.factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(y)
    y = nn.gelu(y)
    y = BlockGatingUnit(use_bias=self.use_bias, name="BlockGatingUnit")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):
  """The multi-axis gated MLP block."""
  block_size: Sequence[int]
  grid_size: Sequence[int]
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  use_bias: bool = True
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    shortcut = x
    n, h, w, num_channels = x.shape
    x = nn.LayerNorm(name="LayerNorm_in")(x)
    x = nn.Dense(num_channels * self.input_proj_factor, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="in_project")(x)
    x = nn.gelu(x)

    u, v = jnp.split(x, 2, axis=-1)
    # GridGMLPLayer
    u = GridGmlpLayer(
        grid_size=self.grid_size,
        factor=self.grid_gmlp_factor,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        name="GridGmlpLayer")(u, deterministic)

    # BlockGMLPLayer
    v = BlockGmlpLayer(
        block_size=self.block_size,
        factor=self.block_gmlp_factor,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        name="BlockGmlpLayer")(v, deterministic)

    x = jnp.concatenate([u, v], axis=-1)

    x = nn.Dense(num_channels, use_bias=self.use_bias,
                 kernel_init=weight_initializer, name="out_project")(x)
    x = nn.Dropout(self.dropout_rate)(x, deterministic)
    x = x + shortcut
    return x


class RDCAB(nn.Module):
  """Residual dense channel attention block. Used in Bottlenecks."""
  features: int
  reduction: int = 16
  use_bias: bool = True
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x, deterministic=True):
    y = nn.LayerNorm(name="LayerNorm")(x)
    y = MlpBlock(
        mlp_dim=self.features,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        name="channel_mixing")(
            y, deterministic=deterministic)
    y = CALayer(
        features=self.features,
        reduction=self.reduction,
        use_bias=self.use_bias,
        name="channel_attention")(
            y)
    x = x + y
    return x


class BottleneckBlock(nn.Module):
  """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  num_groups: int = 1
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic):
    """Applies the Mixer block to inputs."""
    assert x.ndim == 4  # Input has shape [batch, h, w, c]
    n, h, w, num_channels = x.shape

    # input projection
    x = Conv1x1(self.features, use_bias=self.use_bias, name="input_proj")(x)
    shortcut_long = x

    for i in range(self.num_groups):
      x = ResidualSplitHeadMultiAxisGmlpLayer(
          grid_size=self.grid_size,
          block_size=self.block_size,
          grid_gmlp_factor=self.grid_gmlp_factor,
          block_gmlp_factor=self.block_gmlp_factor,
          input_proj_factor=self.input_proj_factor,
          use_bias=self.use_bias,
          dropout_rate=self.dropout_rate,
          name=f"SplitHeadMultiAxisGmlpLayer_{i}")(x, deterministic)
      # Channel-mixing part, which provides within-patch communication.
      x = RDCAB(
          features=self.features,
          reduction=self.channels_reduction,
          use_bias=self.use_bias,
          name=f"channel_attention_block_1_{i}")(
              x)

    # long skip-connect
    x = x + shortcut_long
    return x


class UNetEncoderBlock(nn.Module):
  """Encoder block in MAXIM."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  num_groups: int = 1
  lrelu_slope: float = 0.2
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  dropout_rate: float = 0.0
  downsample: bool = True
  use_global_mlp: bool = True
  use_bias: bool = True
  use_cross_gating: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray, skip: jnp.ndarray = None,
               enc: jnp.ndarray = None, dec: jnp.ndarray = None, *,
               deterministic: bool = True) -> jnp.ndarray:
    if skip is not None:
      x = jnp.concatenate([x, skip], axis=-1)

    # convolution-in
    x = Conv1x1(self.features, use_bias=self.use_bias)(x)
    shortcut_long = x

    for i in range(self.num_groups):
      if self.use_global_mlp:
        x = ResidualSplitHeadMultiAxisGmlpLayer(
            grid_size=self.grid_size,
            block_size=self.block_size,
            grid_gmlp_factor=self.grid_gmlp_factor,
            block_gmlp_factor=self.block_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            name=f"SplitHeadMultiAxisGmlpLayer_{i}")(x, deterministic)
      x = RCAB(
          features=self.features,
          reduction=self.channels_reduction,
          use_bias=self.use_bias,
          name=f"channel_attention_block_1{i}")(x)

    x = x + shortcut_long

    if enc is not None and dec is not None:
      assert self.use_cross_gating
      x, _ = CrossGatingBlock(
          features=self.features,
          block_size=self.block_size,
          grid_size=self.grid_size,
          dropout_rate=self.dropout_rate,
          input_proj_factor=self.input_proj_factor,
          upsample_y=False,
          use_bias=self.use_bias,
          name="cross_gating_block")(
              x, enc + dec, deterministic=deterministic)

    if self.downsample:
      x_down = Conv_down(self.features, use_bias=self.use_bias)(x)
      return x_down, x
    else:
      return x


class UNetDecoderBlock(nn.Module):
  """Decoder block in MAXIM."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  num_groups: int = 1
  lrelu_slope: float = 0.2
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  dropout_rate: float = 0.0
  downsample: bool = True
  use_global_mlp: bool = True
  use_bias: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, bridge: jnp.ndarray = None,
               deterministic: bool = True) -> jnp.ndarray:
    x = ConvT_up(self.features, use_bias=self.use_bias)(x)

    x = UNetEncoderBlock(
        self.features,
        num_groups=self.num_groups,
        lrelu_slope=self.lrelu_slope,
        block_size=self.block_size,
        grid_size=self.grid_size,
        block_gmlp_factor=self.block_gmlp_factor,
        grid_gmlp_factor=self.grid_gmlp_factor,
        channels_reduction=self.channels_reduction,
        use_global_mlp=self.use_global_mlp,
        dropout_rate=self.dropout_rate,
        downsample=False,
        use_bias=self.use_bias)(x, skip=bridge, deterministic=deterministic)
    return x


class GetSpatialGatingWeights(nn.Module):
  """Get gating weights for cross-gating MLP block."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  input_proj_factor: int = 2
  dropout_rate: float = 0.0
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, deterministic):
    n, h, w, num_channels = x.shape

    # input projection
    x = nn.LayerNorm(name="LayerNorm_in")(x)
    x = nn.Dense(
        num_channels * self.input_proj_factor,
        use_bias=self.use_bias,
        name="in_project")(
            x)
    x = nn.gelu(x)
    u, v = jnp.split(x, 2, axis=-1)

    # Get grid MLP weights
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    u = block_images_einops(u, patch_size=(fh, fw))
    dim_u = u.shape[-3]
    u = jnp.swapaxes(u, -1, -3)
    u = nn.Dense(
        dim_u, use_bias=self.use_bias, kernel_init=nn.initializers.normal(2e-2),
        bias_init=nn.initializers.ones)(u)
    u = jnp.swapaxes(u, -1, -3)
    u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))

    # Get Block MLP weights
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    v = block_images_einops(v, patch_size=(fh, fw))
    dim_v = v.shape[-2]
    v = jnp.swapaxes(v, -1, -2)
    v = nn.Dense(
        dim_v, use_bias=self.use_bias, kernel_init=nn.initializers.normal(2e-2),
        bias_init=nn.initializers.ones)(v)
    v = jnp.swapaxes(v, -1, -2)
    v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))

    x = jnp.concatenate([u, v], axis=-1)
    x = nn.Dense(num_channels, use_bias=self.use_bias, name="out_project")(x)
    x = nn.Dropout(self.dropout_rate)(x, deterministic)
    return x


class CrossGatingBlock(nn.Module):
  """Cross-gating MLP block."""
  features: int
  block_size: Sequence[int]
  grid_size: Sequence[int]
  dropout_rate: float = 0.0
  input_proj_factor: int = 2
  upsample_y: bool = True
  use_bias: bool = True

  @nn.compact
  def __call__(self, x, y, deterministic=True):
    # Upscale Y signal, y is the gating signal.
    if self.upsample_y:
      y = ConvT_up(self.features, use_bias=self.use_bias)(y)

    x = Conv1x1(self.features, use_bias=self.use_bias)(x)
    n, h, w, num_channels = x.shape
    y = Conv1x1(num_channels, use_bias=self.use_bias)(y)

    assert y.shape == x.shape
    shortcut_x = x
    shortcut_y = y

    # Get gating weights from X
    x = nn.LayerNorm(name="LayerNorm_x")(x)
    x = nn.Dense(num_channels, use_bias=self.use_bias, name="in_project_x")(x)
    x = nn.gelu(x)
    gx = GetSpatialGatingWeights(
        features=num_channels,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        name="SplitHeadMultiAxisGating_x")(
            x, deterministic=deterministic)

    # Get gating weights from Y
    y = nn.LayerNorm(name="LayerNorm_y")(y)
    y = nn.Dense(num_channels, use_bias=self.use_bias, name="in_project_y")(y)
    y = nn.gelu(y)
    gy = GetSpatialGatingWeights(
        features=num_channels,
        block_size=self.block_size,
        grid_size=self.grid_size,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        name="SplitHeadMultiAxisGating_y")(
            y, deterministic=deterministic)

    # Apply cross gating: X = X * GY, Y = Y * GX
    y = y * gx
    y = nn.Dense(num_channels, use_bias=self.use_bias, name="out_project_y")(y)
    y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
    y = y + shortcut_y

    x = x * gy  # gating x using y
    x = nn.Dense(num_channels, use_bias=self.use_bias, name="out_project_x")(x)
    x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
    x = x + y + shortcut_x  # get all aggregated signals
    return x, y


class SAM(nn.Module):
  """Supervised attention module for multi-stage training.

  Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
  """
  features: int
  output_channels: int = 3
  use_bias: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray, x_image: jnp.ndarray, *,
               train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the SAM module to the input and features.

    Args:
      x: the output features from UNet decoder with shape (h, w, c)
      x_image: the input image with shape (h, w, 3)
      train: Whether it is training

    Returns:
      A tuple of tensors (x1, image) where (x1) is the sam features used for the
        next stage, and (image) is the output restored image at current stage.
    """
    # Get features
    x1 = Conv3x3(self.features, use_bias=self.use_bias)(x)

    # Output restored image X_s
    if self.output_channels == 3:
      image = Conv3x3(self.output_channels, use_bias=self.use_bias)(x) + x_image
    else:
      image = Conv3x3(self.output_channels, use_bias=self.use_bias)(x)

    # Get attention maps for features
    x2 = nn.sigmoid(Conv3x3(self.features, use_bias=self.use_bias)(image))

    # Get attended feature maps
    x1 = x1 * x2

    # Residual connection
    x1 = x1 + x
    return x1, image


class MAXIM(nn.Module):
  """The MAXIM model function with multi-stage and multi-scale supervision.

  For more model details, please check the CVPR paper:
  MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)

  Attributes:
    features: initial hidden dimension for the input resolution.
    depth: the number of downsampling depth for the model.
    num_stages: how many stages to use. It will also affects the output list.
    num_groups: how many blocks each stage contains.
    use_bias: whether to use bias in all the conv/mlp layers.
    num_supervision_scales: the number of desired supervision scales.
    lrelu_slope: the negative slope parameter in leaky_relu layers.
    use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
      layer.
    use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
      skip connections and multi-stage feature fusion layers.
    high_res_stages: how many stages are specificied as high-res stages. The
      rest (depth - high_res_stages) are called low_res_stages.
    block_size_hr: the block_size parameter for high-res stages.
    block_size_lr: the block_size parameter for low-res stages.
    grid_size_hr: the grid_size parameter for high-res stages.
    grid_size_lr: the grid_size parameter for low-res stages.
    num_bottleneck_blocks: how many bottleneck blocks.
    block_gmlp_factor: the input projection factor for block_gMLP layers.
    grid_gmlp_factor: the input projection factor for grid_gMLP layers.
    input_proj_factor: the input projection factor for the MAB block.
    channels_reduction: the channel reduction factor for SE layer.
    num_outputs: the output channels.
    dropout_rate: Dropout rate.

  Returns:
    The output contains a list of arrays consisting of multi-stage multi-scale
    outputs. For example, if num_stages = num_supervision_scales = 3 (the
    model used in the paper), the output specs are: outputs =
    [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
     [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
     [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
    The final output can be retrieved by outputs[-1][-1].
  """
  features: int = 64
  depth: int = 3
  num_stages: int = 2
  num_groups: int = 1
  use_bias: bool = True
  num_supervision_scales: int = 1
  lrelu_slope: float = 0.2
  use_global_mlp: bool = True
  use_cross_gating: bool = True
  high_res_stages: int = 2
  block_size_hr: Sequence[int] = (16, 16)
  block_size_lr: Sequence[int] = (8, 8)
  grid_size_hr: Sequence[int] = (16, 16)
  grid_size_lr: Sequence[int] = (8, 8)
  num_bottleneck_blocks: int = 1
  block_gmlp_factor: int = 2
  grid_gmlp_factor: int = 2
  input_proj_factor: int = 2
  channels_reduction: int = 4
  num_outputs: int = 3
  dropout_rate: float = 0.0

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool = False) -> Any:

    n, h, w, c = x.shape  # input image shape
    shortcuts = []
    shortcuts.append(x)
    # Get multi-scale input images
    for i in range(1, self.num_supervision_scales):
      shortcuts.append(jax.image.resize(
          x, shape=(n, h // (2**i), w // (2**i), c), method="nearest"))

    # store outputs from all stages and all scales
    # Eg, [[(64, 64, 3), (128, 128, 3), (256, 256, 3)],   # Stage-1 outputs
    #      [(64, 64, 3), (128, 128, 3), (256, 256, 3)],]  # Stage-2 outputs
    outputs_all = []
    sam_features, encs_prev, decs_prev = [], [], []

    for idx_stage in range(self.num_stages):
      # Input convolution, get multi-scale input features
      x_scales = []
      for i in range(self.num_supervision_scales):
        x_scale = Conv3x3(
            (2**i) * self.features,
            use_bias=self.use_bias,
            name=f"stage_{idx_stage}_input_conv_{i}")(
                shortcuts[i])

        # If later stages, fuse input features with SAM features from prev stage
        if idx_stage > 0:
          # use larger blocksize at high-res stages
          if self.use_cross_gating:
            block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
            grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr
            x_scale, _ = CrossGatingBlock(
                features=(2**i) * self.features,
                block_size=block_size,
                grid_size=grid_size,
                dropout_rate=self.dropout_rate,
                input_proj_factor=self.input_proj_factor,
                upsample_y=False,
                use_bias=self.use_bias,
                name=f"stage_{idx_stage}_input_fuse_sam_{i}")(
                    x_scale, sam_features.pop(), deterministic=not train)
          else:
            x_scale = Conv1x1(
                (2**i) * self.features,
                use_bias=self.use_bias,
                name=f"stage_{idx_stage}_input_catconv_{i}")(
                    jnp.concatenate(
                        [x_scale, sam_features.pop()], axis=-1))

        x_scales.append(x_scale)

      # start encoder blocks
      encs = []
      x = x_scales[0]  # First full-scale input feature

      for i in range(self.depth):  # 0, 1, 2
        # use larger blocksize at high-res stages, vice versa.
        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr
        use_cross_gating_layer = True if idx_stage > 0 else False

        # Multi-scale input if multi-scale supervision
        x_scale = x_scales[i] if i < self.num_supervision_scales else None

        # UNet Encoder block
        enc_prev = encs_prev.pop() if idx_stage > 0 else None
        dec_prev = decs_prev.pop() if idx_stage > 0 else None

        x, bridge = UNetEncoderBlock(
            features=(2**i) * self.features,
            num_groups=self.num_groups,
            downsample=True,
            lrelu_slope=self.lrelu_slope,
            block_size=block_size,
            grid_size=grid_size,
            block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            use_cross_gating=use_cross_gating_layer,
            name=f"stage_{idx_stage}_encoder_block_{i}")(
                x,
                skip=x_scale,
                enc=enc_prev,
                dec=dec_prev,
                deterministic=not train)

        # Cache skip signals
        encs.append(bridge)

      # Global MLP bottleneck blocks
      for i in range(self.num_bottleneck_blocks):
        x = BottleneckBlock(
            block_size=self.block_size_lr,
            grid_size=self.block_size_lr,
            features=(2**(self.depth - 1)) * self.features,
            num_groups=self.num_groups,
            block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            channels_reduction=self.channels_reduction,
            name=f"stage_{idx_stage}_global_block_{i}")(
                x, deterministic=not train)
      # cache global feature for cross-gating
      global_feature = x

      # start cross gating. Use multi-scale feature fusion
      skip_features = []
      for i in reversed(range(self.depth)):  # 2, 1, 0
        # use larger blocksize at high-res stages
        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr

        # get additional multi-scale signals
        signal = jnp.concatenate([
            UpSampleRatio(
                (2**i) * self.features,
                ratio=2**(j - i),
                use_bias=self.use_bias)(enc) for j, enc in enumerate(encs)
        ],
                                 axis=-1)

        # Use cross-gating to cross modulate features
        if self.use_cross_gating:
          skips, global_feature = CrossGatingBlock(
              features=(2**i) * self.features,
              block_size=block_size,
              grid_size=grid_size,
              input_proj_factor=self.input_proj_factor,
              dropout_rate=self.dropout_rate,
              upsample_y=True,
              use_bias=self.use_bias,
              name=f"stage_{idx_stage}_cross_gating_block_{i}")(
                  signal, global_feature, deterministic=not train)
        else:
          skips = Conv1x1(
              (2**i) * self.features, use_bias=self.use_bias)(
                  signal)
          skips = Conv3x3((2**i) * self.features, use_bias=self.use_bias)(skips)

        skip_features.append(skips)

      # start decoder. Multi-scale feature fusion of cross-gated features
      outputs, decs, sam_features = [], [], []
      for i in reversed(range(self.depth)):
        # use larger blocksize at high-res stages
        block_size = self.block_size_hr if i < self.high_res_stages else self.block_size_lr
        grid_size = self.grid_size_hr if i < self.high_res_stages else self.block_size_lr

        # get multi-scale skip signals from cross-gating block
        signal = jnp.concatenate([
            UpSampleRatio(
                (2**i) * self.features,
                ratio=2**(self.depth - j - 1 - i),
                use_bias=self.use_bias)(skip)
            for j, skip in enumerate(skip_features)
        ],
                                 axis=-1)

        # Decoder block
        x = UNetDecoderBlock(
            features=(2**i) * self.features,
            num_groups=self.num_groups,
            lrelu_slope=self.lrelu_slope,
            block_size=block_size,
            grid_size=grid_size,
            block_gmlp_factor=self.block_gmlp_factor,
            grid_gmlp_factor=self.grid_gmlp_factor,
            input_proj_factor=self.input_proj_factor,
            channels_reduction=self.channels_reduction,
            use_global_mlp=self.use_global_mlp,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            name=f"stage_{idx_stage}_decoder_block_{i}")(
                x, bridge=signal, deterministic=not train)

        # Cache decoder features for later-stage's usage
        decs.append(x)

        # output conv, if not final stage, use supervised-attention-block.
        if i < self.num_supervision_scales:
          if idx_stage < self.num_stages - 1:  # not last stage, apply SAM
            sam, output = SAM(
                (2**i) * self.features,
                output_channels=self.num_outputs,
                use_bias=self.use_bias,
                name=f"stage_{idx_stage}_supervised_attention_module_{i}")(
                    x, shortcuts[i], train=train)
            outputs.append(output)
            sam_features.append(sam)
          else:  # Last stage, apply output convolutions
            output = Conv3x3(self.num_outputs,
                             use_bias=self.use_bias,
                             name=f"stage_{idx_stage}_output_conv_{i}")(x)
            output = output + shortcuts[i]
            outputs.append(output)
      # Cache encoder and decoder features for later-stage's usage
      encs_prev = encs[::-1]
      decs_prev = decs

      # Store outputs
      outputs_all.append(outputs)
    return outputs_all


def Model(*, variant=None, **kw):
  """Factory function to easily create a Model variant like "S".

  Every model file should have this Model() function that returns the flax
  model function. The function name should be fixed.

  Args:
    variant: UNet model variants. Options: 'S-1' | 'S-2' | 'S-3'
        | 'M-1' | 'M-2' | 'M-3'
    **kw: Other UNet config dicts.

  Returns:
    The MAXIM() model function
  """

  if variant is not None:
    config = {
        # params: 6.108515000000001 M, GFLOPS: 93.163716608
        "S-1": {
            "features": 32,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 13.35383 M, GFLOPS: 206.743273472
        "S-2": {
            "features": 32,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 20.599145 M, GFLOPS: 320.32194560000005
        "S-3": {
            "features": 32,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 19.361219000000002 M, 308.495712256 GFLOPs
        "M-1": {
            "features": 64,
            "depth": 3,
            "num_stages": 1,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 40.83911 M, 675.25541888 GFLOPs
        "M-2": {
            "features": 64,
            "depth": 3,
            "num_stages": 2,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
        # params: 62.317001 M, 1042.014666752 GFLOPs
        "M-3": {
            "features": 64,
            "depth": 3,
            "num_stages": 3,
            "num_groups": 2,
            "num_bottleneck_blocks": 2,
            "block_gmlp_factor": 2,
            "grid_gmlp_factor": 2,
            "input_proj_factor": 2,
            "channels_reduction": 4,
        },
    }[variant]

    for k, v in config.items():
      kw.setdefault(k, v)

  return MAXIM(**kw)
