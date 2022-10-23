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

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import subprocess
subprocess.call(["pip", "install", "."])

import numpy as np
from PIL import Image
import importlib
import ml_collections
import tempfile
import jax.numpy as jnp
import flax
from cog import BasePredictor, Path, Input, BaseModel

from maxim.run_eval import (
    _MODEL_FILENAME,
    _MODEL_VARIANT_DICT,
    _MODEL_CONFIGS,
    get_params,
    mod_padding_symmetric,
    make_shape_even,
    augment_image,
)


class Predictor(BasePredictor):
    def setup(self):

        self.params = {
            "Image Denoising": get_params("checkpoints/denoising-SIDD/checkpoint.npz"),
            "Image Deblurring (GoPro)": get_params(
                "checkpoints/debluring-GoPro/checkpoint.npz"
            ),
            "Image Deblurring (REDS)": get_params(
                "checkpoints/debluring-REDS/checkpoint.npz"
            ),
            "Image Deblurring (RealBlur_R)": get_params(
                "checkpoints/debluring-Real-Blur-R/checkpoint.npz"
            ),
            "Image Deblurring (RealBlur_J)": get_params(
                "checkpoints/debluring-Real-Blur-J/checkpoint.npz"
            ),
            "Image Deraining (Rain streak)": get_params(
                "checkpoints/deraining-Rain13k/checkpoint.npz"
            ),
            "Image Deraining (Rain drop)": get_params(
                "checkpoints/deraining-Raindrop/checkpoint.npz"
            ),
            "Image Dehazing (Indoor)": get_params(
                "checkpoints/dehazing-RESIDE-Indoor/checkpoint.npz"
            ),
            "Image Dehazing (Outdoor)": get_params(
                "checkpoints/dehazing-RESIDE-Outdoor/checkpoint.npz"
            ),
            "Image Enhancement (Low-light)": get_params(
                "checkpoints/enhancement-LOL/checkpoint.npz"
            ),
            "Image Enhancement (Retouching)": get_params(
                "checkpoints/enhancement-FiveK/checkpoint.npz"
            ),
        }

        model_mod = importlib.import_module(f"maxim.models.{_MODEL_FILENAME}")
        self.models = {}
        for task in _MODEL_VARIANT_DICT.keys():
            model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
            model_configs.variant = _MODEL_VARIANT_DICT[task]
            self.models[task] = model_mod.Model(**model_configs)

    def predict(
        self,
        model: str = Input(
            choices=[
                "Image Denoising",
                "Image Deblurring (GoPro)",
                "Image Deblurring (REDS)",
                "Image Deblurring (RealBlur_R)",
                "Image Deblurring (RealBlur_J)",
                "Image Deraining (Rain streak)",
                "Image Deraining (Rain drop)",
                "Image Dehazing (Indoor)",
                "Image Dehazing (Outdoor)",
                "Image Enhancement (Low-light)",
                "Image Enhancement (Retouching)",
            ],
            description="Choose a model.",
        ),
        image: Path = Input(
            description="Input image.",
        ),
    ) -> Path:

        params = self.params[model]
        task = model.split()[1]
        model = self.models[task]

        input_img = (
            np.asarray(Image.open(str(image)).convert("RGB"), np.float32) / 255.0
        )

        # Padding images to have even shapes
        height, width = input_img.shape[0], input_img.shape[1]
        input_img = make_shape_even(input_img)
        height_even, width_even = input_img.shape[0], input_img.shape[1]

        # padding images to be multiplies of 64
        input_img = mod_padding_symmetric(input_img, factor=64)
        input_img = np.expand_dims(input_img, axis=0)

        # handle multi-stage outputs, obtain the last scale output of last stage
        preds = model.apply({"params": flax.core.freeze(params)}, input_img)
        if isinstance(preds, list):
            preds = preds[-1]
            if isinstance(preds, list):
                preds = preds[-1]

        preds = np.array(preds[0], np.float32)

        # unpad images to get the original resolution
        new_height, new_width = preds.shape[0], preds.shape[1]
        h_start = new_height // 2 - height_even // 2
        h_end = h_start + height
        w_start = new_width // 2 - width_even // 2
        w_end = w_start + width
        preds = preds[h_start:h_end, w_start:w_end, :]

        # save files
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        Image.fromarray(
            np.array((np.clip(preds, 0.0, 1.0) * 255.0).astype(jnp.uint8))
        ).save(str(out_path))

        return out_path