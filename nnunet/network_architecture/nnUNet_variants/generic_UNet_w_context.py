#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import numpy as np
import torch
import torch.nn.functional
from torchvision.transforms.functional_tensor import crop

from nnunet.network_architecture.generic_UNet import Generic_UNet, StackedConvLayers, ConvDropoutNormNonlin


class Generic_UNet_w_context(Generic_UNet):
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648

    def __init__(self, context_encoder: torch.nn.Module, *args, **kwargs):
        """
        First version that uses a CNN to encode context around the main patch. Assuming that the following formula holds:
        Encoder contains 5 downsamplings and the input patch is at 8.0 mpp. This results in 2**3 * 2**5 = 2**8 mpp patch
         of 16x16xFM (this depends on resnet18/50). nnUNet will be fixed at 7 poolings to allow for the encoder in VRAM.
         With 7 poolings the input of 0.5 mpp will be 2**-1 * 2**7 = 2**6 mpp. To braze the gap the encoded patch will be
         cropped by 2**2 = 2**(8-6).
        """
        super(Generic_UNet_w_context, self).__init__(*args, **kwargs)
        self.context_encoder = context_encoder
        reduce_conv_kwargs = self.conv_kwargs.copy()
        reduce_conv_kwargs['kernel_size'] = [1, 1]
        reduce_conv_kwargs['padding'] = [0, 0]
        self.reduce_fm_conv = StackedConvLayers(480 + 512, 480, 1,
                                                self.conv_op, reduce_conv_kwargs, self.norm_op,
                                                self.norm_op_kwargs, self.dropout_op,
                                                self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                basic_block=ConvDropoutNormNonlin)

    def forward(self, x):
        main, context = x
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            main = self.conv_blocks_context[d](main)
            skips.append(main)
            if not self.convolutional_pooling:
                main = self.td[d](main)

        main_encoding = self.conv_blocks_context[-1](main)
        context_encoding = self.context_encoder.forward_features(context)

        start_x, start_y = (
                torch.div(torch.tensor(context_encoding.shape[-2:]), 2, rounding_mode='trunc') -
                torch.div(torch.tensor(main_encoding.shape[-2:]), 2, rounding_mode='floor')
        ).type(torch.int)

        w, h = main_encoding.shape[-2:]
        x = torch.cat((crop(context_encoding, start_x, start_y, w, h), main_encoding), dim=1)

        x = self.reduce_fm_conv(x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
