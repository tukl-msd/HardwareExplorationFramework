####################################################################################
#  Copyright (C) 2024, University of Kaiserslautern-Landau (RPTU), Kaiserslautern
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1.  Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2.  Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#  3.  Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

####################################################################################

####################################################################################
#
#  Authors: Vladimir Rybalkin <rybalkin@rptu.de>
#           Mohamed Moursi <mmoursi@rptu.de>
#
#  \file defaults.py
#
#
#  Hardware Exploration Framework 
#  This module defines default values and supported nodes for the Hardware Exploration Framework.
#  
#  The following code has been developed for a work package "Hardware Architectures
#  for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
#  Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
#  Systems). The project has received funding from European Union’s Horizon Europe
#  research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
#  under grant agreement No 101070408.
#
####################################################################################

from enum import Enum

class SupportedNodes:
    conv = 'Conv'
    tr_conv = 'ConvTranspose'
    linear = 'Linear'
    bn = 'Bn'  # BatchNormalization
    ins = 'In'  # InstanceNormalization
    relu = 'Relu'
    relu6 = 'Relu6'
    merge = 'Merge'
    split = 'Split'
    upsample = 'UpSample'
    reshape = 'Reshape'
    transpose = 'Transpose'
    quantize = 'QuantizeLinear'
    dequantize = 'DequantizeLinear'
    clip = 'Clip'
    maxpool = 'MaxPool'
    constant = 'Constant'
    mul = 'Mul'
    sigmoid = 'Sigmoid'
    hardswish = 'HardSwish'
    globalavgpool = 'GlobalAveragePool'
    expand = 'Expand'
    concat = 'Concat'

class DefaultValues:
    bits = 32
    int_bits = 16
    simd = 1
    pe = 1
    freq_mhz = 300e6
    target_fps = 30
    min_act_value = -128
    max_act_value = 127
    brevitas_default_act_bits = 8
    brevitas_default_act_int_bits = 1

class Dtypes(Enum):
    FLOAT = 'float'
    FIXED = 'fixed'
    INT = 'int'
