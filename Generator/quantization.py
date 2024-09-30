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
#  \file quantization.py
#
#  Hardware Exploration Framework 
#  This module provides functions and classes for quantizing neural network weights,
#  biases, and activations.
#  
#  The following code has been developed for a work package "Hardware Architectures
#  for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
#  Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
#  Systems). The project has received funding from European Union’s Horizon Europe
#  research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
#  under grant agreement No 101070408.
#
####################################################################################


import torch
import torch.nn as nn
from torch.autograd import Function
from functools import partial
import math
from defaults import Dtypes

##############################################################################################################
# Conversion functions
##############################################################################################################

def bindigits(n, bits):
    """
    | Convert integer to twos complement representation.

    :param int n: Integer number to convert
    :param int bits: Number of bits of twos complement
    :return: String representing twos complement
    """
    s = bin(n & int("1" * bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def float_to_binary_string(q_float, bit_width, int_width):
    """
    | Convert float to binary fixed point representation as needed for HLS ap_fixed datatype.

    :param float q_float: Floating point number to convert
    :param int bit_width: Number of bits of fixed point number
    :param int int_width: Number of integer bits of fixed point number
    :return: String representing binary representation of fixed point number
    """

    frac_width = bit_width - int_width
    mul_fac = 2 ** frac_width
    q_int = int(q_float * mul_fac)
    bin_int = bindigits(q_int, bit_width)
    return str(bin_int)

def conv_bias_mapper_2D(bias, PE, bits, int_bits, dtype=Dtypes.FIXED):
    """
    | Transforms bias of conv layer to a string that can be included in .hpp file.
    | Bias has shape: OUTPUT_CHANNELS.
    | Stores bias in BiasActivation wrapper with ap_fixed type.

    :param bias: Bias of convolutional layer as given by *layer.bias*
    :param PE: Number of parallel computed output channels
    :param str i: index of layer
    :param int bias_bits: Number of bits to use for quantization
    :param int bias_int_bits: Number of integer bits to use for quantization
    :param Dtypes dtype: Type of data to quantize
    :return: String to save in .hpp file
    """
    bias = bias.detach().cpu().numpy()

    output_channels = bias.shape[0]
    bias_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0

    #bias_string += "{"
    for output_channel in range(output_channels):
        bias_string += "\n"
        fraction_bits = bits - int_bits
        if dtype == Dtypes.INT:
            value = f'{bias[base_output_channel_iter + pe_output_channel_iter]}'
        else:
            value = f'{bias[base_output_channel_iter + pe_output_channel_iter]:.{fraction_bits}}'
        bias_string += value
        #bias_string += str(bias[base_output_channel_iter + pe_output_channel_iter])
        #print("bias_string: ", bias_string)
        pe_output_channel_iter += PE
        if pe_output_channel_iter == output_channels:
            pe_output_channel_iter = 0
            base_output_channel_iter += 1
        bias_string += ","
    bias_string = bias_string[:-1]
    #bias_string += "};"

    return bias_string

def fc_weight_mapper_2D(weights, PE, SIMD, weight_bits, weight_int_bits, float_dtype=False):
    """
    | Transforms weights of conv layer to a string that can be included in .hpp file.
    | Weight have shape: OUTPUT_CHANNELS x INPUT_CHANNELS x KERNEL_SIZE x KERNEL_SIZE.
    | Stores weights in ap_uint<input_channels*bit_width> OUTPUT_CHANNELS x KERNEL_SIZE*KERNEL_SIZE two dimensional array.
    | Binary representation of weight corresponds to ap_fixed number in HLS.
    | Therefore *reinterpret_cast* can be used in .cpp to access weight in ap_fixed format.

    :param weights: Weights of convolutional layer as given by *layer.weight*
    :param PE: Number of parallel computed output channels
    :param SIMD: Number of parallel computed input channels
    :param str i: index of layer
    :param int weight_bits: Number of bits to use for quantization
    :param int weight_int_bits: Number of integer bits to use for quantization
    :param bool float_dtype: If True, weights are saved as floating point numbers
    :return: String to save in .hpp file
    """

    output_channels = weights.shape[0]
    input_channels = weights.shape[1]

    TILES = int( (input_channels / SIMD) * (output_channels / PE) )

    weight_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0
    input_channel = 0

    for PE_iter in range(PE):
        if PE_iter != 0:
            weight_string += "\n"
        weight_string += "{"
        for tile in range(TILES):
            weight_string += '"0x' if float_dtype is not True else ''
            bin_string = ""
            for SIMD_iter in reversed(range(SIMD)):
                bin_string += float_to_binary_string(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel], weight_bits, weight_int_bits) if float_dtype is not True else str(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel].item())
                if SIMD_iter == 0:
                    input_channel += SIMD
                    if input_channel == input_channels:
                        input_channel = 0
                        pe_output_channel_iter += PE
                        if pe_output_channel_iter == output_channels:
                            pe_output_channel_iter = 0
                            base_output_channel_iter += 1

            weight_string += format((int(bin_string, 2)), 'x') if float_dtype is not True else bin_string
            weight_string += '",' if float_dtype is not True else ','
        weight_string = weight_string[:-1]
        weight_string += "}"
        weight_string += ","
    weight_string = weight_string[:-1]
    return weight_string

def conv_weight_mapper_2D(weights, PE, SIMD, weight_bits, weight_int_bits, float_dtype=False):
    """
    | Transforms weights of conv layer to a string that can be included in .hpp file.
    | Weight have shape: OUTPUT_CHANNELS x INPUT_CHANNELS x KERNEL_SIZE x KERNEL_SIZE.
    | Stores weights in ap_uint<input_channels*bit_width> OUTPUT_CHANNELS x KERNEL_SIZE*KERNEL_SIZE two dimensional array.
    | Binary representation of weight corresponds to ap_fixed number in HLS.
    | Therefore *reinterpret_cast* can be used in .cpp to access weight in ap_fixed format.

    :param weights: Weights of convolutional layer as given by *layer.weight*
    :param PE: Number of parallel computed output channels
    :param SIMD: Number of parallel computed input channels
    :param str i: index of layer
    :param int weight_bits: Number of bits to use for quantization
    :param int weight_int_bits: Number of integer bits to use for quantization
    :param bool float_dtype: If True, weights are saved as floating point numbers
    :return: String to save in .hpp file
    """

    output_channels = weights.shape[0]
    input_channels = weights.shape[1]
    kernel_size = weights.shape[2]

    TILES = int(kernel_size * kernel_size * (input_channels / SIMD) * (output_channels / PE))

    weight_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0
    input_channel = 0
    dim1 = 0
    dim2 = 0

    for PE_iter in range(PE):
        if PE_iter != 0:
            weight_string += "\n"
        weight_string += "{"
        for tile in range(TILES):
            weight_string += '"0x' if float_dtype is not True else ''
            bin_string = ""
            for SIMD_iter in reversed(range(SIMD)):
                bin_string += float_to_binary_string(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel][dim2][dim1], weight_bits, weight_int_bits) if float_dtype is not True else str(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel][dim2][dim1].item())
                if SIMD_iter == 0:
                    input_channel += SIMD
                    if input_channel == input_channels:
                        input_channel = 0
                        dim1 += 1
                        if dim1 == kernel_size:
                            dim1 = 0
                            dim2 += 1
                            if dim2 == kernel_size:
                                dim2 = 0
                                pe_output_channel_iter += PE
                                if pe_output_channel_iter == output_channels:
                                    pe_output_channel_iter = 0
                                    base_output_channel_iter += 1

            weight_string += format((int(bin_string, 2)), 'x') if float_dtype is not True else bin_string
            weight_string += '",' if float_dtype is not True else ','
        weight_string = weight_string[:-1]
        weight_string += "}"
        weight_string += ","
    weight_string = weight_string[:-1]

    return weight_string

def tr_conv_weight_mapper_2D(weights, PE, SIMD, weight_bits, weight_int_bits, float_dtype=False):
    """
    | Transforms weights of transposed conv layer to a string that can be included in .hpp file.
    | Weight have shape: INPUT_CHANNELS x OUTPUT_CHANNELS x KERNEL_SIZE x KERNEL_SIZE.
    | Stores weights in ap_uint<input_channels*bit_width> OUTPUT_CHANNELS x KERNEL_SIZE*KERNEL_SIZE two dimensional array.
    | Binary representation of weight corresponds to ap_fixed number in HLS.
    | Therefore *reinterpret_cast* can be used in .cpp to access weight in ap_fixed format.

    :param weights: Weights of transposed convolutional layer as given by *layer.weight*
    :param PE: Number of parallel computed output channels
    :param SIMD: Number of parallel computed input channels
    :param str i: index of layer
    :param int weight_bits: Number of bits to use for quantization
    :param int weight_int_bits: Number of integer bits to use for quantization
    :param bool float_dtype: If True, weights are saved as floating point numbers
    :return: String to save in .hpp file
    """

    output_channels = weights.shape[0]
    input_channels = weights.shape[1]
    kernel_size = weights.shape[2]

    TILES = int(kernel_size * kernel_size * (input_channels / SIMD) * (output_channels / PE))

    weight_string = ""

    base_output_channel_iter = 0
    pe_output_channel_iter = 0
    input_channel = 0
    dim1 = 0
    dim2 = 0

    for PE_iter in range(PE):
        if PE_iter != 0:
            weight_string += "\n"
        weight_string += "{"
        for tile in range(TILES):
            weight_string += '"0x' if float_dtype is not True else ''
            bin_string = ""
            for SIMD_iter in reversed(range(SIMD)):
                bin_string += float_to_binary_string(weights[SIMD_iter + input_channel][base_output_channel_iter + pe_output_channel_iter][dim2][dim1], weight_bits, weight_int_bits) if float_dtype is not True else str(weights[base_output_channel_iter + pe_output_channel_iter][SIMD_iter + input_channel][dim2][dim1].item())
                if SIMD_iter == 0:
                    input_channel += SIMD
                    if input_channel == input_channels:
                        input_channel = 0
                        dim1 += 1
                        if dim1 == kernel_size:
                            dim1 = 0
                            dim2 += 1
                            if dim2 == kernel_size:
                                dim2 = 0
                                pe_output_channel_iter += PE
                                if pe_output_channel_iter == output_channels:
                                    pe_output_channel_iter = 0
                                    base_output_channel_iter += 1

            weight_string += format((int(bin_string, 2)), 'x') if float_dtype is not True else bin_string
            weight_string += '",' if float_dtype is not True else ','
        weight_string = weight_string[:-1]
        weight_string += "}"
        weight_string += ","
    weight_string = weight_string[:-1]

    return weight_string
##############################################################################################################
# Profiler
##############################################################################################################

class Profiler(object):
    """
    Profiler class for tracking and quantizing the range of input values.
    Attributes:
        minimum (float): The minimum value observed.
        maximum (float): The maximum value observed.
        int_bit_width (int): The bit width required to represent the integer part of the range.
        name (str): The name of the profiler instance.
    Methods:
        __init__(name):
            Initializes the Profiler with a name and default range values.
        update(input_):
            Updates the minimum and maximum values based on the input tensor.
        status():
            Returns the current minimum and maximum values.
        reset():
            Resets the minimum and maximum values to their default states.
        return_int_bit_width():
            Calculates and returns the bit width required to represent the integer part of the range.
        __repr__():
            Returns a string representation of the Profiler instance, including its name, minimum, maximum, and int_bit_width.
    """

    def __init__(self, name):
        super(Profiler, self).__init__()
        self.minimum = 1000.0
        self.maximum = -1000.0
        self.int_bit_width = 0
        self.name = name

    def update(self, input_):
        minimum = torch.min(input_)
        maximum = torch.max(input_)

        if (minimum < self.minimum):
            self.minimum = minimum

        if (maximum > self.maximum):
            self.maximum = maximum

    def status(self):
        return self.minimum, self.maximum

    def reset(self):
        self.minimum = 1000.0
        self.maximum = -1000.0
        self.int_bit_width = 0

    def return_int_bit_width(self):
        minimum = int(self.minimum)
        maximum = int(self.maximum)
        minimum = int(math.log2(math.fabs(minimum))) + 1 if minimum != 0.0 else 0
        maximum = int(math.log2(math.fabs(maximum))) + 1 if maximum != 0.0 else 0
        self.int_bit_width = minimum if minimum > maximum else maximum
        return self.int_bit_width

    def __repr__(self):

        return self.__class__.__name__ + ': ' \
        + self.name \
        + ': Min=' + str(self.minimum.item()) \
        + ', Max=' + str(self.maximum.item()) \
        + ', int_bit_width=' + str(self.return_int_bit_width())

##############################################################################################################
# Quantization primitives
##############################################################################################################

def quantize(x, params):
    q_tensor = x * params.prescale

    q_tensor_r = q_tensor.round()
    q_tensor_f = q_tensor.floor()
    q_tensor_c = q_tensor.ceil()
    q_tensor_tmp = torch.where(q_tensor-q_tensor.int()==0.5,q_tensor_f,q_tensor_r)
    q_tensor_tmp = torch.where(q_tensor-q_tensor.int()==-0.5,q_tensor_c,q_tensor_tmp)

    q_tensor = q_tensor_tmp * params.postscale
    q_tensor = q_tensor.clamp(params.min_val, params.max_val)
    return q_tensor

class Identity(Function):

    '''@staticmethod
    def symbolic(g, input):
        return g.op('Identity', input)'''

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad

class QuantizeFixedActivation(Function):

    '''@staticmethod
    def symbolic(g, params, input):
        return g.op('QuantizeFixedActivation', input)'''

    @staticmethod
    def forward(ctx, params, a_in):
        ctx.save_for_backward(a_in)
        ctx.params = params
        a_out = quantize(a_in, params)
        return a_out

    @staticmethod
    def backward(ctx, grad_output):
        params = ctx.params
        a_in, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(a_in.ge(params.max_val) | a_in.le(params.min_val), 0)
        return None, grad_input

class QuantizeFixedWeight(Function):

    '''@staticmethod
    def symbolic(g, params, input):
        return g.op('QuantizeFixedWeight', input)'''

    @staticmethod
    def forward(ctx, params, w_in):
        w_out = quantize(w_in, params)
        return w_out

    @staticmethod
    def backward(ctx, grad):
        return None, grad

class QuantizationParams():

    def __init__(self, q_type):
        self.q_type = q_type

class SignedFixedQuantizationParams(QuantizationParams):

    def __init__(self, bit_width, int_bit_width, q_type):
        super(SignedFixedQuantizationParams, self).__init__(q_type)
        self.bit_width = bit_width
        self.int_bit_width = int_bit_width  # including implicit sign bit

        self.frac_bit_width = self.bit_width - self.int_bit_width
        self.prescale = 2 ** self.frac_bit_width
        self.postscale = 2 ** (- self.frac_bit_width)
        self.min_val = - (2 ** (self.int_bit_width - 1))
        self.max_val = - self.min_val - self.postscale

class UnsignedFixedQuantizationParams(QuantizationParams):

    def __init__(self, bit_width, int_bit_width, q_type):
        super(UnsignedFixedQuantizationParams, self).__init__(q_type)
        self.bit_width = bit_width
        self.int_bit_width = int_bit_width

        self.frac_bit_width = self.bit_width - self.int_bit_width
        self.prescale = 2 ** self.frac_bit_width
        self.postscale = 2 ** (- self.frac_bit_width)
        self.min_val = 0.0
        self.max_val = 2 ** self.int_bit_width - self.postscale

class QuantizationScheme(nn.Module):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(QuantizationScheme, self).__init__()
        if q_type == 'identity':
            self.q_params = QuantizationParams(q_type)
        elif q_type == 'fixed_unsigned':
            self.q_params = UnsignedFixedQuantizationParams(bit_width, int_bit_width, q_type)
        elif q_type == 'fixed_signed':
            self.q_params = SignedFixedQuantizationParams(bit_width, int_bit_width, q_type)
        elif q_type == 'binary':
            self.q_params = QuantizationParams(q_type)
        else:
            raise Exception(f'Unknown quantization scheme: {q_type}.')

    def __repr__(self):
        if self.q_params.q_type == 'binary' or self.q_params.q_type == 'identity':
            string = f"q_type: {self.q_params.q_type}"
        else:
            string = f"q_type: {self.q_params.q_type}, bit_width: {self.q_params.bit_width}, int_bit_width: {self.q_params.int_bit_width}"
        return string

class WeightQuantizationScheme(QuantizationScheme):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(WeightQuantizationScheme, self).__init__(q_type, threshold, bit_width, int_bit_width)
        if self.q_params.q_type == 'identity':
            self.q_function = partial(Identity.apply)
        elif self.q_params.q_type == 'fixed_unsigned' or self.q_params.q_type == 'fixed_signed':
            self.q_function = partial(QuantizeFixedWeight.apply, self.q_params)
        elif self.q_params.q_type == 'binary':
            self.q_function = partial(BinarizeWeight.apply)
        else:
            raise ValueError(f'Unknown quantization scheme: {q_type}.')

    def forward(self, x):
        return self.q_function(x)

class BiasQuantizationScheme(QuantizationScheme):

    def __init__(self, q_type, threshold=None, bit_width=None, int_bit_width=None):
        super(BiasQuantizationScheme, self).__init__(q_type, threshold, bit_width, int_bit_width)
        if self.q_params.q_type == 'identity':
            self.q_function = partial(Identity.apply)
        elif self.q_params.q_type == 'fixed_unsigned' or self.q_params.q_type == 'fixed_signed':
            self.q_function = partial(QuantizeFixedWeight.apply, self.q_params)
        else:
            raise ValueError(f'Unknown quantization scheme: {q_type}.')

    def forward(self, x):
        return self.q_function(x)

