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
#  \file utils.py
#
#  Hardware Exploration Framework 
#  This module provides utility functions.
#  
#  The following code has been developed for a work package "Hardware Architectures
#  for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
#  Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
#  Systems). The project has received funding from European Union’s Horizon Europe
#  research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
#  under grant agreement No 101070408.
#
####################################################################################


import os
from brevitas.quant_tensor import QuantTensor
from onnx import numpy_helper
from torchinfo import summary
import math
import numpy as np
import errno

def shift_bit_length(x):
    return 1<<(x-1).bit_length()


def check_path(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def save_to_file(data, path):
    check_path(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        print(f"Save on disc to {path}")
        f.write(data)


def get_weight_bias(initializers, w_name, b_name):
    """Extracts weights and biases from the model
       Loop over the list of initilizers looking for weight and bias of a specific layer based on the name.
    Args:
        initializers: List of initializers
        w_name: Name of the weights tensor
        b_name: Name of the biases tensor
    Returns:
        A tuple containing the weights and biases
    """
    w = None
    b = None
    for i in range(0, len(initializers)):
        if initializers[i].name == w_name:
            w = numpy_helper.to_array(initializers[i])
        if initializers[i].name == b_name:
            b = numpy_helper.to_array(initializers[i])
        if w is not None and b is not None:
            return w, b
    raise ValueError('Weights or biases not found')


def bin_digits(val, num_bits):
    '''Returns the correct binary value of val

    Args:
        val: Decimal value to convert to binary
        num_bits: Number of bits to use
    Returns:
        A string containing the resulting binary value without 0b or minus
    Thanks Jonas:)
    '''
    if num_bits % 4 != 0:
        num_bits = (num_bits+4)-(num_bits % 4)
    s = bin(val & int('1' * num_bits, 2))[2:]
    d = f'{s:0>{num_bits}}'
    res = ''
    for i in range(0, len(d), 4):
        res = f'{res}{hex(int(d[i:i+4],base=2))[2:]}'
    return res


def onnx_to_stream(array):
    '''Takes in a numpy array and yields it's values in depth wise raster order
        (i.e. All channels of pixel (0,0) first, then all channels of pixel (0,1), etc.)

    Args:
        array: A numpy array to be converted to a stream must be 3D or 4D
    Yields:
        Values in the array in depth wise raster order
    '''
    assert array.ndim in [3, 4], f'Only 3D and 4D tensors are supported, got {array.ndim}D'
    if array.ndim == 3:
        array = array.unsqueeze(0)
    for b in range(array.shape[0]):
        for r in range(array.shape[2]):
            for c in range(array.shape[3]):
                for ch in range(array.shape[1]):
                    yield array[b][ch][r][c]


def tensor_to_stream(tensor):
    '''Takes in torch tensor or quant tensor and yields it's values in depth wise raster order
        (i.e. All channels of pixel (0,0) first, then all channels of pixel (0,1), etc.)

    Args:
        tensor: A torch tensor or quant tensor to be converted to a stream must be 3D or 4D
    Yields:
        Values in the tensor in depth wise raster order
    '''
    if isinstance(tensor, QuantTensor):
        tensor = tensor.value
    assert tensor.dim() in [3, 4], f'Only 3D and 4D tensors are supported, got {tensor.dim()}D'
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    for b in range(tensor.shape[0]):
        for r in range(tensor.shape[2]):
            for c in range(tensor.shape[3]):
                for ch in range(tensor.shape[1]):
                    yield tensor[b][ch][r][c]


def tensor_to_stream_tr_conv(tensor):
    '''Same as the original tensor_to_stream, but torch.nn.ConvTranspose2d uses a different weight order compared to torch.nn.Conv2d

    Args:
        tensor: A torch tensor or quant tensor to be converted to a stream must be 3D or 4D
    Yields:
        Values in the tensor in depth wise raster order
    '''

    if isinstance(tensor, QuantTensor):
        tensor = tensor.value
    assert tensor.dim() == 4, f'Only 3D and 4D tensors are supported, got {tensor.dim()}D'
    for a in range(tensor.shape[2]):
        for b in range(tensor.shape[3]):
            for c in range(tensor.shape[1]):
                for d in range(tensor.shape[0]):
                    yield tensor[d][c][a][b]


def rcc_to_dict(rcc):
    '''This function takes a google.protobuf.pyext._message.RepeatedCompositeContainer
    object and converts it to a dictionary.
    Args:
        rcc: a google.protobuf.pyext._message.RepeatedCompositeContainer object
    Returns:
        a dictionary containing the rcc's elements
    '''
    res = {}
    for obj in rcc:
        if obj.type == 'INTS' or obj.type == 7:
            res[obj.name] = obj.ints
        elif obj.type == 'INT' or obj.type == 2:
            res[obj.name] = obj.i
        elif obj.type == 'FLOAT' or obj.type == 1:
            res[obj.name] = obj.f
        elif obj.type == 'STRING' or obj.type == 3:
            res[obj.name] = obj.s
        elif obj.type == 'TENSOR' or obj.type == 4:
            res[obj.name] = obj.t
        else:
            raise ValueError(f'Unknown type {obj.type}, name {obj.name}')
    return res


def value_info_dims(v):
    '''This function takes a ValueInfo object and returns its dimensions.
    Args:
        v: a ValueInfo object Like This:
        name: "/features/features.0/Conv_output_0"
        type {
        tensor_type {
            elem_type: 1
            shape {
            dim {
                dim_value: 1
            }
            dim {
                dim_value: 64
            }
            dim {
                dim_value: 224
            }
            dim {
                dim_value: 224
            }
            }
        }
        }
    Returns:
        a list containing the dimensions of the ValueInfo object v like this: [1, 64, 224, 224]

    '''
    return [v.type.tensor_type.shape.dim[i].dim_value for i in range(len(v.type.tensor_type.shape.dim))]


def model_summarizer(model, input_shape):
    ''' Takes a model object & uses torchinfo summary to generate a summary list & then loop over it to exclude unnecessary layers
    Args: model - a pytorch model object
    Returns: a list containing the summary of the model
    '''
    summary_list = summary(model, input_shape, verbose=0, depth=2).summary_list
    summarized = []
    for l in summary_list:
        if l.class_name in ['Conv2d', 'QuantConv2d', 'QuantMaxPool2d', 'ConvTranspose2d', 'QuantConvTranspose2d', 'UNet_down_block', 'Quant_unet_down_block', 'UNet_up_block', 'Quant_unet_up_block', 'Sequential', 'QuantIdentity', 'ReLU', 'QuantReLU', 'ReLU6', 'BatchNorm2d']:
            summarized.append(l)
    return summarized


def parse_model_init(key_value_pairs):
    """ Generated by ChatGPT :) modified by me :)
    Parses a list of key-value pairs separated by colons (:)
    and returns a dictionary with the values cast to their original type.

    Args:
        key_value_pairs (str): A list of key-value pairs separated by colons (:).

    Returns:
        dict: A dictionary with the key-value pairs from the input string, where the values
        are casted to their original type (int, bool, or str).

    Example:
        key_value_pairs = "name:John age:25 married:t"
        output_dict = parse_model_init(key_value_pairs)
        print(output_dict)
        # Output: {'name': 'John', 'age': 25, 'married': True}
    """
    result = {}
    for item in key_value_pairs:
        key, value = item.split(':')
        if value.isdigit():
            result[key] = int(value)
        elif value.lower() in ['true', 't', '1']:
            result[key] = True
        elif value.lower() in ['false', 'f', '0']:
            result[key] = False
        else:
            result[key] = value
    return result


def find_simd_pe(inch, outch, factor, constant=None):
    if constant == 'pe':
        for simd in range(factor, inch+1, 1):
            if inch % simd == 0:
                return simd, 1
    elif constant == 'simd':
        for pe in range(factor, outch+1, 1):
            if outch % pe == 0:
                return 1, pe
    square_root = math.ceil(math.sqrt(factor))
    if inch % square_root == 0 and outch % square_root == 0:
        if square_root <= inch and square_root <= outch:
            return square_root, square_root
        else:
            return inch, outch
    for simd in range(square_root, inch+1, 1):
        if inch % simd == 0:
            for pe in range(square_root, outch+1, 1):
                if outch % pe == 0:
                    if (pe * simd) >= factor:
                        return simd, pe
    return inch, outch


def equalize_simd_pe(nx_graph, node, source_node=None, direction=None):
    '''
        A recursive function that check if the node's successors have the same SIMD as the PE of the node, and if the node's predecessors have the same PE as the SIMD of the node.
        If not, it changes the values to the maximum of the two. Retuning it to the previous call so it can also be set there.
    '''
    print(f'Equalizing SIMD and PE for node {node.name}...')
    if direction in ['pe', 'both']:
        node_successors = list(nx_graph.successors(node))
        if source_node in node_successors:
            node_successors.remove(source_node)
        print([x.name for x in node_successors])
        if len(node_successors) > 0:
            max_value = max([n.simd for n in node_successors]+[node.pe])
            node.pe = max_value
            for succ in node_successors:
                succ.simd = max_value
                equalize_simd_pe(nx_graph, succ, node, 'simd')
    if direction in ['simd', 'both']:
        node_predecessors = list(nx_graph.predecessors(node))
        if source_node in node_predecessors:
            node_predecessors.remove(source_node)
        print([x.name for x in node_predecessors])
        if len(node_predecessors) > 0:
            max_value = max([n.pe for n in node_predecessors]+[node.simd])
            node.simd = max_value
            for pred in node_predecessors:
                pred.pe = max_value
                equalize_simd_pe(nx_graph, pred, node)


def get_source_node(var_name, graph):
    ''' Get the node from the graph by output name
    '''
    for i in graph.node:
        if var_name in i.output:
            return i
    return None


def get_dest_node(var_name, graph):
    ''' Get the node from the graph by input name
    '''
    for i in graph.node:
        if var_name in i.input:
            return i
    return None


def profile_array(array):
    '''
        This function takes an array and returns the minimum and maximum values of the array.
    '''
    return int(np.min(array)), int(np.max(array))


def calc_bit_width(min_max):
    '''
        This function takes a min_max and returns the minimum bit width required to represent the min_max.
        min_max is a tuple of the form (min, max)
    '''
    return math.ceil(math.log2(abs(min_max[0])+abs(min_max[1])))
