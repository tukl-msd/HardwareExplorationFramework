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
#  \file network_hls.py
#
#
#  Hardware Exploration Framework 
#  This module provides the NetworkHLS class, which handles various operations
#  related to High-Level Synthesis (HLS) for neural networks.
#  
#  The following code has been developed for a work package "Hardware Architectures
#  for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
#  Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
#  Systems). The project has received funding from European Union’s Horizon Europe
#  research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
#  under grant agreement No 101070408.
#
####################################################################################

import math
import os

import networkx as nx
import numpy as np
from onnx import numpy_helper
import onnx
import onnxruntime
from tqdm import tqdm
#sudo apt-get install graphviz
import pydot
import json

from utils import  value_info_dims, rcc_to_dict, save_to_file, find_simd_pe
from brevitas_utils import merge_brevitas
from node_hls import NodeHLS
from defaults import SupportedNodes, DefaultValues, Dtypes


class NetworkHLS():
    """
    NetworkHLS is a class designed to handle various operations related to High-Level Synthesis (HLS) for neural networks.

    This class provides methods to convert ONNX models to networkx graphs, visualize these graphs, merge nodes, add split nodes, 
    and remove unsupported nodes. It also manages various attributes related to the HLS process, such as file names, paths, 
    counters for different types of layers, and data type configurations.

    Methods:
        networkx_to_png(nx_graph, safix='', extra_node_label=None):
            Converts a networkx graph to a PNG image for visualization.
        
        merge_several_nodes(nx_graph):
            Merges several nodes in the networkx graph based on specific conditions.
        
        add_split_nodes(nx_graph):
            Adds split nodes to the networkx graph.
        
        remove_unsupported_nodes(nx_graph, op_type_list=None):
            Removes unsupported nodes from the networkx graph.
        
        onnx_to_networkx(model_path, brevitas, int_quant, scale_bit_width, stop_node=None):
            Converts an ONNX model to a networkx graph.
    """
    def __init__(self, dtype=None):

        self.subgraphs = []

        self.file_name = ''
        self.params_hpp_name = "dnn_params"
        self.config_hpp_name = "dnn_config"
        self.top_level_cpp_name = "dnn_top_level"
        self.top_level_wrapper_name = "dnn_top_level_wrapper"
        self.profilers_name = "dnn_profilers"
        self.profiler = "profiler"

        self.params_hpp_path = f"./{self.params_hpp_name}.hpp"
        self.config_hpp_path = f"./{self.config_hpp_name}.hpp"
        self.top_level_cpp_path = f"./{self.top_level_cpp_name}.cpp"
        self.top_level_wrapper_hpp_path = f"./{self.top_level_wrapper_name}.hpp"
        self.top_level_wrapper_cpp_path = f"./{self.top_level_wrapper_name}.cpp"
        self.profilers_hpp_path = f"./{self.profilers_name}.hpp"
        self.profilers_cpp_path = f"./{self.profilers_name}.cpp"

        self.cnt_conv = 0
        self.cnt_bn = 0
        self.cnt_in = 0
        self.cnt_relu = 0
        self.cnt_linear = 0
        self.cnt_split = 0
        self.cnt_merge = 0
        self.cnt_upsample = 0
        self.cnt_pooling = 0
        self.cnt_avgpool = 0
        self.cnt_mul = 0
        self.cnt_sigmoid = 0
        self.cnt_hardswish = 0
        self.cnt_layer_block = 0
        self.cnt_expand = 0
        self.cnt_concat = 0
        self.cnt_tr_conv = 0

        # default
        cycle_time = 1/DefaultValues.freq_mhz
        target_latency_s = 1/DefaultValues.target_fps
        target_latency_cc = target_latency_s/cycle_time
        self.target_latency_cc = math.ceil(target_latency_cc)

        self.on_chip = "on_chip" # Mapped on chip to a stream
        self.off_chip = "off_chip" # Mapped to off-chip memory using AXI-M
        self.default_mapping = self.on_chip

        self.dtype = dtype
        self._float = "FLOAT"

        if self.dtype is Dtypes.FLOAT:
            self.default_bits = 32
            self.default_int_bits = 32

            self.default_input_mem_width = 32
            self.default_output_mem_width = 32
        else:
            self.default_bits = DefaultValues.bits
            self.default_int_bits = DefaultValues.int_bits

            self.default_input_mem_width = 64
            self.default_output_mem_width = 64

        self.input_mem_width = self.default_input_mem_width
        self.output_mem_width = self.default_output_mem_width
        self._input_mem_width = "INPUT_MEM_WIDTH"
        self._output_mem_width = "OUTPUT_MEM_WIDTH"
        self._input_dim = "INPUT_DIM"
        self._input_channels = "INPUT_CH"
        self._input_bits = "INPUT_BITS"
        self._input_int_bits = "INPUT_INT_BITS"
        self._input_dtype = "INPUT_dtype"
        self.__output_classes = "_TOP_LEVEL_OUTPUT_CLASSES"
        self.__output_bits = "_TOP_LEVEL_OUTPUT_BITS"
        self.__output_int_bits = "_TOP_LEVEL_OUTPUT_INT_BITS"
        self.__output_dtype = "_TOP_LEVEL_OUTPUT_dtype"

    def networkx_to_png(self, nx_graph, safix='', extra_node_label=None):
        '''

            This function copies networkx graph to pydot graph used for visualization

        '''
        pydot_graph = pydot.Dot("dnnhls", graph_type="graph", bgcolor="white")
        if len(self.subgraphs) == 0:
            if extra_node_label is not None:
                for node in nx_graph.nodes():
                    label = f'{node.name}\n'
                    for attr in extra_node_label:
                        if hasattr(node, attr):
                            label += f" {attr}: {getattr(node, attr)}\n"
                    pydot_graph.add_node(pydot.Node(str(node), label=label))
            else:
                for node in nx_graph.nodes():
                    name = node.name
                    buffer_latency, mvau_latency= node.get_latency_cc_node(simd=1, pe=1)
                    buffer_latency = int(buffer_latency)
                    mvau_latency = int(mvau_latency)
                    node_latency = max(buffer_latency, mvau_latency)
                    target_latency = self.target_latency_cc
                    factor = math.ceil(node_latency/target_latency)
                    simd = node.simd
                    pe=node.pe
                    label_1=f"{name}\n {buffer_latency},{mvau_latency}\n {node_latency}/{target_latency}\n opt_scale_f={factor}\n simd={simd},pe={pe}\n"
                    buffer_latency, mvau_latency= node.get_latency_cc_node()
                    buffer_latency = int(buffer_latency)
                    mvau_latency = int(mvau_latency)
                    node_latency = max(buffer_latency, mvau_latency)
                    target_latency = self.target_latency_cc
                    factor = math.ceil(node_latency/target_latency)
                    label_2=f"{buffer_latency},{mvau_latency}\n {node_latency}/{target_latency}\n opt_scale_f={factor}\n"
                    pydot_graph.add_node(pydot.Node(str(node), label=label_1+label_2))
        else:
            for i, subgraph in enumerate(self.subgraphs):
                num_params = self.get_num_params(subgraph)
                label = f"TopLevel_{i}\n num_params={num_params}"
                pydot_subgraph = pydot.Subgraph(f"cluster_{i}", label=label)
                for node in subgraph.nodes():
                    node_label = f'{node.name}\n'
                    if extra_node_label is not None:
                        for attr in extra_node_label:
                            if hasattr(node, attr):
                                node_label += f" {attr}: {getattr(node, attr)}\n"
                    pydot_node = pydot.Node(str(node), label=node_label)
                    pydot_subgraph.add_node(pydot_node)
                pydot_graph.add_subgraph(pydot_subgraph)

        for edge in nx_graph.edges():
            depth = nx_graph.edges[edge]['depth']
            mapping = nx_graph.edges[edge]['mapping']
            shape = edge[0].output_shape
            pydot_graph.add_edge(pydot.Edge(str(edge[0]), str(edge[1]), label=f"{depth}\n {shape}\n {mapping}", color="black"))


        pydot_graph.write_png(f"{self.file_name}_{safix}.png")
        del pydot_graph

    def merge_several_nodes(self, nx_graph):

        add_edges_list = []
        remove_nodes_list = []
        relabel_nodes_dict = {}

        for node in nx_graph.nodes():

            if len(list(nx_graph.successors(node))) != 0:
                successor = list(nx_graph.successors(node))[0]
                if len(list(nx_graph.successors(successor))) != 0:
                    successor_of_successor = list(nx_graph.successors(successor))[0]
                    if len(list(nx_graph.successors(successor_of_successor))) != 0:
                        successor_of_successor_of_successor = list(nx_graph.successors(successor_of_successor))[0]
                    else:
                        successor_of_successor_of_successor = None
                else:
                    successor_of_successor = None
                    successor_of_successor_of_successor = None
            else:
                successor = None
                successor_of_successor = None
                successor_of_successor_of_successor = None

            if (successor_of_successor is not None) and any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]) and any(i in successor.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]) and any(i in successor_of_successor.op_type for i in [SupportedNodes.relu, SupportedNodes.relu6]):
                norm_node = successor
                relu_node = successor_of_successor
                node_attributes = {}
                edge_attributes = {}
                node_attributes['op_type'] = [node.op_type, norm_node.op_type, relu_node.op_type]
                node_attributes['name'] = f"{'_'.join(node_attributes['op_type'])}_{self.cnt_layer_block}"
                node_attributes['dtype'] = node.dtype
                node_attributes['input_shape'] = node.input_shape
                node_attributes['output_shape'] = node.output_shape
                if any(op in node.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
                    node_attributes['kernel_shape'] = node.kernel_shape
                    node_attributes['group'] = node.group
                    node_attributes['strides'] = node.strides
                    node_attributes['pads'] = node.pads
                node_attributes['pe'] = node.pe
                node_attributes['simd'] = node.simd
                node_attributes['weight_bits'] = node.weight_bits
                node_attributes['weight_int_bits'] = node.weight_int_bits
                node_attributes['in_bits'] = node.in_bits
                node_attributes['in_int_bits'] = node.in_int_bits
                node_attributes['acc_bits'] = node.acc_bits
                node_attributes['acc_int_bits'] = node.acc_int_bits
                node_attributes['weight'] = node.weight
                node_attributes['bias'] = node.bias
                if node_attributes['bias'] is not None:
                    node_attributes['bias_bits'] = node.bias_bits
                    node_attributes['bias_int_bits'] = node.bias_int_bits
                if successor.op_type == SupportedNodes.bn:
                    node_attributes['scale_bits'] = norm_node.scale_bits
                    node_attributes['scale_int_bits'] = norm_node.scale_int_bits
                    node_attributes['shift_bits'] = norm_node.shift_bits
                    node_attributes['shift_int_bits'] = norm_node.shift_int_bits
                    node_attributes['epsilon'] = norm_node.epsilon
                    node_attributes['running_mean'] = norm_node.running_mean
                    node_attributes['running_var'] = norm_node.running_var
                    node_attributes['norm_weight'] = norm_node.norm_weight
                    node_attributes['norm_bias'] = norm_node.norm_bias
                if successor.op_type == SupportedNodes.ins:
                    node_attributes['scale_bits'] = norm_node.scale_bits
                    node_attributes['scale_int_bits'] = norm_node.scale_int_bits
                    node_attributes['shift_bits'] = norm_node.shift_bits
                    node_attributes['shift_int_bits'] = norm_node.shift_int_bits
                    node_attributes['scale'] = norm_node.scale
                    node_attributes['shift'] = norm_node.shift
                node_attributes['norm_acc_bits'] = norm_node.norm_acc_bits
                node_attributes['norm_acc_int_bits'] = norm_node.norm_acc_int_bits
                node_attributes['out_bits'] = relu_node.out_bits
                node_attributes['out_int_bits'] = relu_node.out_int_bits
                edge_attributes['depth'] = 2
                edge_attributes['mapping'] = self.default_mapping

                node_merged = NodeHLS(node_attributes)
                relabel_nodes_dict[node] = node_merged
                if len(list(nx_graph.successors(successor_of_successor))) != 0:
                    add_edges_list.append((node, successor_of_successor_of_successor, edge_attributes))
                remove_nodes_list.append(successor)
                remove_nodes_list.append(successor_of_successor)
                self.cnt_layer_block += 1

            elif (successor is not None) and any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]) and any(i in successor.op_type for i in [SupportedNodes.bn, SupportedNodes.ins]):
                norm_node = successor
                node_attributes = {}
                edge_attributes = {}
                node_attributes['op_type'] = [node.op_type, norm_node.op_type]
                node_attributes['name'] = f"{'_'.join(node_attributes['op_type'])}_{self.cnt_layer_block}"
                node_attributes['dtype'] = node.dtype
                node_attributes['input_shape'] = node.input_shape
                node_attributes['output_shape'] = node.output_shape
                if any(op in node.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
                    node_attributes['kernel_shape'] = node.kernel_shape
                    node_attributes['group'] = node.group
                    node_attributes['strides'] = node.strides
                    node_attributes['pads'] = node.pads
                node_attributes['pe'] = node.pe
                node_attributes['simd'] = node.simd
                node_attributes['weight_bits'] = node.weight_bits
                node_attributes['weight_int_bits'] = node.weight_int_bits
                node_attributes['in_bits'] = node.in_bits
                node_attributes['in_int_bits'] = node.in_int_bits
                node_attributes['acc_bits'] = node.acc_bits
                node_attributes['acc_int_bits'] = node.acc_int_bits
                node_attributes['weight'] = node.weight
                node_attributes['bias'] = node.bias
                if node_attributes['bias'] is not None:
                    node_attributes['bias_bits'] = node.bias_bits
                    node_attributes['bias_int_bits'] = node.bias_int_bits
                if successor.op_type == SupportedNodes.bn:
                    node_attributes['scale_bits'] = norm_node.scale_bits
                    node_attributes['scale_int_bits'] = norm_node.scale_int_bits
                    node_attributes['shift_bits'] = norm_node.shift_bits
                    node_attributes['shift_int_bits'] = norm_node.shift_int_bits
                    node_attributes['epsilon'] = norm_node.epsilon
                    node_attributes['running_mean'] = norm_node.running_mean
                    node_attributes['running_var'] = norm_node.running_var
                    node_attributes['norm_weight'] = norm_node.norm_weight
                    node_attributes['norm_bias'] = norm_node.norm_bias
                if successor.op_type == SupportedNodes.ins:
                    node_attributes['scale_bits'] = norm_node.scale_bits
                    node_attributes['scale_int_bits'] = norm_node.scale_int_bits
                    node_attributes['shift_bits'] = norm_node.shift_bits
                    node_attributes['shift_int_bits'] = norm_node.shift_int_bits
                    node_attributes['scale'] = norm_node.scale
                    node_attributes['shift'] = norm_node.shift
                node_attributes['norm_acc_bits'] = norm_node.norm_acc_bits
                node_attributes['norm_acc_int_bits'] = norm_node.norm_acc_int_bits
                node_attributes['out_bits'] = norm_node.out_bits
                node_attributes['out_int_bits'] = norm_node.out_int_bits
                edge_attributes['depth'] = 2
                edge_attributes['mapping'] = self.default_mapping

                node_merged = NodeHLS(node_attributes)
                relabel_nodes_dict[node] = node_merged
                if len(list(nx_graph.successors(successor))) != 0:
                    add_edges_list.append((node, successor_of_successor, edge_attributes))
                remove_nodes_list.append(successor)
                self.cnt_layer_block += 1

            elif (successor is not None) and any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]) and any(i in successor.op_type for i in [SupportedNodes.relu, SupportedNodes.relu6, SupportedNodes.mul, SupportedNodes.sigmoid, SupportedNodes.hardswish]):
                act_node = successor
                node_attributes = {}
                edge_attributes = {}
                node_attributes['op_type'] = [node.op_type, act_node.op_type]
                node_attributes['name'] = f"{'_'.join(node_attributes['op_type'])}_{self.cnt_layer_block}"
                node_attributes['dtype'] = node.dtype
                node_attributes['input_shape'] = node.input_shape
                node_attributes['output_shape'] = node.output_shape
                if any(op in node.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
                    node_attributes['kernel_shape'] = node.kernel_shape
                    node_attributes['group'] = node.group
                    node_attributes['strides'] = node.strides
                    node_attributes['pads'] = node.pads
                node_attributes['pe'] = node.pe
                node_attributes['simd'] = node.simd
                node_attributes['weight_bits'] = node.weight_bits
                node_attributes['weight_int_bits'] = node.weight_int_bits
                node_attributes['in_bits'] = node.in_bits
                node_attributes['in_int_bits'] = node.in_int_bits
                node_attributes['acc_bits'] = node.acc_bits
                node_attributes['acc_int_bits'] = node.acc_int_bits
                node_attributes['weight'] = node.weight
                node_attributes['bias'] = node.bias
                if node_attributes['bias'] is not None:
                    node_attributes['bias_bits'] = node.bias_bits
                    node_attributes['bias_int_bits'] = node.bias_int_bits
                if SupportedNodes.mul in successor.op_type:
                    node_attributes['scale'] = act_node.scale
                    node_attributes['scale_bits'] = act_node.scale_bits
                    node_attributes['scale_int_bits'] = act_node.scale_int_bits
                    node_attributes['act_acc_bits'] = act_node.act_acc_bits
                    node_attributes['act_acc_int_bits'] = act_node.act_acc_int_bits
                node_attributes['out_bits'] = act_node.out_bits
                node_attributes['out_int_bits'] = act_node.out_int_bits
                edge_attributes['depth'] = 2
                edge_attributes['mapping'] = self.default_mapping

                node_merged = NodeHLS(node_attributes)
                relabel_nodes_dict[node] = node_merged
                if len(list(nx_graph.successors(successor))) != 0:
                    add_edges_list.append((node, successor_of_successor, edge_attributes))
                remove_nodes_list.append(act_node)
                self.cnt_layer_block += 1

        nx_graph.add_edges_from(add_edges_list)
        nx_graph.remove_nodes_from(remove_nodes_list)
        nx_graph = nx.relabel_nodes(nx_graph, relabel_nodes_dict)

        return nx_graph

    def add_split_nodes(self, nx_graph):

        add_nodes_list = []
        add_edges_list = []
        remove_edges_list = []

        nx_graph_tmp = nx.DiGraph()

        for node in nx_graph.nodes():
            if len(list(nx_graph.predecessors(node))) > 3 and node.op_type != SupportedNodes.concat:
                raise ValueError(f'Nodes with {len(list(nx_graph.predecessors(node)))} > 3 predecessors are not supported yet')
            nx_graph_tmp.add_node(node)
            if len(list(nx_graph.successors(node))) > 1:
                node_attributes = {}
                edge_attributes = {}
                node_attributes['op_type'] = SupportedNodes.split
                node_attributes['name'] = f"{SupportedNodes.split}_{self.cnt_split}"
                node_attributes['dtype'] = node.dtype
                node_attributes['pe'] = node.pe
                node_attributes['simd'] = node.simd
                node_attributes['in_bits'] = node.out_bits
                node_attributes['in_int_bits'] = node.out_int_bits
                node_attributes['out_bits'] = node.out_bits
                node_attributes['out_int_bits'] = node.out_int_bits
                node_attributes['input_shape'] = node.output_shape
                node_attributes['output_shape'] = node.output_shape
                edge_attributes['depth'] = 2
                edge_attributes['mapping'] = self.default_mapping

                split_node = NodeHLS(node_attributes)
                nx_graph_tmp.add_node(split_node)
                add_nodes_list.append(split_node)
                add_edges_list.append((node, split_node, edge_attributes))
                for successor in list(nx_graph.successors(node)):
                    add_edges_list.append((split_node, successor, edge_attributes))
                    remove_edges_list.append((node, successor))
                self.cnt_split += 1

        nx_graph.add_nodes_from(add_nodes_list)
        nx_graph.add_edges_from(add_edges_list)
        nx_graph.remove_edges_from(remove_edges_list)

        for u, v, attr in nx_graph.edges(data=True):
            nx_graph_tmp.add_edge(u, v, **attr)

        return nx_graph_tmp

    def remove_unsupported_nodes(self, nx_graph, op_type_list=None):

        op_type_list = [SupportedNodes.reshape, SupportedNodes.transpose]

        remove_nodes_list = []
        add_edges_list = []

        for op_type in op_type_list:
            for node in nx_graph.nodes():
                if node.op_type == op_type:
                    edge_attributes = {}
                    edge_attributes['depth'] = 2
                    edge_attributes['mapping'] = self.default_mapping
                    remove_nodes_list.append(node)
                    predecessor = list(nx_graph.predecessors(node))[0]
                    if len(list(nx_graph.successors(node))) != 0:
                        successor = list(nx_graph.successors(node))[0]
                        add_edges_list.append((predecessor, successor, edge_attributes))
            nx_graph.add_edges_from(add_edges_list)
            nx_graph.remove_nodes_from(remove_nodes_list)
            add_edges_list.clear()
            remove_nodes_list.clear()

        # InstanceNormalization layer sometimes comes with all weights == 1 and all biases == 0
        for node in nx_graph.nodes():
            if node.op_type == SupportedNodes.ins:
                if np.all(node.scale == 1) and np.all(node.shift == 0):
                    edge_attributes = {}
                    edge_attributes['depth'] = 2
                    edge_attributes['mapping'] = self.default_mapping
                    remove_nodes_list.append(node)
                    predecessor = list(nx_graph.predecessors(node))[0]
                    if len(list(nx_graph.successors(node))) != 0:
                        successor = list(nx_graph.successors(node))[0]
                        add_edges_list.append((predecessor, successor, edge_attributes))
        nx_graph.add_edges_from(add_edges_list)
        nx_graph.remove_nodes_from(remove_nodes_list)
        add_edges_list.clear()
        remove_nodes_list.clear()

    def onnx_to_networkx(self, model_path, brevitas, stop_node=None):
        '''This function takes a path to an onnx model and returns a networkx graph representing the model
        Args:
            model_path: path to onnx model
            brevitas: Is the model quantized with Brevitas
            stop_node: stop node for the graph
        '''
        extension = os.path.splitext(model_path)[1]
        if extension != ".onnx":
            raise ValueError(f"The expected file extension is '.onnx', got {extension}.")

        self.file_name = os.path.splitext(os.path.basename(model_path))[0]
        model = onnx.load(model_path)
        model = onnx.shape_inference.infer_shapes(model)
        sess = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        value_info_shapes = {vi.name: value_info_dims(vi) for vi in model.graph.value_info}  # Shapes of nodes' outputs
        initilizers_dict = {}
        for inp in sess.get_inputs():
            value_info_shapes[inp.name] = inp.shape
        for out in sess.get_outputs():
            value_info_shapes[out.name] = out.shape
        for i, initializer in enumerate(model.graph.initializer):
            initilizers_dict[initializer.name] = numpy_helper.to_array(initializer)
        nx_graph = nx.DiGraph()
        processed_nodes = []
        for i, n in enumerate(model.graph.node):
            node_attributes = rcc_to_dict(n.attribute)

            if 'pads' in node_attributes:
                assert node_attributes['pads'][0] == node_attributes['pads'][1] == node_attributes['pads'][2] == node_attributes['pads'][3], f'Currenly only all around padding is supported, got {node_attributes["pads"]}'
            if n.op_type == 'Conv' or n.op_type == 'ConvTranspose':
                assert len(n.input) == 3 or len(n.input) == 2, f'Conv layer must have 2 or 3 inputs, got {len(n.input)}'
                node_attributes['op_type'] = SupportedNodes.conv if n.op_type == 'Conv' else SupportedNodes.tr_conv
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_conv if n.op_type == 'Conv' else self.cnt_tr_conv}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['acc_bits'] = self.default_bits
                node_attributes['acc_int_bits'] = self.default_int_bits
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                node_attributes['weight_bits'] = self.default_bits
                node_attributes['weight_int_bits'] = self.default_int_bits
                node_attributes['bias_bits'] = self.default_bits
                node_attributes['bias_int_bits'] = self.default_int_bits
                if n.op_type == 'Conv':
                    self.cnt_conv += 1
                else:
                    self.cnt_tr_conv += 1
            elif n.op_type == 'Gemm' or n.op_type == 'MatMul':
                # 'Linear' with bias from PyTorch is converted to 'Gemm' in ONNX
                # 'Linear' w/o bias from PyTorch is converted to 'MatMul' in ONNX
                assert len(n.input) == 3 or len(n.input) == 2, f'Linear layer must have 2 or 3 inputs, got {len(n.input)}'
                node_attributes['op_type'] = SupportedNodes.linear
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_linear}"
                node_attributes['dtype'] = self.dtype
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['acc_bits'] = self.default_bits
                node_attributes['acc_int_bits'] = self.default_int_bits
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                node_attributes['weight_bits'] = self.default_bits
                node_attributes['weight_int_bits'] = self.default_int_bits
                node_attributes['bias_bits'] = self.default_bits
                node_attributes['bias_int_bits'] = self.default_int_bits
                node_attributes['input_shape'] = [value_info_shapes[n.input[0]][0]] if brevitas else [value_info_shapes[n.input[0]][0], node_attributes['weight'].shape[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                self.cnt_linear += 1
            elif n.op_type == 'BatchNormalization':
                assert len(n.input) == 5, f'BatchNormalization layer must have 5 inputs, got {len(n.input)}'
                node_attributes['op_type'] = SupportedNodes.bn
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_bn}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['scale_bits'] = self.default_bits
                node_attributes['scale_int_bits'] = self.default_int_bits
                node_attributes['shift_bits'] = self.default_bits
                node_attributes['shift_int_bits'] = self.default_int_bits
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                node_attributes['norm_acc_bits'] = self.default_bits
                node_attributes['norm_acc_int_bits'] = self.default_int_bits
                node_attributes['norm_weight'] = np.copy(initilizers_dict[n.input[1]])
                node_attributes['norm_bias'] = np.copy(initilizers_dict[n.input[2]])
                node_attributes['running_mean'] = np.copy(initilizers_dict[n.input[3]])
                node_attributes['running_var'] = np.copy(initilizers_dict[n.input[4]])
                self.cnt_bn += 1
            elif n.op_type == 'InstanceNormalization':
                assert len(n.input) == 3, f'InstanceNormalization layer must have 3 inputs, got {len(n.input)}'
                node_attributes['op_type'] = SupportedNodes.ins
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_in}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                node_attributes['scale_bits'] = self.default_bits
                node_attributes['scale_int_bits'] = self.default_int_bits
                node_attributes['shift_bits'] = self.default_bits
                node_attributes['shift_int_bits'] = self.default_int_bits
                node_attributes['norm_acc_bits'] = self.default_bits
                node_attributes['norm_acc_int_bits'] = self.default_int_bits
                node_attributes['scale'] = np.copy(initilizers_dict[n.input[1]])
                node_attributes['shift'] = np.copy(initilizers_dict[n.input[2]])
                self.cnt_in += 1
            elif n.op_type == 'Relu' or n.op_type == 'Relu6':
                assert len(n.input) == 1, f'Relu layer must have 1 input, got {len(n.input)}'
                if n.op_type == 'Relu':
                    node_attributes['op_type'] = SupportedNodes.relu
                else:
                    node_attributes['op_type'] = SupportedNodes.relu6
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_relu}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['acc_bits'] = self.default_bits
                node_attributes['acc_int_bits'] = self.default_int_bits
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_relu += 1
            elif n.op_type == 'Add':
                assert len(n.input) == 2, f'Add layer must have 2 inputs, got {len(n.input)}'
                assert value_info_shapes[n.input[0]] == value_info_shapes[n.input[1]], 'Add layer with different input shapes is not supported'
                node_attributes['op_type'] = SupportedNodes.merge
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_merge}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = [self.default_bits, self.default_bits]
                node_attributes['in_int_bits'] = [self.default_int_bits, self.default_int_bits]
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_merge += 1
            elif n.op_type == 'Resize':
                assert len(n.input) == 4, f'Resize layer must have 4 inputs, got {len(n.input)}'
                ib, ich, iw, ih = value_info_shapes[n.input[0]]
                ob, och, ow, oh = value_info_shapes[n.output[0]]
                if ib == ob and ich == och and iw < ow and ih < oh:
                    node_attributes['op_type'] = SupportedNodes.upsample
                    node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_upsample}"
                    node_attributes['dtype'] = self.dtype
                    node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                    node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                    node_attributes['pe'] = DefaultValues.pe
                    node_attributes['simd'] = DefaultValues.simd
                    node_attributes['in_bits'] = self.default_bits
                    node_attributes['in_int_bits'] = self.default_int_bits
                    node_attributes['out_bits'] = self.default_bits
                    node_attributes['out_int_bits'] = self.default_int_bits
                    self.cnt_upsample += 1
                else:
                    print(f"NetworkHLS: onnx_to_networkx: unsupported layer: {n.op_type}")
                    node_attributes['op_type'] = n.op_type
                    node_attributes['name'] = n.name
                    node_attributes['dtype'] = self.dtype
                    node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                    node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                    node_attributes['pe'] = DefaultValues.pe
                    node_attributes['simd'] = DefaultValues.simd
            elif n.op_type == 'Reshape':
                node_attributes['op_type'] = SupportedNodes.reshape
                node_attributes['name'] = n.name
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
            elif n.op_type == 'Transpose':
                node_attributes['op_type'] = SupportedNodes.transpose
                node_attributes['name'] = n.name
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
            elif n.op_type == 'QuantizeLinear':
                node_attributes['op_type'] = SupportedNodes.quantize
                node_attributes['name'] = n.name
                node_attributes['var_name'] = n.input[0]
                node_attributes['scale'] = float(initilizers_dict[n.input[1]])
                node_attributes['zero_point'] = int(initilizers_dict[n.input[2]])
            elif n.op_type == 'DequantizeLinear':
                node_attributes['op_type'] = SupportedNodes.dequantize
                node_attributes['name'] = n.name
                node_attributes['var_name'] = n.input[0]
                node_attributes['scale'] = float(initilizers_dict[n.input[1]])
                node_attributes['zero_point'] = int(initilizers_dict[n.input[2]])
            elif n.op_type == 'Clip':
                if 'max' in node_attributes.keys() and node_attributes['max'] == 6.0 and node_attributes['min'] == 0.0:
                    node_attributes['op_type'] = SupportedNodes.relu6
                    node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_relu}"
                    node_attributes['dtype'] = self.dtype
                    node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                    node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                    node_attributes['pe'] = DefaultValues.pe
                    node_attributes['simd'] = DefaultValues.simd
                    node_attributes['acc_bits'] = self.default_bits
                    node_attributes['acc_int_bits'] = self.default_int_bits
                    node_attributes['in_bits'] = self.default_bits
                    node_attributes['in_int_bits'] = self.default_int_bits
                    node_attributes['out_bits'] = self.default_bits
                    node_attributes['out_int_bits'] = self.default_int_bits
                    self.cnt_relu += 1
                elif len(n.input) == 3:  # Clip with min and max as constants
                    node_attributes['max'] = int(initilizers_dict[n.input[2]])
                    node_attributes['min'] = int(initilizers_dict[n.input[1]])
                    if node_attributes['max'] == 6.0 and node_attributes['min'] == 0.0:
                        node_attributes['op_type'] = SupportedNodes.relu6
                    else:
                        node_attributes['op_type'] = SupportedNodes.clip
                    node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_relu}"
                    node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                    node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                    node_attributes['pe'] = DefaultValues.pe
                    node_attributes['simd'] = DefaultValues.simd
                    node_attributes['in_bits'] = self.default_bits
                    node_attributes['in_int_bits'] = self.default_int_bits
                    node_attributes['out_bits'] = self.default_bits
                    node_attributes['out_int_bits'] = self.default_int_bits
                    node_attributes['dtype'] = self.dtype
                    self.cnt_relu += 1
                else:
                    print(f'NetworkHLS: onnx_to_networkx: unsupported layer: {n.op_type}')
                    node_attributes['op_type'] = SupportedNodes.clip
                    node_attributes['name'] = n.name
                    node_attributes['dtype'] = self.dtype
                    node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                    node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                    node_attributes['pe'] = DefaultValues.pe
                    node_attributes['simd'] = DefaultValues.simd
            elif n.op_type == 'MaxPool':
                node_attributes['op_type'] = SupportedNodes.maxpool
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_pooling}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_pooling += 1
            elif n.op_type == 'Constant':
                initilizers_dict[n.output[0]] = onnx.numpy_helper.to_array(n.attribute[0].t)
                # Dummy values the node will be removed later
                node_attributes['name'] = n.name
                node_attributes['op_type'] = SupportedNodes.constant
                node_attributes['input_shape'] = []
                node_attributes['output_shape'] = []
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['dtype'] = self.dtype
            elif n.op_type == 'Mul':
                node_attributes['op_type'] = SupportedNodes.mul
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_mul}"
                try:
                    node_attributes['scale'] = float(initilizers_dict[n.input[1]])
                except KeyError:
                    # TODO: Add support for stream multiplication
                    raise ValueError(f"\033[91mException: Faced a layer without a scale value!\033[0m\n {n}")
                except TypeError:
                    raise TypeError(f"\033[91mException: Faced Muli value scale of size {initilizers_dict[n.input[1]].shape} !\033[0m") from exc
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['scale_bits'] = self.default_bits
                node_attributes['scale_int_bits'] = self.default_int_bits
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['act_acc_bits'] = self.default_bits
                node_attributes['act_acc_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_mul += 1
            elif n.op_type == 'Sigmoid':
                node_attributes['op_type'] = SupportedNodes.sigmoid
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_sigmoid}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_sigmoid += 1
            elif n.op_type == 'HardSwish':
                node_attributes['op_type'] = SupportedNodes.hardswish
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_hardswish}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_hardswish += 1
            elif n.op_type == 'GlobalAveragePool':
                node_attributes['op_type'] = SupportedNodes.globalavgpool
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_avgpool}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                node_attributes['acc_bits'] = self.default_bits
                node_attributes['acc_int_bits'] = self.default_int_bits
                self.cnt_avgpool += 1
            elif n.op_type == 'Expand':
                node_attributes['op_type'] = SupportedNodes.expand
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_expand}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_expand += 1
            elif n.op_type == 'Concat':
                node_attributes['op_type'] = SupportedNodes.concat
                node_attributes['name'] = f"{node_attributes['op_type']}_{self.cnt_concat}"
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
                node_attributes['in_bits'] = self.default_bits
                node_attributes['in_int_bits'] = self.default_int_bits
                node_attributes['out_bits'] = self.default_bits
                node_attributes['out_int_bits'] = self.default_int_bits
                self.cnt_concat += 1
            else:
                print(f"NetworkHLS: onnx_to_networkx: unsupported layer: {n.op_type}")
                node_attributes['op_type'] = n.op_type
                node_attributes['name'] = n.name
                node_attributes['dtype'] = self.dtype
                node_attributes['input_shape'] = value_info_shapes[n.input[0]]
                node_attributes['output_shape'] = value_info_shapes[n.output[0]]
                node_attributes['pe'] = DefaultValues.pe
                node_attributes['simd'] = DefaultValues.simd
            if n.op_type in ['Conv', 'Gemm', 'MatMul', 'ConvTranspose']:
                if brevitas:
                    node_attributes['weight'] = None
                    node_attributes['bias'] = None
                else:
                    if n.op_type in ['Conv', 'ConvTranspose']:
                        node_attributes['weight'] = np.copy(initilizers_dict[n.input[1]])
                    else:
                        node_attributes['weight'] = np.copy(np.transpose(initilizers_dict[n.input[1]]))
                    node_attributes['bias'] = None if len(n.input) == 2 else np.copy(initilizers_dict[n.input[2]])
            nx_graph.add_node(i, **node_attributes)
            processed_nodes.append(i)
            if stop_node is not None and n.name == stop_node:
                break
        # Add edges according to ONNX graph
        attribute = {}
        attribute['depth'] = 2
        attribute['mapping'] = self.default_mapping
        for source in processed_nodes:
            for out in model.graph.node[source].output:
                for dest in processed_nodes:
                    if out in model.graph.node[dest].input:
                        if dest in nx_graph.nodes():
                            nx_graph.add_edge(source, dest, **attribute)

        del value_info_shapes
        del model

        if brevitas:
            nx_graph = merge_brevitas(nx_graph, initilizers_dict)
        del initilizers_dict

        # Remove unsported nodes
        remove_nodes_list = []
        add_edges_list = []
        for op_type in [SupportedNodes.quantize, SupportedNodes.dequantize, SupportedNodes.constant, SupportedNodes.reshape]:
            for node in nx_graph.nodes():
                if nx_graph.nodes[node]['op_type'] == op_type:
                    remove_nodes_list.append(node)
                    if nx_graph.in_degree(node) != 0:
                        edge_attributes = {}
                        edge_attributes['depth'] = 2
                        edge_attributes['mapping'] = self.default_mapping
                        predecessor = list(nx_graph.predecessors(node))
                        successors = list(nx_graph.successors(node))
                        for pred in predecessor:
                            for succ in successors:
                                add_edges_list.append((pred, succ, edge_attributes))
            nx_graph.add_edges_from(add_edges_list)
            nx_graph.remove_nodes_from(remove_nodes_list)
            add_edges_list.clear()
            remove_nodes_list.clear()

        # Relable nodes from str to NodeHLS instance, and remove all attributes from the nodes
        mapping = {}
        for idx in nx_graph.nodes():
            mapping[idx] = NodeHLS(nx_graph.nodes[idx])
            nx_graph.nodes[idx].clear()
        nx_graph = nx.relabel_nodes(nx_graph, mapping)

        return nx_graph

    def generate_config_hpp(self, nx_graph, config_hpp_path=None):
        print("Generate file with configuration...")
        if config_hpp_path is not None:
            self.config_hpp_name = os.path.splitext(os.path.basename(config_hpp_path))[0]
            self.config_hpp_path = config_hpp_path
        macros = ""
        for i, node in enumerate(tqdm(nx_graph.nodes())):
            if len(list(nx_graph.predecessors(node))) == 0:
                macros += f"#ifndef {self.config_hpp_name.upper()}_HPP_\n"
                macros += f"#define {self.config_hpp_name.upper()}_HPP_\n"
                macros += "\n"
                macros += f"#define {self._float:70}{0 if self.dtype is Dtypes.FIXED else 0}\n"
                macros += "\n"
                macros += f"#define {self._input_mem_width:70}{self.input_mem_width}\n"
                macros += f"#define {self._output_mem_width:70}{self.output_mem_width}\n"
                if any(op in node.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]):
                    macros += f"#define {self._input_dim:70}{node.in_dim}\n"
                    macros += f"#define {self._input_channels:70}{node.in_channels}\n"
                macros += f"#define {self._input_bits:70}{node.in_bits}\n"
                macros += f"#define {self._input_int_bits:70}{node.in_int_bits}\n"
                macros += f"typedef ap_fixed<{self._input_bits},{self._input_int_bits},AP_RND_ZERO,AP_WRAP> {self._input_dtype};\n"
                macros += "\n"

            macros += node.get_macros()

            if len(list(nx_graph.successors(node))) == 0:
                if SupportedNodes.linear in node.op_type:
                    _output_classes = node.macros_def(self.__output_classes)
                    macros += f"#define {_output_classes:70}{node.out_features}\n"

                _output_bits = node.macros_def(self.__output_bits)
                _output_dtype = node.dtype_def(self.__output_dtype)
                macros += f"#define {_output_bits:70}{node.out_bits}\n"
                _output_int_bits = node.macros_def(self.__output_int_bits)
                macros += f"#define {_output_int_bits:70}{node.out_int_bits}\n"
                macros += f"typedef ap_fixed<{_output_bits},{_output_int_bits},AP_RND_ZERO,AP_WRAP> {_output_dtype};\n"
                macros += "\n"

            if i == len(nx_graph.nodes()) - 1:
                macros += f"#define Reps {1}\n"
                macros += "\n"
                macros += f"#define MAX_OUTPUT_CLASSES {1*12800*10}\n"
                macros += "\n"
                macros += "#endif"

        save_to_file(macros, self.config_hpp_path)

    def generate_params_hpp(self, params_hpp_path=None):
        print("Generate file with parameters...")
        if params_hpp_path is not None:
            self.params_hpp_name = os.path.splitext(os.path.basename(params_hpp_path))[0]
            self.params_hpp_path = params_hpp_path

        num_params = 0
        total_latency = 0
        for top_lelvel_idx, subgraph in enumerate(tqdm(self.subgraphs)):
            sorted_nodes = sorted(subgraph.nodes(), key=lambda node: nx.ancestors(subgraph, node))
            params = ""
            params += f"#ifndef {self.params_hpp_name.upper()}_{top_lelvel_idx}_HPP_\n"
            params += f"#define {self.params_hpp_name.upper()}_{top_lelvel_idx}_HPP_\n"
            params += "\n"
            params += f"#include \"{self.config_hpp_name.lower()}.hpp\"\n"
            params += "\n"

            for node in sorted_nodes:
                params += node.get_params()
                num_params += node.get_num_params()
                total_latency += max(node.get_latency_cc_node()) + node.get_depth_cc_node()

            params += "\n"
            params += "#endif"

            basename = os.path.splitext(self.params_hpp_path)[0]
            extension = os.path.splitext(self.params_hpp_path)[1]
            params_hpp_path = f"{basename}_{top_lelvel_idx}{extension}"
            save_to_file(params, params_hpp_path)
        print(f"Number of parameters: {num_params}")
        print(f"Total latency: {total_latency} cycles")

    def generate_profilers_hpp_cpp(self, nx_graph, profilers_hpp_path=None, profilers_cpp_path=None):
        print("Generate file with profilers...")

        if profilers_hpp_path is not None:
            self.profilers_name = os.path.splitext(os.path.basename(profilers_hpp_path))[0]
            self.profilers_hpp_path = profilers_hpp_path
        profilers = ""
        profilers += f"#ifndef {self.profilers_name.upper()}_HPP_\n"
        profilers += f"#define {self.profilers_name.upper()}_HPP_\n"
        profilers += "\n"
        profilers += f"#include \"{self.profiler.lower()}.hpp\"\n"
        profilers += "\n"
        for node in tqdm(nx_graph.nodes()):
            profilers += f"EXTERN Profiler {node.name};\n"
        profilers += "\n"
        profilers += "#endif"
        save_to_file(profilers, self.profilers_hpp_path)

        if profilers_cpp_path is not None:
            self.profilers_cpp_name = os.path.splitext(os.path.basename(profilers_cpp_path))[0]
            self.profilers_cpp_path = profilers_cpp_path
        profilers = ""
        profilers += f"#include \"{self.profilers_name.lower()}.hpp\"\n"
        profilers += "\n"
        profilers += "void get_json_params(std::string profiled_activations_path)\n"
        profilers += "{{\n"
        profilers += "\tstd::ofstream act_json;\n"
        profilers += "\tact_json.open(profiled_activations_path);\n"
        profilers += "\tact_json << \"[\" << std::endl;\n"
        for i, node in enumerate(tqdm(nx_graph.nodes())):
            if i == len(nx_graph.nodes()) - 1:
                profilers += f"\tact_json << {node.name}.get_json_string(\"{node.name}\") << std::endl;\n"
            else:
                profilers += f"\tact_json << {node.name}.get_json_string(\"{node.name}\") << \",\" << std::endl;\n"
        profilers += "\tact_json << \"]\" << std::endl;\n"
        profilers += "\tact_json.close();\n"
        profilers += "}}\n"
        save_to_file(profilers, self.profilers_cpp_path)

    def json_params_to_networkx(self, nx_graph, profiled_activations_path=None):
        #TODO: Should we add support for tr_conv here?
        try:
            assert profiled_activations_path is not None, "profiled_activations_path is None"
            with open(profiled_activations_path, 'r', encoding='utf-8') as profiled_activations:
                profiled_activations = json.load(profiled_activations)
        except FileNotFoundError:
            print(f"NetworkHLs: json_params_to_networkx: input file {profiled_activations_path} does not exist.")
            return nx_graph

        print("Reading profiled configuration from json...")
        for json_idx, node in zip(profiled_activations, nx_graph.nodes()):
            json_idx = json_idx[0]
            if json_idx['name'] == node.name:
                if node.op_type == SupportedNodes.conv or node.op_type == SupportedNodes.linear:
                    node.in_int_bits = json_idx['ia_int_bits']
                    if len(list(nx_graph.predecessors(node))) == 0:
                        node.in_bits = 8
                    else:
                        node.in_bits = node.in_int_bits + 8
                    node.weight_int_bits = json_idx['weight_int_bits']
                    node.weight_bits = 8
                    if node.bias is not None:
                        node.bias_int_bits = json_idx['bias_int_bits']
                        node.bias_bits = 8
                    node.acc_int_bits = 16#json_idx['acc_int_bits']
                    node.acc_bits = 32#node.acc_int_bits + 8
                    node.out_int_bits = json_idx['oa_int_bits']
                    node.out_bits = node.out_int_bits + 8
                elif any(i in node.op_type for i in [SupportedNodes.bn, SupportedNodes.ins, SupportedNodes.mul]) and any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.linear]):
                    node.in_int_bits = json_idx['ia_int_bits']
                    if len(list(nx_graph.predecessors(node))) == 0:
                        node.in_bits = 8
                    else:
                        node.in_bits = node.in_int_bits + 8
                    node.weight_int_bits = json_idx['weight_int_bits']
                    node.weight_bits = 8
                    if node.bias is not None:
                        node.bias_int_bits = json_idx['bias_int_bits']
                        node.bias_bits = 8
                    node.acc_int_bits = 16#json_idx['acc_int_bits']
                    node.acc_bits = 32#node.acc_int_bits + 8
                    node.scale_int_bits = json_idx['scale_int_bits']
                    node.scale_bits = node.scale_int_bits + 8
                    if not any(i in node.op_type for i in [SupportedNodes.mul]):
                        node.shift_int_bits = json_idx['shift_int_bits']
                        node.shift_bits = node.shift_int_bits + 8
                    node.norm_acc_int_bits = json_idx['act_acc_int_bits']
                    node.norm_acc_bits = node.norm_acc_int_bits + 8
                    if any(i in node.op_type for i in [SupportedNodes.relu6]):
                        node.out_int_bits = 4
                        node.out_bits = 8
                    elif any(i in node.op_type for i in [SupportedNodes.relu]):
                        node.out_int_bits = json_idx['oa_int_bits']
                        node.out_bits = 8
                    else:
                        node.out_int_bits = json_idx['oa_int_bits']
                        node.out_bits = node.out_int_bits + 8
                elif any(i in node.op_type for i in [SupportedNodes.relu, SupportedNodes.relu6, SupportedNodes.sigmoid]) and any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.linear]):
                    node.in_int_bits = json_idx['ia_int_bits']
                    if len(list(nx_graph.predecessors(node))) == 0:
                        node.in_bits = 8
                    else:
                        node.in_bits = node.in_int_bits + 8
                    node.weight_int_bits = json_idx['weight_int_bits']
                    node.weight_bits = 8
                    if node.bias is not None:
                        node.bias_int_bits = json_idx['bias_int_bits']
                        node.bias_bits = 8
                    node.acc_int_bits = 16#json_idx['acc_int_bits']
                    node.acc_bits = 32#node.acc_int_bits + 8
                    if any(i in node.op_type for i in [SupportedNodes.relu6]):
                        node.out_int_bits = 4
                        node.out_bits = 8
                    elif any(i in node.op_type for i in [SupportedNodes.relu]):
                        node.out_int_bits = json_idx['oa_int_bits']
                        node.out_bits = 8
                    else:
                        node.out_int_bits = json_idx['oa_int_bits']
                        node.out_bits = node.out_int_bits + 8
                elif any(i in node.op_type for i in [SupportedNodes.split, SupportedNodes.upsample]):
                    node.in_int_bits = json_idx['ia_int_bits']
                    node.in_bits = node.in_int_bits + 8
                    node.out_int_bits = json_idx['oa_int_bits']
                    node.out_bits = node.out_int_bits + 8
                elif node.op_type == SupportedNodes.merge:
                    node.in_int_bits = json_idx['ia_int_bits']
                    node.in_bits = node.in_int_bits + 8
                    node.fifo_in_int_bits = json_idx['fifo_ia_int_bits']
                    node.fifo_in_bits = node.fifo_in_int_bits + 8
                    node.out_int_bits = json_idx['oa_int_bits']
                    node.out_bits = node.out_int_bits + 8
                else:
                    raise ValueError(f"Unsupported layer: {node.op_type}")
            else:
                raise ValueError(f"The node in the Networkx graph name: {node.name} and op_type: {node.op_type} does not correspond to the layer from the profiled activations file name: {json_idx['name']}")

        return nx_graph

    def set_fifo_depth(self, nx_graph):
        '''
            The function sets FIFO's -skip connections- depth to the sum of depths of all nodes in the parallel path
        '''
        print("Setting FIFO's depth...")

        for node in nx_graph.nodes():
            if node.op_type == SupportedNodes.merge:
                merge = node
                nodes_path_depth = []
                edges_path_depth = []
                pair = list(nx_graph.predecessors(merge))
                split = nx.lowest_common_ancestor(nx_graph, pair[0], pair[1])
                for path in list(nx.all_simple_paths(nx_graph, split, merge)):
                    depth = [n.get_depth_cc_node() for n in path]
                    nodes_path_depth.append(sum(depth))
                for path in list(nx.all_simple_edge_paths(nx_graph, split, merge)):
                    depth = [nx_graph.edges[e]['depth'] for e in path]
                    edges_path_depth.append(sum(depth))
                path_depth = [sum(d) for d in zip(nodes_path_depth, edges_path_depth)]
                fifo_depth = max(path_depth)-min(path_depth)
                shortest_path = nx.shortest_path(nx_graph, split, merge)
                nx_graph.edges[shortest_path[-2], merge]['depth'] += fifo_depth
                #nx_graph.edges[shortest_path[-2], merge]['mapping'] = self.off_chip

        return nx_graph

    def set_parallelism(self, nx_graph, target_latency_s=(1/30), expected_freq_mhz=300, chain_rule=True):
        '''
            The function sets SIMD and PE depending on:
                1. target_latency_s - target latency in seconds, which is the reciprocal of FPS
                2. expected_freq_mhz - expected frequency in MHz
         '''
        expected_freq_mhz *= 1e6
        cycle_time = 1/expected_freq_mhz
        target_latency_cc = target_latency_s/cycle_time
        self.target_latency_cc = math.ceil(target_latency_cc)
        print(f'Setting parallelism to achieve target latency of {target_latency_s}s which is {target_latency_cc} cc...')
        for node in nx_graph.nodes():
            node_latency = node.get_latency_cc_node()
            if max(node_latency) > target_latency_cc:
                if node_latency[0] > node_latency[1]:  # Buffer is the bottleneck
                    constant = 'pe'
                elif node.group == node.in_channels:  # Conv is depthwise and mac unit is the bottleneck
                    constant = 'simd'
                else:
                    constant = None
                paralleism_factor = math.ceil(max(node_latency)/target_latency_cc)
                num_inputs = node.in_channels if any(op in node.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]) else node.in_features
                num_outputs = node.out_channels if any(op in node.op_type for op in [SupportedNodes.conv, SupportedNodes.tr_conv]) else node.out_features
                node.simd, node.pe = find_simd_pe(num_inputs, num_outputs, paralleism_factor, constant)
                print(f"Node {node.name}, Setting SIMD to {node.simd} and PE to {node.pe}, {paralleism_factor} times")
        if chain_rule:
            for node in nx_graph.nodes():
                if SupportedNodes.split in node.op_type or SupportedNodes.merge in node.op_type:
                    successors = [x for x in list(nx_graph.successors(node)) if SupportedNodes.split not in x.op_type and SupportedNodes.merge not in x.op_type]
                    predecessors = [x for x in list(nx_graph.predecessors(node)) if SupportedNodes.split not in x.op_type and SupportedNodes.merge not in x.op_type]
                    max_value = max([n.simd for n in successors] + [n.pe for n in predecessors])
                    for succ in successors:
                        succ.simd = max_value
                    for pred in predecessors:
                        pred.pe = max_value
                    node.pe = max_value
                    node.simd = max_value
                else:
                    node_successors = list(nx_graph.successors(node))
                    if len(node_successors) > 0:
                        max_value = max([n.simd for n in node_successors]+[node.pe])
                        node.pe = max_value
                        for succ in node_successors:
                            succ.simd = max_value
                    node_predecessors = [x for x in list(nx_graph.predecessors(node)) if SupportedNodes.split not in x.op_type]  # Skip split nodes for now as they can create loops, a recursive function is needed
                    if len(node_predecessors) > 0:
                        max_value = max([n.pe for n in node_predecessors]+[node.simd])
                        for pred in node_predecessors:
                            pred.pe = max_value
        return nx_graph

    def get_num_params(self, nx_graph):
        num_params = 0
        for node in nx_graph:
            num_params += node.get_num_params()
        return num_params

    def update_nodes_default(self, nx_graph, brevitas):
        '''
            The function updates the number of input and output accesses
            for all layers except Conv and Linear. Conv and Linear have accessses
            computed in constructor.

            The function can be also used for propagating PE from Conv and Linear layers to the following layers

            The function can be also used for propagating 'out_bit_width' to 'in_bit_width' of the following layers

        '''
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            if node.op_type == SupportedNodes.merge:
                merge = node
                for predecessor in list(nx_graph.predecessors(merge)):
                    if nx_graph.edges[predecessor, merge]['depth'] == 2:
                        # The functions expects that 'direct' edge has a depth of 2, while 'fifo' edge has a depth > 2
                        # The Merge inherits the number of accesses from the 'direct' edge
                        merge.input_accesses = predecessor.output_accesses
                        merge.output_accesses = predecessor.output_accesses
                        merge.simd = predecessor.pe
                        merge.pe = predecessor.pe
            elif any(i in node.op_type for i in [SupportedNodes.maxpool, SupportedNodes.upsample]):
                # MaxPool and UpSample compute the number of accesses from own params
                # But pe and simd from the previous Conv or Linear
                node.simd = list(nx_graph.predecessors(node))[0].pe
                node.pe = list(nx_graph.predecessors(node))[0].pe
            elif not any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear, SupportedNodes.expand, SupportedNodes.globalavgpool]):
                # Only Merge layer has more than one predecessor
                # All layers have pe = simd axcept Conv and Linear
                node.input_accesses = list(nx_graph.predecessors(node))[0].output_accesses
                node.output_accesses = list(nx_graph.predecessors(node))[0].output_accesses
                node.simd = list(nx_graph.predecessors(node))[0].pe
                node.pe = list(nx_graph.predecessors(node))[0].pe

        if brevitas is False:
            # Temporal solution for assigning values to input activation, output of ReLU, and weights
            sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
            for node in sorted_nodes:
                node.in_bits = 16
                node.in_int_bits = 8

                if any(i in node.op_type for i in [SupportedNodes.merge]):
                    node.fifo_in_bits = 16
                    node.fifo_in_int_bits = 8

                if len(list(nx_graph.predecessors(node))) == 0:
                    node.in_bits = 8
                    node.in_int_bits = 1

                if any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
                    node.weight_bits = 16
                    node.weight_int_bits = 6

                node.out_bits = 16
                node.out_int_bits = 8

                if any(i in node.op_type for i in [SupportedNodes.relu, SupportedNodes.relu6]):
                    node.out_bits = 8
                    node.out_int_bits = 4

                if any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
                    node.acc_bits = 64
                    node.acc_int_bits = 32

                if any(i in node.op_type for i in [SupportedNodes.split]):
                    node.out_bits = node.in_bits
                    node.out_int_bits = node.in_int_bits

        # Propagate output precision to input precision in the case of ReLu as previous activation function
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            for predecessor in list(nx_graph.predecessors(node)):
                #if any(i in predecessor.op_type for i in [SupportedNodes.relu, SupportedNodes.relu6]):
                if nx_graph.edges[predecessor, node]['depth'] == 2:
                    node.in_bits = predecessor.out_bits
                    node.in_bits_aligned = predecessor.out_bits_aligned
                    node.in_int_bits = predecessor.out_int_bits
                else:
                    node.fifo_in_bits = predecessor.out_bits
                    node.fifo_in_int_bits = predecessor.out_int_bits

                if any(i in node.op_type for i in [SupportedNodes.split, SupportedNodes.merge, SupportedNodes.maxpool]):
                    node.out_bits = node.in_bits
                    node.out_int_bits = node.in_int_bits


        # Round off-chip inputs and outputs to supported memory interface bitwidth: 32, 64, 128
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            if len(list(nx_graph.successors(node))) == 0:
                node.update_off_chip_outputs()
            else:
                for successor in list(nx_graph.successors(node)):
                    if nx_graph.edges[node, successor]['mapping'] == self.off_chip:
                        node.update_off_chip_outputs()

            if len(list(nx_graph.predecessors(node))) == 0:
                node.update_off_chip_inputs()
            else:
                for predecessor in list(nx_graph.predecessors(node)):
                    if nx_graph.edges[predecessor, node]['mapping'] == self.off_chip:
                        node.update_off_chip_inputs()

    def update_nodes(self, nx_graph):
        '''
            The function updates the number of input and output accesses
            for all layers except Conv and Linear. Conv and Linear have accessses
            computed in constructor.

            The function can be also used for propagating PE from Conv and Linear layers to the following layers

            The function can be also used for propagating 'out_bit_width' to 'in_bit_width' of the following layers

        '''
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            if node.op_type == SupportedNodes.merge:
                merge = node
                for predecessor in list(nx_graph.predecessors(merge)):
                    if nx_graph.edges[predecessor, merge]['depth'] == 2:
                        # The functions expects that 'direct' edge has a depth of 2, while 'fifo' edge has a depth > 2
                        # The Merge inherits the number of accesses from the 'direct' edge
                        merge.input_accesses = predecessor.output_accesses
                        merge.output_accesses = predecessor.output_accesses
                        merge.simd = predecessor.pe
                        merge.pe = predecessor.pe
            elif any(i in node.op_type for i in [SupportedNodes.maxpool, SupportedNodes.upsample]):
                # MaxPool and UpSample compute the number of accesses from own params
                # But pe and simd from the previous Conv or Linear
                node.simd = list(nx_graph.predecessors(node))[0].pe
                node.pe = list(nx_graph.predecessors(node))[0].pe
            elif not any(i in node.op_type for i in [SupportedNodes.conv, SupportedNodes.tr_conv, SupportedNodes.linear]):
                # Only Merge layer has more than one predecessor
                node.input_accesses = list(nx_graph.predecessors(node))[0].output_accesses
                node.output_accesses = list(nx_graph.predecessors(node))[0].output_accesses
                node.simd = list(nx_graph.predecessors(node))[0].pe
                node.pe = list(nx_graph.predecessors(node))[0].pe

        # Temporal solution for assigning values to input activation, output of ReLU, and weights
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            if any(i in node.op_type for i in [SupportedNodes.split, SupportedNodes.upsample]):
                node.out_bits = node.in_bits
                node.out_int_bits = node.in_int_bits

        # Propagate output precision to input precision in the case of ReLu as output activation function
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            for predecessor in list(nx_graph.predecessors(node)):
                if nx_graph.edges[predecessor, node]['depth'] == 2:
                    node.in_bits = predecessor.out_bits
                    node.in_int_bits = predecessor.out_int_bits
                else:
                    node.fifo_in_bits = predecessor.out_bits
                    node.fifo_in_int_bits = predecessor.out_int_bits

            if any(i in node.op_type for i in [SupportedNodes.split, SupportedNodes.upsample]):
                node.out_bits = node.in_bits
                node.out_int_bits = node.in_int_bits

        # Round off-chip inputs and outputs to supported memory interface bitwidth: 32, 64, 128
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            if len(list(nx_graph.successors(node))) == 0:
                node.update_off_chip_outputs()
            else:
                for successor in list(nx_graph.successors(node)):
                    if nx_graph.edges[node, successor]['mapping'] == self.off_chip:
                        node.update_off_chip_outputs()

            if len(list(nx_graph.predecessors(node))) == 0:
                node.update_off_chip_inputs()
            else:
                for predecessor in list(nx_graph.predecessors(node)):
                    if nx_graph.edges[predecessor, node]['mapping'] == self.off_chip:
                        node.update_off_chip_inputs()

    def create_subgraphs(self, graph, max_weight, margin, weight_attr='num_params'):
        '''
        Partition the graph into partitions of size less than max_weight * margin
            The function loops over the nodes in the graph, if it faces a split node it follows the following steps:
                if skip connection is off_chip:
                    continue partitioning normally
                else:
                    if the split-merge block fits in the current partition:
                        add it to the current partition
                    elif the split-merge block is smaller than max_weight * margin:
                        create a new partition and add the split-merge block to it
                    else:
                        move the parallel skip connection off-chip and continue partitioning normally
        Args:
            graph (nx.DiGraph): the graph to be partitioned
            max_weight (int): the maximum weight of a partition
            margin (float): the margin to be added to the maximum weight
            weight_attr (str): the attribute of the node that represents its weight

        '''
        node = next(iter(graph.nodes()))
        current_partition = []
        partition_weight = 0
        max_weight *= margin
        while node is not None:
            node_weight = getattr(node, weight_attr)
            if SupportedNodes.split in node.op_type:
                parallel_path_weight = 0
                out_edges = graph.out_edges(node, data=True)
                assert len(out_edges) == 2, f'Split node should have 2 out edges, got {len(out_edges)}'
                skip_path = None
                for e in out_edges:  # find the skip connection
                    #if e[1]._merge in e[1].op_type:
                    if SupportedNodes.merge in e[1].op_type:
                        skip_path = e
                assert skip_path is not None, 'Split node should have a skip connection'
                if skip_path[2]['mapping'] == self.on_chip:  # if the skip connection is on_chip try not to cut it
                    parallel_path = []
                    for path in nx.all_simple_paths(graph, node, skip_path[1]):
                        if len(path) > 2:  # ignore the direct path from split to merge
                            for n in path:
                                parallel_path_weight += getattr(n, weight_attr)
                                parallel_path.append(n)
                    if partition_weight + parallel_path_weight < max_weight:  # if the parallel path is small enough, put it in the same partition and continue normally
                        current_partition += parallel_path
                        partition_weight += parallel_path_weight
                        node = next(graph.successors(skip_path[1]))  # Skip the split merge block
                        continue
                    elif parallel_path_weight < max_weight:  # if the parallel path does not fit in the current partition, but it is small enough to fit in the next partition, put it in the next partition
                        self.subgraphs.append(graph.subgraph(current_partition))
                        current_partition = parallel_path
                        partition_weight = parallel_path_weight
                        node = next(graph.successors(skip_path[1]))  # Skip the split merge block
                        continue
                    else:  # if the parallel path is bigger than the partition size we have to cut the on chip skip connection
                        current_partition.append(node)
                        partition_weight += node_weight
                        print(f'The parallel path between {node.name} and {skip_path[1].name} is too big to fit in the current partition, the skip connection will be cut')
                        nx.set_edge_attributes(graph, {(node, skip_path[1]): {'mapping': self.off_chip}})
                else:  # if the skip connection is off_chip, continue normally
                    current_partition.append(node)
                    partition_weight += node_weight
            elif partition_weight + node_weight < max_weight:
                current_partition.append(node)
                partition_weight += node_weight
            else:
                self.subgraphs.append(graph.subgraph(current_partition))
                current_partition = []
                current_partition.append(node)
                partition_weight = node_weight
            node = next(graph.successors(node), None)
        self.subgraphs.append(graph.subgraph(current_partition))
        return graph

    def create_single_subgraph_default(self, nx_graph):
        print("Creating single subgraph...")
        subgraph = []
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            subgraph.append(node)
        self.subgraphs.append(nx_graph.subgraph(subgraph))
        subgraph.clear()

    def create_subgraphs_default(self, nx_graph, criteria=None):
        '''
            The function creates the subgraphs, which will correspond to separate top_level, depending on criteria, for example:
                1. number of parameters
                2. presence of a Split/Merge function, minimum number of cuts
                3. depending on the depth of a fifo
        '''
        print("Creating subgraphs...")

        subgraph = []
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: nx.ancestors(nx_graph, node))
        for node in sorted_nodes:
            if len(list(nx_graph.successors(node))) == 0:
                self.subgraphs.append(nx_graph.subgraph(subgraph))
                subgraph.clear()
                subgraph.append(node)
                self.subgraphs.append(nx_graph.subgraph(subgraph))
                subgraph.clear()
            elif any(i in node.op_type for i in [SupportedNodes.linear]):
                self.subgraphs.append(nx_graph.subgraph(subgraph))
                subgraph.clear()
                subgraph.append(node)
            elif node.op_type == SupportedNodes.split:
                self.subgraphs.append(nx_graph.subgraph(subgraph))
                subgraph.clear()
                subgraph.append(node)
            else:
                subgraph.append(node)

    def generate_top_level_wrapper_hpp(self, nx_graph, top_level_wrapper_hpp_path=None):
        '''
            The function generates the top level wrapper .hpp

        '''
        print("Generating top level wrapper hpp...")

        if top_level_wrapper_hpp_path is not None:
            self.top_level_wrapper_name = os.path.splitext(os.path.basename(top_level_wrapper_hpp_path))[0]
            self.top_level_wrapper_hpp_path = top_level_wrapper_hpp_path

        # String for a top_level_wrapper_hpp
        top_level_hpp = ""
        top_level_hpp += f"#ifndef {self.top_level_wrapper_name.upper()}_HPP\n"
        top_level_hpp += f"#define {self.top_level_wrapper_name.upper()}_HPP\n"
        top_level_hpp += "\n"
        top_level_hpp += "#include <ap_fixed.h>\n"
        top_level_hpp += "#include <ap_int.h>\n"
        top_level_hpp += "#include <hls_stream.h>\n"
        top_level_hpp += f"#include \"{self.config_hpp_name.lower()}.hpp\"\n"
        top_level_hpp += f"#include \"{self.profilers_hpp_path}\"\n"
        top_level_hpp += "\n"

        top_level_hpp += self.generate_top_level_wrapper_signature(nx_graph)
        top_level_hpp += ";\n"

        # Iterate over all top level modules
        for top_lelvel_idx, subgraph in enumerate(tqdm(self.subgraphs)):

            input_ports = []
            output_ports = []

            sorted_nodes = sorted(subgraph.nodes(), key=lambda node: nx.ancestors(subgraph, node))
            for node in sorted_nodes:

                # Generate input ports
                if len(list(nx_graph.predecessors(node))) == 0:
                    # Primary off-chip memory port
                    name = f"{node.name.lower()}"
                    input_mem_name = f"{name}_input_mem"
                    input_ports.append(f"ap_uint<{self._input_mem_width}> *{input_mem_name}") if self.dtype is not Dtypes.FLOAT else input_ports.append(f"float *{input_mem_name}")

                else:
                    # Get predecessor with more ancestors first. Predecessor with less ancestors is a shortcut
                    sorted_predecessors = sorted(nx_graph.predecessors(node), key=lambda node: nx.ancestors(nx_graph, node), reverse=True)
                    for predecessor in sorted_predecessors:
                        name = f"{predecessor.name.lower()}_{node.name.lower()}"
                        if nx_graph.edges[predecessor, node]['mapping'] == self.off_chip:
                            # Off-chip memory port
                            input_mem_name = f"{name}_input_mem"
                            if node.op_type == SupportedNodes.merge and nx_graph.edges[predecessor, node]['depth'] > 2:
                                _input_mem_width = node._fifo_input_mem_width
                            else:
                                _input_mem_width = node._input_mem_width
                            input_ports.append(f"ap_uint<{_input_mem_width}> *{input_mem_name}") if self.dtype is not Dtypes.FLOAT else input_ports.append(f"float *{input_mem_name}")
                        else:
                            # On-chip streams from one module to another
                            if subgraph.has_node(predecessor) is not True:
                                input_stream_name = f"{name}_input_stream"
                                if node.op_type == SupportedNodes.merge and nx_graph.edges[predecessor, node]['depth'] > 2:
                                    input_stream_width = node.fifo_input_stream_width
                                    _input_stream_width = node._fifo_input_stream_width

                                else:
                                    input_stream_width = node.input_stream_width
                                    _input_stream_width = node._input_stream_width
                                if predecessor.output_stream_width != input_stream_width:
                                    input_ports.append(f"hls::stream<ap_uint<{predecessor._output_stream_width}>> & {input_stream_name}") if self.dtype is not Dtypes.FLOAT else input_ports.append(f"hls::stream<float> & {input_stream_name}")
                                else:
                                    input_ports.append(f"hls::stream<ap_uint<{_input_stream_width}>> & {input_stream_name}") if self.dtype is not Dtypes.FLOAT else input_ports.append(f"hls::stream<float> & {input_stream_name}")


                # Generate output ports and outgoing streams
                if len(list(nx_graph.successors(node))) == 0:
                    # Off-chip memory port
                    name = f"{node.name.lower()}"
                    output_mem_name = f"{name}_output_mem"
                    output_ports.append(f"ap_uint<{self._output_mem_width}> *{output_mem_name}") if self.dtype is not Dtypes.FLOAT else output_ports.append(f"float *{output_mem_name}")
                else:
                    # Get successor with more descendants first. Successor with less descendants is a shortcut
                    sorted_successors = sorted(nx_graph.successors(node), key=lambda node: nx.descendants(nx_graph, node), reverse=True)
                    for successor in sorted_successors:
                        name = f"{node.name.lower()}_{successor.name.lower()}"
                        if nx_graph.edges[node, successor]['mapping'] == self.off_chip:
                            # Off-chip memory port
                            output_mem_name = f"{name}_output_mem"
                            output_ports.append(f"ap_uint<{node._output_mem_width}> *{output_mem_name}") if self.dtype is not Dtypes.FLOAT else output_ports.append(f"float *{output_mem_name}")
                        else:
                            # On-chip streams from one module to another
                            if subgraph.has_node(successor) is not True:
                                output_stream_name = f"{name}_output_stream"
                                output_ports.append(f"hls::stream<ap_uint<{node._output_stream_width}>> & {output_stream_name}") if self.dtype is not Dtypes.FLOAT else  output_ports.append(f"hls::stream<float> & {output_stream_name}")

            # Convert to a string format
            ports = input_ports + output_ports
            interface = ""
            for idx, port in enumerate(ports):
                if idx == 0 and len(ports) == 1:
                    interface += port
                elif idx == 0:
                    interface += port + ",\n"
                elif idx == len(ports)-1:
                    interface += "\t" + port
                else:
                    interface += "\t" + port + ",\n"

            top_level_hpp += f"void top_level_{top_lelvel_idx}({interface});\n"
            top_level_hpp += "\n"

        top_level_hpp += "#endif\n"
        save_to_file(top_level_hpp, self.top_level_wrapper_hpp_path)

    def generate_top_level_wrapper_cpp(self, nx_graph, top_level_wrapper_cpp_path=None):
        '''
            The function generates the top level wrapper .cpp

        '''
        print("Generating top level wrapper cpp...")

        if top_level_wrapper_cpp_path is not None:
            self.top_level_wrapper_name = os.path.splitext(os.path.basename(top_level_wrapper_cpp_path))[0]
            self.top_level_wrapper_cpp_path = top_level_wrapper_cpp_path

        # String for a top_level_wrapper_cpp
        top_level_cpp = ""
        top_level_cpp += "#include \"bnn-library.h\"\n"
        top_level_cpp += f"#include \"{self.config_hpp_name.lower()}.hpp\"\n"
        top_level_cpp += f"#include \"{self.top_level_wrapper_name.lower()}.hpp\"\n"
        for top_lelvel_idx, subgraph in enumerate(self.subgraphs):
            top_level_cpp += f"#include \"{self.params_hpp_name}_{top_lelvel_idx}.hpp\"\n"
            top_level_path = f".{self.top_level_cpp_path.split('.')[1]}_{top_lelvel_idx}.cpp"
            top_level_cpp += f"#include \"{top_level_path}\"\n"
        top_level_cpp += "\n"
        # The "top_level_wrapper" is a part of a testbench, so it is connected only to data input (test dataset) and data output (classifications)
        top_level_cpp += self.generate_top_level_wrapper_signature(nx_graph)
        top_level_cpp += "\n{\n\n"

        # Iterate over all top level modules
        for top_lelvel_idx, subgraph in enumerate(tqdm(self.subgraphs)):

            input_ports = []
            output_ports = []
            off_chip_mem = ""
            on_chip_stream = ""

            sorted_nodes = sorted(subgraph.nodes(), key=lambda node: nx.ancestors(subgraph, node))
            for node in sorted_nodes:

                # Generate input ports
                if len(list(nx_graph.predecessors(node))) == 0:
                    # Primary off-chip memory port
                    name = f"{node.name.lower()}"
                    input_mem_name = f"{name}_input_mem"
                    input_ports.append(f"{input_mem_name}")
                else:
                    # Get predecessor with more ancestors first. Predecessor with less ancestors is a shortcut
                    sorted_predecessors = sorted(nx_graph.predecessors(node), key=lambda node: nx.ancestors(nx_graph, node), reverse=True)
                    for predecessor in sorted_predecessors:
                        name = f"{predecessor.name.lower()}_{node.name.lower()}"
                        if nx_graph.edges[predecessor, node]['mapping'] == self.off_chip:
                            # Off-chip memory port
                            input_mem_name = f"{name}_mem"
                            input_ports.append(f"{input_mem_name}")
                        else:
                            # On-chip streams from one module to another
                            if subgraph.has_node(predecessor) is not True:
                                input_stream_name = f"{name}_stream"
                                input_ports.append(f"{input_stream_name}")

                # Generate output ports and outgoing streams
                if len(list(nx_graph.successors(node))) == 0:
                    # Off-chip memory port
                    name = f"{node.name.lower()}"
                    output_mem_name = f"{name}_output_mem"
                    output_ports.append(f"{output_mem_name}")
                else:
                    # Get successor with more descendants first. Successor with less descendants is a shortcut
                    sorted_successors = sorted(nx_graph.successors(node), key=lambda node: nx.descendants(nx_graph, node), reverse=True)
                    for successor in sorted_successors:
                        name = f"{node.name.lower()}_{successor.name.lower()}"
                        if nx_graph.edges[node, successor]['mapping'] == self.off_chip:
                            # Off-chip memory port
                            output_mem_name = f"{name}_mem"
                            output_ports.append(f"ap_uint<{node._output_mem_width}> *{output_mem_name}") if self.dtype is not Dtypes.FLOAT else output_ports.append(f"float*{output_mem_name}")
                            off_chip_mem += f"\tap_uint<{node._output_mem_width}> {output_mem_name}[{nx_graph.edges[node, successor]['depth']}];\n" if self.dtype is not Dtypes.FLOAT else f"\tfloat {output_mem_name}[{nx_graph.edges[node, successor]['depth']}];\n"
                            off_chip_mem += "\n"
                        else:
                            # On-chip streams from one module to another
                            if subgraph.has_node(successor) is not True:
                                output_stream_name = f"{name}_stream"
                                output_ports.append(f"{output_stream_name}")
                                on_chip_stream += f"\thls::stream<ap_uint<{node._output_stream_width}>> {output_stream_name};\n" if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {output_stream_name};\n"
                                on_chip_stream += f"#pragma HLS STREAM variable={output_stream_name} depth={nx_graph.edges[node, successor]['depth']}\n"
                                on_chip_stream += "\n"

            # Convert to a string format
            ports = input_ports + output_ports
            interface = ""
            for idx, port in enumerate(ports):
                if idx == len(ports)-1:
                    interface += "\t\t" + port
                elif idx == 0:
                    interface += port + ",\n"
                else:
                    interface += "\t\t" + port + ",\n"

            top_level_cpp += on_chip_stream
            top_level_cpp += off_chip_mem
            top_level_cpp += f"\tstd::cout << \"Top level: {top_lelvel_idx}\" << std::endl;\n"
            top_level_cpp += f"\ttop_level_{top_lelvel_idx}({interface});\n"
            top_level_cpp += "\n"

        top_level_cpp += "}\n"
        save_to_file(top_level_cpp, self.top_level_wrapper_cpp_path)

    def generate_top_level_cpp(self, nx_graph, top_level_cpp_path=None):
        '''
            The function generates the top level .cpp

            _output_mem_width is next power of 2 _output_stream_width

        '''
        print("Generating top level cpp...")

        if top_level_cpp_path is not None:
            self.top_level_cpp_name = os.path.splitext(os.path.basename(top_level_cpp_path))[0]
            self.top_level_cpp_path = top_level_cpp_path

        # Iterate over all subgraphs to generate top level modules
        for top_lelvel_idx, subgraph in enumerate(tqdm(self.subgraphs)):

            input_ports = []
            output_ports = []
            mem_pragmas = ""
            stream_pragmas = ""
            m2s_convertors = ""
            s2m_convertors = ""
            width_convertors = ""
            on_chip_streams = ""
            module_instances = ""

            sorted_nodes = sorted(subgraph.nodes(), key=lambda node: nx.ancestors(subgraph, node))
            for node in sorted_nodes:

                input_streams = []
                output_streams = []

                # Generate input ports
                if len(list(nx_graph.predecessors(node))) == 0:
                    # Primary off-chip memory port
                    name = f"{node.name.lower()}"
                    input_mem_name = f"{name}_input_mem"
                    m2s_name = f"{name}_m2s"
                    input_stream_name = f"{name}_input_stream"
                    reads_per_input_mem = f"{name}_reads_per_input_mem"
                    input_streams.append(input_stream_name)
                    # Only memory interface is instanciated as a port
                    input_ports.append(f"ap_uint<{self._input_mem_width}> *{input_mem_name}") if self.dtype is not Dtypes.FLOAT else input_ports.append(f"float *{input_mem_name}")

                    mem_pragmas += f"\tunsigned const {reads_per_input_mem} = {node.get_input_accesses(self._input_mem_width)};\n"
                    mem_pragmas += f"#pragma HLS INTERFACE m_axi offset=slave port={input_mem_name} bundle={input_mem_name} depth={reads_per_input_mem}\n"
                    mem_pragmas += f"#pragma HLS INTERFACE s_axilite port={input_mem_name} bundle=control\n"

                    assert self.input_mem_width == 32 or self.input_mem_width == 64 or self.input_mem_width == 128, f'Width of the input memory port should be 32, 64 or 128, got {self.input_mem_width}'
                    # It is expected that 'in_bits' and 'input_stream_width' are adjusted in prior, no need for 'input_mem_width'
                    assert self.input_mem_width % node.input_stream_width == 0, f'Width of the input memory port should be multiple of the input bit width, got {self.input_mem_width} % {node.input_stream_width} = {self.input_mem_width % node.input_stream_width}'

                    if self.input_mem_width != node.input_stream_width:
                        # Needs data width convertor
                        on_chip_streams += f"\thls::stream<ap_uint<{self._input_mem_width}>> {m2s_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {m2s_name};\n"
                        on_chip_streams += f"#pragma HLS STREAM variable={m2s_name} depth={2}\n"

                        on_chip_streams += f"\thls::stream<ap_uint<{node._input_stream_width}>> {input_stream_name};\n" if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {input_stream_name};\n"
                        on_chip_streams += f"#pragma HLS STREAM variable={input_stream_name} depth={2}\n"

                        m2s_convertors += f"\tMem2Stream_Batch<{reads_per_input_mem}>({input_mem_name}, {m2s_name}, Reps);\n"
                        m2s_convertors += f"\tStreamingDataWidthConverter_Batch<{self._input_mem_width}, {node._input_stream_width}, {reads_per_input_mem}>({m2s_name}, {input_stream_name}, Reps);\n"
                    else:
                        on_chip_streams += f"\thls::stream<ap_uint<{node._input_stream_width}>> {input_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {input_stream_name};\n"
                        on_chip_streams += f"#pragma HLS STREAM variable={input_stream_name} depth={2}\n"

                        m2s_convertors += f"\tMem2Stream_Batch<{reads_per_input_mem}>({input_mem_name}, {input_stream_name}, Reps);\n"

                else:
                    # Get predecessor with more ancestors first. Predecessor with less ancestors is a shortcut
                    sorted_predecessors = sorted(nx_graph.predecessors(node), key=lambda node: nx.ancestors(nx_graph, node), reverse=True)
                    for predecessor in sorted_predecessors:
                    #for predecessor in list(nx_graph.predecessors(node)):
                        name = f"{predecessor.name.lower()}_{node.name.lower()}"
                        if nx_graph.edges[predecessor, node]['mapping'] == self.off_chip:
                            # Off-chip memory port
                            input_mem_name = f"{name}_input_mem"
                            m2s_name = f"{name}_m2s"
                            input_stream_name = f"{name}_input_stream"
                            reads_per_input_mem = f"{name}_reads_per_input_mem"
                            input_streams.append(input_stream_name)
                            if node.op_type == SupportedNodes.merge and nx_graph.edges[predecessor, node]['depth'] > 2:
                                input_stream_width = node.fifo_input_stream_width
                                _input_stream_width = node._fifo_input_stream_width
                                # input_mem_width is in_bits_aligned packed to 32, 64 or 128
                                input_mem_width = node.fifo_input_mem_width
                                _input_mem_width = node._fifo_input_mem_width
                            else:
                                input_stream_width = node.input_stream_width
                                _input_stream_width = node._input_stream_width
                                # input_mem_width is in_bits_aligned packed to 32, 64 or 128
                                input_mem_width = node.input_mem_width
                                _input_mem_width = node._input_mem_width

                            # Only memory interface is instanciated as port
                            input_ports.append(f"ap_uint<{_input_mem_width}> *{input_mem_name}")  if self.dtype is not Dtypes.FLOAT else input_ports.append(f"float*{input_mem_name}")

                            # It is expected that if the source and target are decoupled (no backpressure due to off-chip memory), the source should have the same or higher parallelism than target.
                            # Otherwise, the target will read faster than source writes. The target will read rubbish.
                            assert predecessor.pe >= node.simd, f'The parallelizm of the source memory port is lower than the parallelizm of the target memory port. The target reads faster than source writes, got {predecessor.pe} < {node.simd}'
                            # input_mem_width is in_bits_aligned packed to 32, 64 or 128
                            assert input_mem_width == 32 or input_mem_width == 64 or input_mem_width == 128, f'Width of the input memory port should be 32, 64 or 128, got {input_mem_width}'
                            # It is expected that in_bits, input_stream_width, and input_mem_width are adjusted in prior
                            assert input_mem_width % input_stream_width == 0, f'Width of the input memory port should be multiple of the input bit width, got {input_mem_width} % {input_stream_width} = {input_mem_width % input_stream_width}'

                            mem_pragmas += f"\tunsigned const {reads_per_input_mem} = {node.get_input_accesses(_input_mem_width)};\n"
                            mem_pragmas += f"#pragma HLS INTERFACE m_axi offset=slave port={input_mem_name} bundle={input_mem_name} depth={reads_per_input_mem}\n"
                            mem_pragmas += f"#pragma HLS INTERFACE s_axilite port={input_mem_name} bundle=control\n"

                            if input_mem_width != input_stream_width:
                                # Needs data width convertor
                                on_chip_streams += f"\thls::stream<ap_uint<{_input_mem_width}>> {m2s_name};\n" if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {m2s_name};\n"
                                on_chip_streams += f"#pragma HLS STREAM variable={m2s_name} depth={2}\n"

                                on_chip_streams += f"\thls::stream<ap_uint<{_input_stream_width}>> {input_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {input_stream_name};\n"
                                on_chip_streams += f"#pragma HLS STREAM variable={input_stream_name} depth={2}\n"

                                m2s_convertors += f"\tMem2Stream_Batch<{reads_per_input_mem}>({input_mem_name}, {m2s_name}, Reps);\n"
                                m2s_convertors += f"\tStreamingDataWidthConverter_Batch<{_input_mem_width}, {_input_stream_width}, {reads_per_input_mem}>({m2s_name}, {input_stream_name}, Reps);\n"
                            else:
                                on_chip_streams += f"\thls::stream<ap_uint<{_input_stream_width}>> {input_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {input_stream_name};\n"
                                on_chip_streams += f"#pragma HLS STREAM variable={input_stream_name} depth={2}\n"

                                m2s_convertors += f"\tMem2Stream_Batch<{reads_per_input_mem}>({input_mem_name}, {input_stream_name}, Reps);\n"

                        else:
                            # On-chip streams from one module to another
                            if subgraph.has_node(predecessor) is not True:
                                input_stream_name = f"{name}_input_stream"
                                stream_width_convertor_name = f"{name}_stream_width_convertor"
                                reads_per_stream = f"{name}_reads_per_stream"

                                if node.op_type == SupportedNodes.merge and nx_graph.edges[predecessor, node]['depth'] > 2:
                                    input_stream_width = node.fifo_input_stream_width
                                    _input_stream_width = node._fifo_input_stream_width
                                else:
                                    input_stream_width = node.input_stream_width
                                    _input_stream_width = node._input_stream_width

                                if predecessor.output_stream_width != input_stream_width:
                                    # Needs data width convertor
                                    input_ports.append(f"hls::stream<ap_uint<{predecessor._output_stream_width}>> & {input_stream_name}")  if self.dtype is not Dtypes.FLOAT else input_ports.append(f"hls::stream<float> & {input_stream_name}")
                                    input_streams.append(stream_width_convertor_name)

                                    stream_pragmas += f"#pragma HLS INTERFACE axis port={input_stream_name}\n"

                                    # The 'stream_width_convertor_name' is an input to a module
                                    # The 'stream_width_convertor_name' is always in the closest proximity to the input module, the stream depth of 2 is enough
                                    on_chip_streams += f"\thls::stream<ap_uint<{_input_stream_width}>> {stream_width_convertor_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {stream_width_convertor_name};\n"
                                    on_chip_streams += f"#pragma HLS STREAM variable={stream_width_convertor_name} depth={2}\n"

                                    width_convertors += f"\tunsigned const {reads_per_stream} = {predecessor._output_accesses};\n"
                                    width_convertors += f"\tStreamingDataWidthConverter_Batch<{predecessor._output_stream_width}, {_input_stream_width}, {reads_per_stream}>({input_stream_name}, {stream_width_convertor_name}, Reps);\n"

                                else:
                                    input_ports.append(f"hls::stream<ap_uint<{_input_stream_width}>> & {input_stream_name}")  if self.dtype is not Dtypes.FLOAT else input_ports.append(f"hls::stream<float> & {input_stream_name}")
                                    input_streams.append(input_stream_name)

                                    stream_pragmas += f"#pragma HLS INTERFACE axis port={input_stream_name}\n"
                            else:
                                # On-chip streams inside module
                                input_stream_name = f"{name}_stream"
                                stream_width_convertor_name = f"{name}_stream_width_convertor"
                                reads_per_stream = f"{name}_reads_per_stream"

                                if node.op_type == SupportedNodes.merge and nx_graph.edges[predecessor, node]['depth'] > 2:
                                    input_stream_width = node.fifo_input_stream_width
                                    _input_stream_width = node._fifo_input_stream_width
                                else:
                                    input_stream_width = node.input_stream_width
                                    _input_stream_width = node._input_stream_width

                                if predecessor.output_stream_width != input_stream_width:
                                    # Needs data width convertor
                                    # The 'stream_width_convertor_name' is an input to a module, while 'input_stream_name' is used as input to a width convertor
                                    input_streams.append(stream_width_convertor_name)

                                    # The 'input_stream_name' has no instantiation in top level module's ports
                                    # The 'input_stream_name' is an input to a module
                                    on_chip_streams += f"\thls::stream<ap_uint<{predecessor._output_stream_width}>> {input_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {input_stream_name};\n"
                                    on_chip_streams += f"#pragma HLS STREAM variable={input_stream_name} depth={subgraph.edges[predecessor, node]['depth']}\n"

                                    # The 'stream_width_convertor_name' is an input to a module
                                    # The 'stream_width_convertor_name' is always in the closest proximity to the input module, the stream depth of 2 is enough
                                    on_chip_streams += f"\thls::stream<ap_uint<{_input_stream_width}>> {stream_width_convertor_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {stream_width_convertor_name};\n"
                                    on_chip_streams += f"#pragma HLS STREAM variable={stream_width_convertor_name} depth={2}\n"

                                    # The number of accesses is a number of reads from the predecessor
                                    width_convertors += f"\tunsigned const {reads_per_stream} = {predecessor._output_accesses};\n"
                                    width_convertors += f"\tStreamingDataWidthConverter_Batch<{predecessor._output_stream_width}, {_input_stream_width}, {reads_per_stream}>({input_stream_name}, {stream_width_convertor_name}, Reps);\n"
                                else:
                                    # The 'input_stream_name' has no instantiation in top level module's ports
                                    # The 'input_stream_name' is an input to a module
                                    input_streams.append(input_stream_name)

                                    on_chip_streams += f"\thls::stream<ap_uint<{_input_stream_width}>> {input_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {input_stream_name};\n"
                                    on_chip_streams += f"#pragma HLS STREAM variable={input_stream_name} depth={subgraph.edges[predecessor, node]['depth']}\n"



                # Generate output ports and outgoing streams
                if len(list(nx_graph.successors(node))) == 0:
                    # Off-chip stream
                    name = f"{node.name.lower()}"
                    output_mem_name = f"{name}_output_mem"
                    s2m_name = f"{name}_s2m"
                    output_stream_name = f"{name}_output_stream"
                    writes_per_output_mem = f"{name}_writes_per_output_mem"
                    writes_per_output_stream = f"{name}_writes_per_output_stream"
                    output_streams.append(output_stream_name)
                    output_ports.append(f"ap_uint<{self._output_mem_width}> *{output_mem_name}") if self.dtype is not Dtypes.FLOAT else output_ports.append(f"float *{output_mem_name}")

                    mem_pragmas += f"\tunsigned const {writes_per_output_mem} = {node.get_output_accesses(self._output_mem_width)};\n"
                    mem_pragmas += f"\tunsigned const {writes_per_output_stream} = {node._output_accesses};\n"
                    mem_pragmas += f"#pragma HLS INTERFACE m_axi offset=slave port={output_mem_name} bundle={output_mem_name} depth={writes_per_output_mem}\n"
                    mem_pragmas += f"#pragma HLS INTERFACE s_axilite port={output_mem_name} bundle=control\n"

                    assert self.output_mem_width == 32 or self.output_mem_width == 64 or self.output_mem_width == 128, f'Width of the output memory port should be 32, 64 or 128, got {self.output_mem_width}'
                    # It is expected that 'out_bits' and 'output_stream_width' are adjusted in prior, no need for 'output_mem_width'
                    assert self.output_mem_width % node.output_stream_width == 0, f'Width of the output memory port should be multiple of the output bit width, got {self.output_mem_width} % {node.output_stream_width} = {self.output_mem_width % node.output_stream_width}'

                    if self.output_mem_width != node.output_stream_width:
                        # Needs data width convertor
                        on_chip_streams += f"\thls::stream<ap_uint<{node._output_stream_width}>> {output_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {output_stream_name};\n"
                        on_chip_streams += f"#pragma HLS STREAM variable={output_stream_name} depth={2}\n"

                        on_chip_streams += f"\thls::stream<ap_uint<{self._output_mem_width}>> {s2m_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {s2m_name};\n"
                        on_chip_streams += f"#pragma HLS STREAM variable={s2m_name} depth={2}\n"

                        s2m_convertors += f"\tStreamingDataWidthConverter_Batch<{node._output_stream_width}, {self._output_mem_width}, {writes_per_output_stream}>({output_stream_name}, {s2m_name}, Reps);\n"
                        s2m_convertors += f"\tStream2Mem_Batch<{writes_per_output_mem}>({s2m_name}, {output_mem_name}, Reps);\n"
                    else:
                        on_chip_streams += f"\thls::stream<ap_uint<{node._output_stream_width}>> {output_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {output_stream_name};\n"
                        on_chip_streams += f"#pragma HLS STREAM variable={output_stream_name} depth={2}\n"

                        s2m_convertors += f"\tStream2Mem_Batch<{writes_per_output_stream}>({output_stream_name}, {output_mem_name}, Reps);\n"

                else:
                    # Get successor with more descendants first. Successor with less descendants is a shortcut
                    sorted_successors = sorted(nx_graph.successors(node), key=lambda node: nx.descendants(nx_graph, node), reverse=True)
                    for successor in sorted_successors:
                    #for successor in list(nx_graph.successors(node)):
                        name = f"{node.name.lower()}_{successor.name.lower()}"
                        if nx_graph.edges[node, successor]['mapping'] == self.off_chip:
                            # Off-chip memory port
                            output_mem_name = f"{name}_output_mem"
                            s2m_name = f"{name}_s2m"
                            output_stream_name = f"{name}_output_stream"
                            writes_per_output_mem = f"{name}_writes_per_output_mem"
                            writes_per_output_stream = f"{name}_writes_per_output_stream"
                            output_streams.append(output_stream_name)
                            output_ports.append(f"ap_uint<{node._output_mem_width}> *{output_mem_name}")  if self.dtype is not Dtypes.FLOAT else output_ports.append(f"float *{output_mem_name}")

                            # It is expected that if the source and target are decoupled (no backpressure due to off-chip memory), the source should have the same or higher parallelism than target.
                            # Otherwise, the target will read faster than source writes. The target will read rubbish.
                            assert node.pe >= successor.simd, f'The parallelizm of the source memory port is lower than the parallelizm of the target memory port. The target reads faster than source writes, got {node.pe} < {successor.simd}'
                            # output_mem_width is out_bits_aligned packed to 32, 64 or 128
                            assert node.output_mem_width == 32 or node.output_mem_width == 64 or node.output_mem_width == 128, f'Width of the output memory port should be 32, 64 or 128, got {node.output_mem_width}'
                            # It is expected that out_bits, output_stream_width, and output_mem_width are adjusted in prior
                            assert node.output_mem_width % node.output_stream_width == 0, f'Width of the output memory port should be multiple of the output bit width, got {node.output_mem_width} % {node.output_stream_width} = {node.output_mem_width % node.output_stream_width}'

                            # IMPORTANT!!!: writes_per_output_mem is a number of memory writes
                            mem_pragmas += f"\tunsigned const {writes_per_output_mem} = {node.get_output_accesses(node._output_mem_width)};\n"
                            mem_pragmas += f"\tunsigned const {writes_per_output_stream} = {node._output_accesses};\n"
                            mem_pragmas += f"#pragma HLS INTERFACE m_axi offset=slave port={output_mem_name} bundle={output_mem_name} depth={writes_per_output_mem}\n"
                            mem_pragmas += f"#pragma HLS INTERFACE s_axilite port={output_mem_name} bundle=control\n"

                            if node.output_mem_width != node.output_stream_width:
                                # Needs data width convertor
                                on_chip_streams += f"\thls::stream<ap_uint<{node._output_stream_width}>> {output_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {output_stream_name};\n"
                                on_chip_streams += f"#pragma HLS STREAM variable={output_stream_name} depth={2}\n"

                                on_chip_streams += f"\thls::stream<ap_uint<{node._output_mem_width}>> {s2m_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {s2m_name};\n"
                                on_chip_streams += f"#pragma HLS STREAM variable={s2m_name} depth={2}\n"

                                s2m_convertors += f"\tStreamingDataWidthConverter_Batch<{node._output_stream_width}, {node._output_mem_width}, {writes_per_output_stream}>({output_stream_name}, {s2m_name}, Reps);\n"
                                s2m_convertors += f"Stream2Mem_Batch<{writes_per_output_mem}>({s2m_name}, {output_mem_name}, Reps);\n"
                            else:
                                on_chip_streams += f"\thls::stream<ap_uint<{node._output_stream_width}>> {output_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {output_stream_name};\n"
                                on_chip_streams += f"#pragma HLS STREAM variable={output_stream_name} depth={2}\n"

                                s2m_convertors += f"Stream2Mem_Batch<{writes_per_output_stream}>({output_stream_name}, {output_mem_name}, Reps);\n"
                        else:
                            # On-chip streams from one module to another
                            if subgraph.has_node(successor) is not True:
                                output_stream_name = f"{name}_output_stream"
                                output_streams.append(output_stream_name)
                                output_ports.append(f"hls::stream<ap_uint<{node._output_stream_width}>> & {output_stream_name}")  if self.dtype is not Dtypes.FLOAT else output_ports.append(f"hls::stream<float> & {output_stream_name}")
                                stream_pragmas += f"#pragma HLS INTERFACE axis port={output_stream_name}\n"
                            else:
                                # On-chip streams inside module
                                output_stream_name = f"{name}_stream"
                                output_streams.append(output_stream_name)
                                #on_chip_streams += f"\thls::stream<ap_uint<{node._output_stream_width}>> {output_stream_name};\n"  if self.dtype is not Dtypes.FLOAT else f"\thls::stream<float> {output_stream_name};\n"
                                #on_chip_streams += f"#pragma HLS STREAM variable={output_stream_name} depth={subgraph.edges[node, successor]['depth']}\n"

                # Instantiate hardware modules
                # Following the precedence constraints, layers with less ancestors are instantiated earlier
                module_instances += f"\tstd::cout << \"{node.name}\" << std::endl;\n\n"
                module_instances += node.get_module_instance(input_streams, output_streams)
                module_instances += "\n"

            # Convert to a string format
            ports = input_ports + output_ports
            interface = ""
            for idx, port in enumerate(ports):
                if idx == 0 and len(ports) == 1:
                    interface += port
                elif idx == 0:
                    interface += port + ",\n"
                elif idx == len(ports)-1:
                    interface += "\t" + port
                else:
                    interface += "\t" + port + ",\n"


            top_level = ""
            top_level += "#include \"bnn-library.h\"\n"
            top_level += f"#include \"{self.config_hpp_name.lower()}.hpp\"\n"
            top_level += f"#include \"{self.params_hpp_name.lower()}_{top_lelvel_idx}.hpp\"\n"
            top_level += "\n"
            top_level += f"void top_level_{top_lelvel_idx}({interface})\n"
            top_level += "{\n"
            top_level += "\n"
            if len(mem_pragmas) == 0:
                top_level += "#pragma HLS INTERFACE ap_ctrl_none port=return\n"
            else:
                top_level += "#pragma HLS INTERFACE s_axilite port=return bundle=control\n"
            top_level += "\n"
            top_level += mem_pragmas
            top_level += "\n"
            top_level += stream_pragmas
            top_level += "\n"
            top_level += "#pragma HLS DATAFLOW\n"
            top_level += "\n"
            top_level += on_chip_streams
            top_level += "\n"
            top_level += m2s_convertors
            top_level += "\n"
            top_level += width_convertors
            top_level += "\n"
            top_level += module_instances
            top_level += "\n"
            top_level += s2m_convertors
            top_level += "\n"
            top_level += "}\n"

            basename = os.path.splitext(self.top_level_cpp_path)[0]
            extension = os.path.splitext(self.top_level_cpp_path)[1]
            top_level_cpp_path = f"{basename}_{top_lelvel_idx}{extension}"
            save_to_file(top_level, top_level_cpp_path)

    def generate_top_level_wrapper_signature(self, nx_graph):
        '''
            This function generates the top level wrapper signature to be used in top_level_wrapper.cpp and top_level_wrapper.hpp
        '''
        input_ports = []
        output_ports = []
        for node in nx_graph.nodes():
            if len(list(nx_graph.predecessors(node))) == 0:
                input_ports.append(f"ap_uint<{self._input_mem_width}> *{node.name.lower()}_input_mem") if self.dtype is not Dtypes.FLOAT else input_ports.append(f"float *{node.name.lower()}_input_mem")
            if len(list(nx_graph.successors(node))) == 0:
                try:
                    output_ports.append(f"ap_uint<{self._output_mem_width}> *{node.name.lower()}_output_mem") if self.dtype is not Dtypes.FLOAT else output_ports.append(f"float *{node.name.lower()}_output_mem")
                except AttributeError:
                    print(f"Unsupported layer, got {node.name}")
        # Convert to a string format
        ports = input_ports + output_ports
        interface = interface = ",\n\t".join(ports)
        return f"void top_level_wrapper({interface})"
