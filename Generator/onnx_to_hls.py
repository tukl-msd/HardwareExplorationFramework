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
#  \file onnx_to_hls.py
#
#  Hardware Exploration Framework 
#  This module provides a command-line interface for converting ONNX models
#  to High-Level Synthesis (HLS) code.
#  
#  The following code has been developed for a work package "Hardware Architectures
#  for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
#  Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
#  Systems). The project has received funding from European Union’s Horizon Europe
#  research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
#  under grant agreement No 101070408.
#
####################################################################################

"""
Module: onnx_to_hls

This module performs various transformations and optimizations on the ONNX model,
such as removing unsupported nodes, adding split nodes, merging nodes, setting FIFO depth, and dividing the model into multiple top_levels.
The final HLS model is saved as .hpp and .cpp files.

Classes:
    NetworkHLS: A class that provides methods to convert ONNX models to HLS models.

Functions:
    main(): Parses command-line arguments and performs the conversion from ONNX to HLS.

Command-line Arguments:
    --model: Path to an ONNX model (required).
    --result_hpp_path: Path to save the generated .hpp files (default: './include_hw/').
    --result_cpp_path: Path to save the generated .cpp files (default: './src_hw/').
    --input_mem_width: Input memory width (default: 64).
    --output_mem_width: Output memory width (default: 64).
    --max_subgraph_weight: Maximum subgraph weight (default: 25e4).
    --subgraph_weight_margin: Allowed subgraph weight margin (default: 1.1).
    --brevitas: Flag indicating if the model is a Brevitas model (default: False).
    --int: Flag indicating if the model is a quantized ONNX model using QDQ method (default: False).
    --float: Flag indicating if the model should be treated as a float model (default: False).
    --scale_bit_width: Bit width for scale (default: 16).
    --profiler_json: Path to the profiler JSON file (default: './profiled_activations.json').
    --stop_node: Node that will be the last node in the graph (default: '').
"""
import argparse
from pathlib import Path
from network_hls import NetworkHLS
from defaults import Dtypes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic ONNX based HLS model generator')
    parser.add_argument('--model', type=Path, required=True, help='path to an ONNX model')
    parser.add_argument('--result_hpp_path', type=Path, default='./include_hw/', help='path to save the generated .hpp files')
    parser.add_argument('--result_cpp_path', type=Path, default='./src_hw/', help='path to save the generated .cpp files')
    parser.add_argument('--input_mem_width', type=int, help='input memory width the default is 64', default=64)
    parser.add_argument('--output_mem_width', type=int, help='output memory width the default is 64', default=64)
    parser.add_argument('--max_subgraph_weight', type=float, help='max subgraph weight the default is 25e4', default=25e4)
    parser.add_argument('--subgraph_weight_margin', type=float, help='allowed subgraph weight margin the default is 1.1', default=1.1)
    parser.add_argument('--brevitas', action='store_true', help='brevitas model', default=False)
    parser.add_argument('--float', action='store_true', help='treat the model as float', default=False)
    parser.add_argument('--profiler_json', type=str, help='path to the profiler json file', default='./profiled_activations.json')
    parser.add_argument('--stop_node', type=str, help='This node will be the last node in the graph', default='')
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.result_hpp_path = args.result_hpp_path.resolve()
    args.result_cpp_path = args.result_cpp_path.resolve()
    assert args.model.is_file(), 'Model file does not exist'
    if args.float:
        dtype = Dtypes.FLOAT
    else:
        dtype = Dtypes.FIXED
    networkhls = NetworkHLS(dtype=dtype)
    nx_graph = networkhls.onnx_to_networkx(args.model, args.brevitas, args.stop_node)

    # Visualize the graph
    networkhls.networkx_to_png(nx_graph, "after_onnx")

    # Remove unsupported layers, currently "Reshape", "Transpose", ..
    networkhls.remove_unsupported_nodes(nx_graph)
    #networkhls.networkx_to_png(nx_graph, "after_removing_unsupported_nodes")

    # Add Split layer
    nx_graph = networkhls.add_split_nodes(nx_graph)
    #networkhls.networkx_to_png(nx_graph, "after_adding_split")

    # Merge several layers to one, like
    # [Conv/Linear Bn/In, Relu]
    # [Conv/Linear, Bn/In]
    # [Conv/Linear, Relu]
    nx_graph = networkhls.merge_several_nodes(nx_graph)
    #networkhls.networkx_to_png(nx_graph, "after_merging_nodes")

    # Set a depth of the FIFOs on a path between Split and Merge
    nx_graph = networkhls.set_fifo_depth(nx_graph)
    networkhls.networkx_to_png(nx_graph, "after_setting_fifo_depth")

    # Set a parallelism of the layers
    #nx_graph = networkhls.set_parallelism(nx_graph, expected_freq_mhz=100)
    #networkhls.networkx_to_png(nx_graph, "after_setting_parallelism")

    # Create subgraphs
    networkhls.create_single_subgraph_default(nx_graph)
    #networkhls.create_subgraphs_default(nx_graph)
    #networkhls.create_subgraphs(nx_graph, args.max_subgraph_weight, args.subgraph_weight_margin)
    networkhls.networkx_to_png(nx_graph, "after_creating_subgraphs")

    networkhls.update_nodes_default(nx_graph, args.brevitas)
    networkhls.generate_top_level_wrapper_hpp(nx_graph, args.result_hpp_path / 'dnn_top_level_wrapper.hpp')

    networkhls.generate_top_level_wrapper_cpp(nx_graph, args.result_cpp_path / 'dnn_top_level_wrapper.cpp')

    networkhls.generate_top_level_cpp(nx_graph, args.result_cpp_path / 'dnn_top_level.cpp')

    # Generate profilers file
    networkhls.generate_profilers_hpp_cpp(nx_graph, args.result_hpp_path / 'dnn_profilers.hpp', args.result_cpp_path / 'dnn_profilers.cpp')

    # Generate configuration file
    networkhls.generate_config_hpp(nx_graph, args.result_hpp_path / 'dnn_config.hpp')

    # Generate params file
    networkhls.generate_params_hpp(args.result_hpp_path / 'dnn_params.hpp')

    print('\nDo you have a valid .json file with profiled activations? Would you like to continue....press any key or abort Ctrl+C')
    input()

    # Update the attributes with values based on hardware profiling
    nx_graph = networkhls.json_params_to_networkx(nx_graph, './profiled_activations.json')

    networkhls.update_nodes(nx_graph)
    networkhls.generate_top_level_wrapper_hpp(nx_graph, args.result_hpp_path /'dnn_top_level_wrapper.hpp')

    networkhls.generate_top_level_wrapper_cpp(nx_graph, args.result_cpp_path /'dnn_top_level_wrapper.cpp')

    networkhls.generate_top_level_cpp(nx_graph, args.result_cpp_path /'dnn_top_level.cpp')

    # Generate profilers file
    networkhls.generate_profilers_hpp_cpp(nx_graph, args.result_hpp_path / 'dnn_profilers.hpp', args.result_cpp_path / 'dnn_profilers.cpp')

    # Generate configuration file
    networkhls.generate_config_hpp(nx_graph, args.result_hpp_path / 'dnn_config.hpp')

    # Generate params file
    networkhls.generate_params_hpp(args.result_hpp_path / 'dnn_params.hpp')
