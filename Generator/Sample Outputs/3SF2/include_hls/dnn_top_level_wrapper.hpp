#ifndef DNN_TOP_LEVEL_WRAPPER_HPP
#define DNN_TOP_LEVEL_WRAPPER_HPP

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include "dnn_config.hpp"
#include "./dnn_profilers.hpp"

void top_level_wrapper(ap_uint<INPUT_MEM_WIDTH> *conv_bn_relu_0_input_mem,
	ap_uint<OUTPUT_MEM_WIDTH> *conv_14_output_mem);
void top_level_0(ap_uint<INPUT_MEM_WIDTH> *conv_bn_relu_0_input_mem,
	ap_uint<OUTPUT_MEM_WIDTH> *conv_14_output_mem);

#endif
