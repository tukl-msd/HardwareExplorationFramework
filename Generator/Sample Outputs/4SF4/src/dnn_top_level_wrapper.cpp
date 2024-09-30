#include "bnn-library.h"
#include "dnn_config.hpp"
#include "dnn_top_level_wrapper.hpp"
#include "dnn_params_0.hpp"
#include "./dnn_top_level_0.cpp"

void top_level_wrapper(ap_uint<INPUT_MEM_WIDTH> *conv_bn_relu_0_input_mem,
	ap_uint<OUTPUT_MEM_WIDTH> *conv_18_output_mem)
{

	std::cout << "Top level: 0" << std::endl;
	top_level_0(conv_bn_relu_0_input_mem,
		conv_18_output_mem);

}
