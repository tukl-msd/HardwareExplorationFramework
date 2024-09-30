#include "dnn_profilers.hpp"

void get_json_params(std::string profiled_activations_path)
{{
	std::ofstream act_json;
	act_json.open(profiled_activations_path);
	act_json << "[" << std::endl;
	act_json << Conv_Bn_Relu_0.get_json_string("Conv_Bn_Relu_0") << "," << std::endl;
	act_json << Conv_Bn_Relu_1.get_json_string("Conv_Bn_Relu_1") << "," << std::endl;
	act_json << Split_0.get_json_string("Split_0") << "," << std::endl;
	act_json << MaxPool_0.get_json_string("MaxPool_0") << "," << std::endl;
	act_json << Conv_Bn_Relu_2.get_json_string("Conv_Bn_Relu_2") << "," << std::endl;
	act_json << Conv_Bn_Relu_3.get_json_string("Conv_Bn_Relu_3") << "," << std::endl;
	act_json << Split_1.get_json_string("Split_1") << "," << std::endl;
	act_json << MaxPool_1.get_json_string("MaxPool_1") << "," << std::endl;
	act_json << Conv_Relu_4.get_json_string("Conv_Relu_4") << "," << std::endl;
	act_json << Conv_Relu_5.get_json_string("Conv_Relu_5") << "," << std::endl;
	act_json << ConvTranspose_Relu_6.get_json_string("ConvTranspose_Relu_6") << "," << std::endl;
	act_json << Concat_0.get_json_string("Concat_0") << "," << std::endl;
	act_json << Conv_Bn_Relu_7.get_json_string("Conv_Bn_Relu_7") << "," << std::endl;
	act_json << Conv_Bn_Relu_8.get_json_string("Conv_Bn_Relu_8") << "," << std::endl;
	act_json << ConvTranspose_Relu_9.get_json_string("ConvTranspose_Relu_9") << "," << std::endl;
	act_json << Concat_1.get_json_string("Concat_1") << "," << std::endl;
	act_json << Conv_Bn_Relu_10.get_json_string("Conv_Bn_Relu_10") << "," << std::endl;
	act_json << Conv_Bn_Relu_11.get_json_string("Conv_Bn_Relu_11") << "," << std::endl;
	act_json << Conv_10.get_json_string("Conv_10") << std::endl;
	act_json << "]" << std::endl;
	act_json.close();
}}
