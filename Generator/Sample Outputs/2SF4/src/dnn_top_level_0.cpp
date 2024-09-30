#include "bnn-library.h"
#include "dnn_config.hpp"
#include "dnn_params_0.hpp"

void top_level_0(ap_uint<INPUT_MEM_WIDTH> *conv_bn_relu_0_input_mem,
	ap_uint<OUTPUT_MEM_WIDTH> *conv_10_output_mem)
{

#pragma HLS INTERFACE s_axilite port=return bundle=control

	unsigned const conv_bn_relu_0_reads_per_input_mem = CONV_BN_RELU_0_IFM_BITS_ALIGNED * CONV_BN_RELU_0_IFM_CH * CONV_BN_RELU_0_IFM_DIM * CONV_BN_RELU_0_IFM_DIM  / INPUT_MEM_WIDTH;
#pragma HLS INTERFACE m_axi offset=slave port=conv_bn_relu_0_input_mem bundle=conv_bn_relu_0_input_mem depth=conv_bn_relu_0_reads_per_input_mem
#pragma HLS INTERFACE s_axilite port=conv_bn_relu_0_input_mem bundle=control
	unsigned const conv_10_writes_per_output_mem = CONV_10_OFM_BITS_ALIGNED * CONV_10_OFM_CH * CONV_10_OFM_DIM * CONV_10_OFM_DIM  / OUTPUT_MEM_WIDTH;
	unsigned const conv_10_writes_per_output_stream = CONV_10_OUPUT_ACCESSESS;
#pragma HLS INTERFACE m_axi offset=slave port=conv_10_output_mem bundle=conv_10_output_mem depth=conv_10_writes_per_output_mem
#pragma HLS INTERFACE s_axilite port=conv_10_output_mem bundle=control


#pragma HLS DATAFLOW

	hls::stream<ap_uint<INPUT_MEM_WIDTH>> conv_bn_relu_0_m2s;
#pragma HLS STREAM variable=conv_bn_relu_0_m2s depth=2
	hls::stream<ap_uint<CONV_BN_RELU_0_INPUT_STREAM_WIDTH>> conv_bn_relu_0_input_stream;
#pragma HLS STREAM variable=conv_bn_relu_0_input_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_1_INPUT_STREAM_WIDTH>> conv_bn_relu_0_conv_bn_relu_1_stream;
#pragma HLS STREAM variable=conv_bn_relu_0_conv_bn_relu_1_stream depth=2
	hls::stream<ap_uint<SPLIT_0_INPUT_STREAM_WIDTH>> conv_bn_relu_1_split_0_stream;
#pragma HLS STREAM variable=conv_bn_relu_1_split_0_stream depth=2
	hls::stream<ap_uint<MAXPOOL_0_INPUT_STREAM_WIDTH>> split_0_maxpool_0_stream;
#pragma HLS STREAM variable=split_0_maxpool_0_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_2_INPUT_STREAM_WIDTH>> maxpool_0_conv_bn_relu_2_stream;
#pragma HLS STREAM variable=maxpool_0_conv_bn_relu_2_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_3_INPUT_STREAM_WIDTH>> conv_bn_relu_2_conv_bn_relu_3_stream;
#pragma HLS STREAM variable=conv_bn_relu_2_conv_bn_relu_3_stream depth=2
	hls::stream<ap_uint<SPLIT_1_INPUT_STREAM_WIDTH>> conv_bn_relu_3_split_1_stream;
#pragma HLS STREAM variable=conv_bn_relu_3_split_1_stream depth=2
	hls::stream<ap_uint<MAXPOOL_1_INPUT_STREAM_WIDTH>> split_1_maxpool_1_stream;
#pragma HLS STREAM variable=split_1_maxpool_1_stream depth=2
	hls::stream<ap_uint<CONV_RELU_4_INPUT_STREAM_WIDTH>> maxpool_1_conv_relu_4_stream;
#pragma HLS STREAM variable=maxpool_1_conv_relu_4_stream depth=2
	hls::stream<ap_uint<CONV_RELU_5_INPUT_STREAM_WIDTH>> conv_relu_4_conv_relu_5_stream;
#pragma HLS STREAM variable=conv_relu_4_conv_relu_5_stream depth=2
	hls::stream<ap_uint<CONVTRANSPOSE_RELU_6_INPUT_STREAM_WIDTH>> conv_relu_5_convtranspose_relu_6_stream;
#pragma HLS STREAM variable=conv_relu_5_convtranspose_relu_6_stream depth=2
	hls::stream<ap_uint<CONCAT_0_INPUT_STREAM_WIDTH>> convtranspose_relu_6_concat_0_stream;
#pragma HLS STREAM variable=convtranspose_relu_6_concat_0_stream depth=2
	hls::stream<ap_uint<CONCAT_0_INPUT_STREAM_WIDTH>> split_1_concat_0_stream;
#pragma HLS STREAM variable=split_1_concat_0_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_7_INPUT_STREAM_WIDTH>> concat_0_conv_bn_relu_7_stream;
#pragma HLS STREAM variable=concat_0_conv_bn_relu_7_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_8_INPUT_STREAM_WIDTH>> conv_bn_relu_7_conv_bn_relu_8_stream;
#pragma HLS STREAM variable=conv_bn_relu_7_conv_bn_relu_8_stream depth=2
	hls::stream<ap_uint<CONVTRANSPOSE_RELU_9_INPUT_STREAM_WIDTH>> conv_bn_relu_8_convtranspose_relu_9_stream;
#pragma HLS STREAM variable=conv_bn_relu_8_convtranspose_relu_9_stream depth=2
	hls::stream<ap_uint<CONCAT_1_INPUT_STREAM_WIDTH>> convtranspose_relu_9_concat_1_stream;
#pragma HLS STREAM variable=convtranspose_relu_9_concat_1_stream depth=2
	hls::stream<ap_uint<CONCAT_1_INPUT_STREAM_WIDTH>> split_0_concat_1_stream;
#pragma HLS STREAM variable=split_0_concat_1_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_10_INPUT_STREAM_WIDTH>> concat_1_conv_bn_relu_10_stream;
#pragma HLS STREAM variable=concat_1_conv_bn_relu_10_stream depth=2
	hls::stream<ap_uint<CONV_BN_RELU_11_INPUT_STREAM_WIDTH>> conv_bn_relu_10_conv_bn_relu_11_stream;
#pragma HLS STREAM variable=conv_bn_relu_10_conv_bn_relu_11_stream depth=2
	hls::stream<ap_uint<CONV_10_INPUT_STREAM_WIDTH>> conv_bn_relu_11_conv_10_stream;
#pragma HLS STREAM variable=conv_bn_relu_11_conv_10_stream depth=2
	hls::stream<ap_uint<CONV_10_OUTPUT_STREAM_WIDTH>> conv_10_output_stream;
#pragma HLS STREAM variable=conv_10_output_stream depth=2
	hls::stream<ap_uint<OUTPUT_MEM_WIDTH>> conv_10_s2m;
#pragma HLS STREAM variable=conv_10_s2m depth=2

	Mem2Stream_Batch<conv_bn_relu_0_reads_per_input_mem>(conv_bn_relu_0_input_mem, conv_bn_relu_0_m2s, Reps);
	StreamingDataWidthConverter_Batch<INPUT_MEM_WIDTH, CONV_BN_RELU_0_INPUT_STREAM_WIDTH, conv_bn_relu_0_reads_per_input_mem>(conv_bn_relu_0_m2s, conv_bn_relu_0_input_stream, Reps);


	std::cout << "Conv_Bn_Relu_0" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_0_K,
	CONV_BN_RELU_0_IFM_CH,
	CONV_BN_RELU_0_IFM_DIM,
	CONV_BN_RELU_0_OFM_CH,
	CONV_BN_RELU_0_OFM_DIM,
	CONV_BN_RELU_0_STRIDE,
	CONV_BN_RELU_0_PADDING,
	CONV_BN_RELU_0_SIMD,
	CONV_BN_RELU_0_PE,
	conv_bn_relu_0_weight_dtype,
	Slice<conv_bn_relu_0_input_dtype>,
	Slice<conv_bn_relu_0_output_dtype>
	>
	(conv_bn_relu_0_input_stream,
	conv_bn_relu_0_conv_bn_relu_1_stream,
	conv_bn_relu_0_weights,
	conv_bn_relu_0_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_0);

	std::cout << "Conv_Bn_Relu_1" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_1_K,
	CONV_BN_RELU_1_IFM_CH,
	CONV_BN_RELU_1_IFM_DIM,
	CONV_BN_RELU_1_OFM_CH,
	CONV_BN_RELU_1_OFM_DIM,
	CONV_BN_RELU_1_STRIDE,
	CONV_BN_RELU_1_PADDING,
	CONV_BN_RELU_1_SIMD,
	CONV_BN_RELU_1_PE,
	conv_bn_relu_1_weight_dtype,
	Slice<conv_bn_relu_1_input_dtype>,
	Slice<conv_bn_relu_1_output_dtype>
	>
	(conv_bn_relu_0_conv_bn_relu_1_stream,
	conv_bn_relu_1_split_0_stream,
	conv_bn_relu_1_weights,
	conv_bn_relu_1_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_1);

	std::cout << "Split_0" << std::endl;

	DuplicateStreams_Batch
	<
	SPLIT_0_INPUT_ACCESSESS,
	split_0_input_dtype
	>
	(conv_bn_relu_1_split_0_stream,
	split_0_maxpool_0_stream,
	split_0_concat_1_stream,
	Reps,
	&Split_0);

	std::cout << "MaxPool_0" << std::endl;

	Streaming_Maxpool_batch2d
	<
	MAXPOOL_0_IFM_CH,
	MAXPOOL_0_IFM_DIM,
	MAXPOOL_0_PE,
	MAXPOOL_0_K,
	MAXPOOL_0_STRIDE,
	Slice<maxpool_0_input_dtype>,
	maxpool_0_input_dtype
	>
	(split_0_maxpool_0_stream,
	maxpool_0_conv_bn_relu_2_stream,
	Reps);

	std::cout << "Conv_Bn_Relu_2" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_2_K,
	CONV_BN_RELU_2_IFM_CH,
	CONV_BN_RELU_2_IFM_DIM,
	CONV_BN_RELU_2_OFM_CH,
	CONV_BN_RELU_2_OFM_DIM,
	CONV_BN_RELU_2_STRIDE,
	CONV_BN_RELU_2_PADDING,
	CONV_BN_RELU_2_SIMD,
	CONV_BN_RELU_2_PE,
	conv_bn_relu_2_weight_dtype,
	Slice<conv_bn_relu_2_input_dtype>,
	Slice<conv_bn_relu_2_output_dtype>
	>
	(maxpool_0_conv_bn_relu_2_stream,
	conv_bn_relu_2_conv_bn_relu_3_stream,
	conv_bn_relu_2_weights,
	conv_bn_relu_2_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_2);

	std::cout << "Conv_Bn_Relu_3" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_3_K,
	CONV_BN_RELU_3_IFM_CH,
	CONV_BN_RELU_3_IFM_DIM,
	CONV_BN_RELU_3_OFM_CH,
	CONV_BN_RELU_3_OFM_DIM,
	CONV_BN_RELU_3_STRIDE,
	CONV_BN_RELU_3_PADDING,
	CONV_BN_RELU_3_SIMD,
	CONV_BN_RELU_3_PE,
	conv_bn_relu_3_weight_dtype,
	Slice<conv_bn_relu_3_input_dtype>,
	Slice<conv_bn_relu_3_output_dtype>
	>
	(conv_bn_relu_2_conv_bn_relu_3_stream,
	conv_bn_relu_3_split_1_stream,
	conv_bn_relu_3_weights,
	conv_bn_relu_3_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_3);

	std::cout << "Split_1" << std::endl;

	DuplicateStreams_Batch
	<
	SPLIT_1_INPUT_ACCESSESS,
	split_1_input_dtype
	>
	(conv_bn_relu_3_split_1_stream,
	split_1_maxpool_1_stream,
	split_1_concat_0_stream,
	Reps,
	&Split_1);

	std::cout << "MaxPool_1" << std::endl;

	Streaming_Maxpool_batch2d
	<
	MAXPOOL_1_IFM_CH,
	MAXPOOL_1_IFM_DIM,
	MAXPOOL_1_PE,
	MAXPOOL_1_K,
	MAXPOOL_1_STRIDE,
	Slice<maxpool_1_input_dtype>,
	maxpool_1_input_dtype
	>
	(split_1_maxpool_1_stream,
	maxpool_1_conv_relu_4_stream,
	Reps);

	std::cout << "Conv_Relu_4" << std::endl;

	Conv_Padding_Batch
	<
	CONV_RELU_4_K,
	CONV_RELU_4_IFM_CH,
	CONV_RELU_4_IFM_DIM,
	CONV_RELU_4_OFM_CH,
	CONV_RELU_4_OFM_DIM,
	CONV_RELU_4_STRIDE,
	CONV_RELU_4_PADDING,
	CONV_RELU_4_SIMD,
	CONV_RELU_4_PE,
	conv_relu_4_weight_dtype,
	Slice<conv_relu_4_input_dtype>,
	Slice<conv_relu_4_output_dtype>
	>
	(maxpool_1_conv_relu_4_stream,
	conv_relu_4_conv_relu_5_stream,
	conv_relu_4_weights,
	conv_relu_4_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Relu_4);

	std::cout << "Conv_Relu_5" << std::endl;

	Conv_Padding_Batch
	<
	CONV_RELU_5_K,
	CONV_RELU_5_IFM_CH,
	CONV_RELU_5_IFM_DIM,
	CONV_RELU_5_OFM_CH,
	CONV_RELU_5_OFM_DIM,
	CONV_RELU_5_STRIDE,
	CONV_RELU_5_PADDING,
	CONV_RELU_5_SIMD,
	CONV_RELU_5_PE,
	conv_relu_5_weight_dtype,
	Slice<conv_relu_5_input_dtype>,
	Slice<conv_relu_5_output_dtype>
	>
	(conv_relu_4_conv_relu_5_stream,
	conv_relu_5_convtranspose_relu_6_stream,
	conv_relu_5_weights,
	conv_relu_5_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Relu_5);

	std::cout << "ConvTranspose_Relu_6" << std::endl;

	TransposeConv
	<
	CONVTRANSPOSE_RELU_6_K,
	CONVTRANSPOSE_RELU_6_IFM_CH,
	CONVTRANSPOSE_RELU_6_IFM_DIM,
	CONVTRANSPOSE_RELU_6_OFM_CH,
	CONVTRANSPOSE_RELU_6_SIMD,
	CONVTRANSPOSE_RELU_6_PE,
	convtranspose_relu_6_weight_dtype,
	Slice<convtranspose_relu_6_input_dtype>,
	Slice<convtranspose_relu_6_output_dtype>
	>
	(conv_relu_5_convtranspose_relu_6_stream,
	convtranspose_relu_6_concat_0_stream,
	convtranspose_relu_6_weights,
	convtranspose_relu_6_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&ConvTranspose_Relu_6);

	std::cout << "Concat_0" << std::endl;

	ConcatStreams_Batch
	<
	CONCAT_0_IFM_CH,
	CONCAT_0_IFM_DIM
	>
	(split_1_concat_0_stream,
	convtranspose_relu_6_concat_0_stream,
	concat_0_conv_bn_relu_7_stream,
	Reps);

	std::cout << "Conv_Bn_Relu_7" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_7_K,
	CONV_BN_RELU_7_IFM_CH,
	CONV_BN_RELU_7_IFM_DIM,
	CONV_BN_RELU_7_OFM_CH,
	CONV_BN_RELU_7_OFM_DIM,
	CONV_BN_RELU_7_STRIDE,
	CONV_BN_RELU_7_PADDING,
	CONV_BN_RELU_7_SIMD,
	CONV_BN_RELU_7_PE,
	conv_bn_relu_7_weight_dtype,
	Slice<conv_bn_relu_7_input_dtype>,
	Slice<conv_bn_relu_7_output_dtype>
	>
	(concat_0_conv_bn_relu_7_stream,
	conv_bn_relu_7_conv_bn_relu_8_stream,
	conv_bn_relu_7_weights,
	conv_bn_relu_7_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_7);

	std::cout << "Conv_Bn_Relu_8" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_8_K,
	CONV_BN_RELU_8_IFM_CH,
	CONV_BN_RELU_8_IFM_DIM,
	CONV_BN_RELU_8_OFM_CH,
	CONV_BN_RELU_8_OFM_DIM,
	CONV_BN_RELU_8_STRIDE,
	CONV_BN_RELU_8_PADDING,
	CONV_BN_RELU_8_SIMD,
	CONV_BN_RELU_8_PE,
	conv_bn_relu_8_weight_dtype,
	Slice<conv_bn_relu_8_input_dtype>,
	Slice<conv_bn_relu_8_output_dtype>
	>
	(conv_bn_relu_7_conv_bn_relu_8_stream,
	conv_bn_relu_8_convtranspose_relu_9_stream,
	conv_bn_relu_8_weights,
	conv_bn_relu_8_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_8);

	std::cout << "ConvTranspose_Relu_9" << std::endl;

	TransposeConv
	<
	CONVTRANSPOSE_RELU_9_K,
	CONVTRANSPOSE_RELU_9_IFM_CH,
	CONVTRANSPOSE_RELU_9_IFM_DIM,
	CONVTRANSPOSE_RELU_9_OFM_CH,
	CONVTRANSPOSE_RELU_9_SIMD,
	CONVTRANSPOSE_RELU_9_PE,
	convtranspose_relu_9_weight_dtype,
	Slice<convtranspose_relu_9_input_dtype>,
	Slice<convtranspose_relu_9_output_dtype>
	>
	(conv_bn_relu_8_convtranspose_relu_9_stream,
	convtranspose_relu_9_concat_1_stream,
	convtranspose_relu_9_weights,
	convtranspose_relu_9_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&ConvTranspose_Relu_9);

	std::cout << "Concat_1" << std::endl;

	ConcatStreams_Batch
	<
	CONCAT_1_IFM_CH,
	CONCAT_1_IFM_DIM
	>
	(split_0_concat_1_stream,
	convtranspose_relu_9_concat_1_stream,
	concat_1_conv_bn_relu_10_stream,
	Reps);

	std::cout << "Conv_Bn_Relu_10" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_10_K,
	CONV_BN_RELU_10_IFM_CH,
	CONV_BN_RELU_10_IFM_DIM,
	CONV_BN_RELU_10_OFM_CH,
	CONV_BN_RELU_10_OFM_DIM,
	CONV_BN_RELU_10_STRIDE,
	CONV_BN_RELU_10_PADDING,
	CONV_BN_RELU_10_SIMD,
	CONV_BN_RELU_10_PE,
	conv_bn_relu_10_weight_dtype,
	Slice<conv_bn_relu_10_input_dtype>,
	Slice<conv_bn_relu_10_output_dtype>
	>
	(concat_1_conv_bn_relu_10_stream,
	conv_bn_relu_10_conv_bn_relu_11_stream,
	conv_bn_relu_10_weights,
	conv_bn_relu_10_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_10);

	std::cout << "Conv_Bn_Relu_11" << std::endl;

	Conv_Padding_Batch
	<
	CONV_BN_RELU_11_K,
	CONV_BN_RELU_11_IFM_CH,
	CONV_BN_RELU_11_IFM_DIM,
	CONV_BN_RELU_11_OFM_CH,
	CONV_BN_RELU_11_OFM_DIM,
	CONV_BN_RELU_11_STRIDE,
	CONV_BN_RELU_11_PADDING,
	CONV_BN_RELU_11_SIMD,
	CONV_BN_RELU_11_PE,
	conv_bn_relu_11_weight_dtype,
	Slice<conv_bn_relu_11_input_dtype>,
	Slice<conv_bn_relu_11_output_dtype>
	>
	(conv_bn_relu_10_conv_bn_relu_11_stream,
	conv_bn_relu_11_conv_10_stream,
	conv_bn_relu_11_weights,
	conv_bn_relu_11_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_Bn_Relu_11);

	std::cout << "Conv_10" << std::endl;

	Conv_Pw_Batch
	<
	CONV_10_K,
	CONV_10_IFM_CH,
	CONV_10_IFM_DIM,
	CONV_10_OFM_CH,
	CONV_10_OFM_DIM,
	CONV_10_STRIDE,
	CONV_10_PADDING,
	CONV_10_SIMD,
	CONV_10_PE,
	conv_10_weight_dtype,
	Slice<conv_10_input_dtype>,
	Slice<conv_10_output_dtype>
	>
	(conv_bn_relu_11_conv_10_stream,
	conv_10_output_stream,
	conv_10_weights,
	conv_10_activations,
	Reps,
	ap_resource_dflt(),
	0,
	&Conv_10);


	StreamingDataWidthConverter_Batch<CONV_10_OUTPUT_STREAM_WIDTH, OUTPUT_MEM_WIDTH, conv_10_writes_per_output_stream>(conv_10_output_stream, conv_10_s2m, Reps);
	Stream2Mem_Batch<conv_10_writes_per_output_mem>(conv_10_s2m, conv_10_output_mem, Reps);

}
