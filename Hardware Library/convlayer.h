/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *****************************************************************************/

/***********************************************************************************
*
*   The following code is partially derived from the code provided by Xilinx, Inc.
*   in https://github.com/Xilinx/finn-hlslib.
*
*   Significant modifications and additions have been made to the original code to
*   suit the specific needs of SustainML project https://sustainml.eu.
*
*   The following code has been developed for a work package "Hardware Architectures
*   for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
*   Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
*   Systems). The project has received funding from European Union’s Horizon Europe
*   research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
*   under grant agreement No 101070408.
*
 **********************************************************************************/

/***********************************************************************************
 *  Copyright (C) 2024, University of Kaiserslautern-Landau (RPTU), Kaiserslautern
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ************************************************************************************/

/***********************************************************************************
 *
 *  Authors: Vladimir Rybalkin <rybalkin@rptu.de>
 *           Mohamed Moursi <mmoursi@rptu.de>
 *
 *  \file convlayer.h
 *
 *  Library of templated HLS functions for DNN deployment.
 *  This file lists a set of convenience functions used to implement
 *  convolutional layers.
 *
 ************************************************************************************/


#ifndef CONVLAYER_H
#define CONVLAYER_H

#include <ap_int.h>
#include <hls_stream.h>

#include "streamtools.h"
#include "mvau.hpp"
#include "tmrcheck.hpp"

/*
********************************************************************************************************************************************************
* Conv_Pw_Batch, when ConvKernelDim = 1x1.
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,
  unsigned int Stride,  // !!!: always 1
  unsigned int Padding, // !!!: always 0 
  unsigned int SIMD,
  unsigned int PE, 
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Pw_Batch(hls::stream<ap_uint<InStreamW>>  &in,
                   hls::stream<ap_uint<OutStreamW>> &out,
                   TW const &weights,
                   TA const &activation,
                   unsigned const reps,
                   R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;  
  
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_in (in,  reps);// wa_in <- in  
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);// mvOut -> out
  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("ConvolutionInputGenerator_kernel1.Matrix_Vector_Activate_Batch.convInp");

  ConvolutionInputGenerator_kernel1<IFMChannels, TSrcI::width, IFMDim, SIMD, 1>(wa_in, convInp, reps);
   
  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Pw_Batch_StreamWeights, when ConvKernelDim = 1x1. "weights" is a stream
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,	
  unsigned int Stride,  // !!!: always 1
  unsigned int Padding, // !!!: always 0  
  unsigned int SIMD,
  unsigned int PE,  
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Pw_Batch_StreamWeights(hls::stream<ap_uint<InStreamW>>  &in,
                                 hls::stream<ap_uint<OutStreamW>> &out,
                                 hls::stream<ap_uint<PE*SIMD*Tw::width>> &weights,
                                 TA const &activation,
                                 unsigned const reps,
                                 R const &r, int flag = 0, Profiler *profiler = nullptr)
{

  CASSERT_DATAFLOW(Stride == 1);
  CASSERT_DATAFLOW(Padding == 0);

#pragma HLS INLINE

  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;
   
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_in (in,  reps);
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("ConvolutionInputGenerator_kernel1.Matrix_Vector_Activate_Batch.convInp");

  ConvolutionInputGenerator_kernel1<IFMChannels, TSrcI::width, IFMDim, SIMD, 1>(wa_in, convInp, reps);   
   
  Matrix_Vector_Activate_Batch_StreamWeights<MatrixW, MatrixH, SIMD, PE, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Pw_Batch_StreamWeights, when ConvKernelDim = 1x1. "weights" is a off-chip memory
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,	
  unsigned int Stride,  // !!!: always 1
  unsigned int Padding, // !!!: always 0  
  unsigned int SIMD,
  unsigned int PE,  
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (width must be int though!)
  int WeightMemW, int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Pw_Batch_OffChipWeights(hls::stream<ap_uint<InStreamW>>  &in_stream,
                                  hls::stream<ap_uint<OutStreamW>> &out_stream,
                                  ap_uint<WeightMemW> *weight_mem,
                                  TA const &activation,
                                  unsigned const reps,
                                  R const &r, int flag = 0, Profiler *profiler = nullptr)
{
  
  CASSERT_DATAFLOW(WeightMemW % Tw::width == 0);
  CASSERT_DATAFLOW(Stride == 1);
  CASSERT_DATAFLOW(Padding == 0);

#pragma HLS INLINE

  unsigned const ReadsPerWeightMem = (IFMChannels * OFMChannels * ConvKernelDim * ConvKernelDim * Tw::width + WeightMemW - 1) / WeightMemW;

  hls::stream<ap_uint<WeightMemW>> m2s_out("m2s_out");
  Mem2Stream_Batch<ReadsPerWeightMem>(weight_mem, m2s_out, OFMDim * OFMDim * reps);
  WidthAdjustedInputStream <WeightMemW, SIMD*PE*Tw::width, ReadsPerWeightMem>  weight_stream (m2s_out, OFMDim * OFMDim * reps);	

	Conv_Pw_Batch_StreamWeights
  <
  ConvKernelDim,
  IFMChannels,
  IFMDim,
  OFMChannels,
  OFMDim,
  Stride,
  Padding,
  SIMD,
  PE,
  Tw,
  TSrcI,
  TDstI
  >
  (in_stream,
  out_stream,
  weight_stream,
  activation,
  reps,
  r, flag, profiler);	
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Batch, when ConvKernelDim % Stride = 0 and Padding = 0.
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,
  unsigned int Stride,	
  unsigned int Padding, // !!!: always 0		
  unsigned int SIMD,
  unsigned int PE,		
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Dw_Batch(hls::stream<ap_uint<InStreamW>> & in,
                   hls::stream<ap_uint<OutStreamW>> & out,
                   TW const & weights,
                   TA const & activation,
                   unsigned const reps,
                   R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE
	unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;

	WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_in (in,  reps);	
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
	hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("ConvolutionInputGenerator_dws.Vector_Vector_Activate_Batch.convInp");

	ConvolutionInputGenerator_dws<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim, SIMD, Stride>
		(wa_in, convInp, reps, ap_resource_dflt());	

	Vector_Vector_Activate_Batch<IFMChannels, ConvKernelDim*ConvKernelDim, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
		weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Batch_StreamWeights, when ConvKernelDim % Stride = 0 and Padding = 0. "weights" is a stream
********************************************************************************************************************************************************
*/
template
<
unsigned int ConvKernelDim,		
unsigned int IFMChannels,		
unsigned int IFMDim,			
unsigned int OFMChannels,		
unsigned int OFMDim,
unsigned int Stride,
unsigned int Padding, // !!!: always 0	
unsigned int SIMD,
unsigned int PE,
typename Tw,
typename TSrcI = Identity,
typename TDstI = Identity,
typename TWeightI = Identity,
// safely deducible (stream width must be int though!)
int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Dw_Batch_StreamWeights(hls::stream<ap_uint<InStreamW>> & in,
                                 hls::stream<ap_uint<OutStreamW>> & out,
                                 hls::stream<ap_uint<PE*Tw::width>> & weights,
                                 TA const & activation,
                                 unsigned const reps,
                                 R const & r, int flag = 0, Profiler *profiler = nullptr)
{
  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE
	unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;

	WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_in (in,  reps);	
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
	hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_dws.Vector_Vector_Activate_Batch.convInp");

	ConvolutionInputGenerator_dws<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim, SIMD, Stride>
		(wa_in, convInp, reps, ap_resource_dflt());
	
	Vector_Vector_Activate_Batch_StreamWeights<IFMChannels, ConvKernelDim*ConvKernelDim, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
		weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Batch_StreamWeights, when ConvKernelDim % Stride = 0 and Padding = 0. "weights" is a off-chip memory
********************************************************************************************************************************************************
*/
template
<
unsigned int ConvKernelDim,		
unsigned int IFMChannels,		
unsigned int IFMDim,			
unsigned int OFMChannels,		
unsigned int OFMDim,
unsigned int Stride,
unsigned int Padding, // !!!: always 0	
unsigned int SIMD,
unsigned int PE,
typename Tw,
typename TSrcI = Identity,
typename TDstI = Identity,
typename TWeightI = Identity,
// safely deducible (width must be int though!)
int WeightMemW, int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Dw_Batch_OffChipWeights(hls::stream<ap_uint<InStreamW>> & in_stream,
                                  hls::stream<ap_uint<OutStreamW>> & out_stream,
                                  ap_uint<WeightMemW> *weight_mem,
                                  TA const & activation,
                                  unsigned const reps,
                                  R const & r, int flag = 0, Profiler *profiler = nullptr)
{
  CASSERT_DATAFLOW(WeightMemW % Tw::width == 0);
  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE

  unsigned const ReadsPerWeightMem = (OFMChannels * ConvKernelDim * ConvKernelDim * Tw::width + WeightMemW - 1) / WeightMemW;

  hls::stream<ap_uint<WeightMemW>> m2s_out("m2s_out");
  Mem2Stream_Batch<ReadsPerWeightMem>(weight_mem, m2s_out, OFMDim * OFMDim * reps);
  WidthAdjustedInputStream <WeightMemW, PE*Tw::width, ReadsPerWeightMem>  weight_stream (m2s_out, OFMDim * OFMDim * reps);	

  Conv_Dw_Batch_StreamWeights
  <
  ConvKernelDim,
  IFMChannels,
  IFMDim,
  OFMChannels,
  OFMDim,
  Stride,
  Padding,
  SIMD,
  PE,
  Tw,
  TSrcI,
  TDstI
  >
  (in_stream,
  out_stream,
  weight_stream,
  activation,
  reps,
  r, flag, profiler);
	
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Padding_Batch, when Padding != 0
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD,
  unsigned int PE,	
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Dw_Padding_Batch(hls::stream<ap_uint<InStreamW>> &in,
                           hls::stream<ap_uint<OutStreamW>> &out,
                           TW const &weights,
                           TA const &activation,
                           unsigned const reps,
                           R const &r, int flag = 0, Profiler *profiler = nullptr)
{
  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE
	unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;

	WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_padded (in,  reps); 
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
	hls::stream<ap_uint<SIMD*TSrcI::width>> padded_conv("WidthAdjustedInputStream.SameResize_Batch.padded_conv");
  hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_dws.Vector_Vector_Activate_Batch.convInp");

	SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(wa_padded, padded_conv, reps);	

	ConvolutionInputGenerator_dws<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim+2, OFMDim, SIMD, Stride>
		(padded_conv, convInp, reps, ap_resource_dflt());	

	Vector_Vector_Activate_Batch<IFMChannels, ConvKernelDim*ConvKernelDim, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
		weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Padding_Batch_StreamWeights, when Padding != 0, "weight" is a stream
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD,
  unsigned int PE,	
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Dw_Padding_Batch_StreamWeights(hls::stream<ap_uint<InStreamW>> &in,
                                         hls::stream<ap_uint<OutStreamW>> &out,
                                         hls::stream<ap_uint<PE*Tw::width>> & weights,
                                         TA const &activation,
                                         unsigned const reps,
                                         R const &r, int flag = 0, Profiler *profiler = nullptr)
{

  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE
	unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;

	WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_padded (in,  reps); 
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
	hls::stream<ap_uint<SIMD*TSrcI::width>> padded_conv("WidthAdjustedInputStream.SameResize_Batch.padded_conv");
  hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_dws.Vector_Vector_Activate_Batch.convInp");

	SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(wa_padded, padded_conv, reps);	

	ConvolutionInputGenerator_dws<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim+2, OFMDim, SIMD, Stride>
		(padded_conv, convInp, reps, ap_resource_dflt());	

	Vector_Vector_Activate_Batch_StreamWeights<IFMChannels, ConvKernelDim*ConvKernelDim, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
		weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Padding_Batch_OffChipWeights, when Padding != 0, "weight" is in off-chip memory
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD,
  unsigned int PE,	
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (width must be int though!)
  int WeightMemW, int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Dw_Padding_Batch_OffChipWeights(hls::stream<ap_uint<InStreamW>> &in_stream,
                                          hls::stream<ap_uint<OutStreamW>> &out_stream,
                                          ap_uint<WeightMemW> *weight_mem,
                                          TA const &activation,
                                          unsigned const reps,
                                          R const &r, int flag = 0, Profiler *profiler = nullptr)
{
	
	CASSERT_DATAFLOW(WeightMemW % Tw::width == 0);
  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE

  unsigned const ReadsPerWeightMem = (OFMChannels * ConvKernelDim * ConvKernelDim * Tw::width + WeightMemW - 1) / WeightMemW;

  hls::stream<ap_uint<WeightMemW>> m2s_out("m2s_out");
  Mem2Stream_Batch<ReadsPerWeightMem>(weight_mem, m2s_out, OFMDim * OFMDim * reps);
  WidthAdjustedInputStream <WeightMemW, PE*Tw::width, ReadsPerWeightMem>  weight_stream (m2s_out, OFMDim * OFMDim * reps);	

	Conv_Dw_Padding_Batch_StreamWeights
  <
  ConvKernelDim,
  IFMChannels,
  IFMDim,
  OFMChannels,
  OFMDim,
  Stride,
  Padding,
  SIMD,
  PE,
  Tw,
  TSrcI,
  TDstI
  >
  (in_stream,
  out_stream,
  weight_stream,
  activation,
  reps,
  r, flag, profiler);
	
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Kernel_Stride_Padding_Batch, when ConvKernelDim % Stride != 0 and Padding != 0.
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD,
  unsigned int PE,		
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Dw_Kernel_Stride_Padding_Batch(hls::stream<ap_uint<InStreamW>> &in,
                                         hls::stream<ap_uint<OutStreamW>> &out,
                                         TW const &weights,
                                         TA const &activation,
                                         unsigned const reps,
                                         R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE
	unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;

	WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_padded (in,  reps);
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
	hls::stream<ap_uint<SIMD*TSrcI::width>> padded_conv("WidthAdjustedInputStream.SameResize_Batch.padded_conv");
  hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_dws.Vector_Vector_Activate_Batch.convInp");

	SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(wa_padded, padded_conv, reps);	

	ConvolutionInputGenerator_kernel_stride_dws<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim+1, OFMDim, SIMD, Stride>
	(padded_conv, convInp, reps, ap_resource_dflt());
  
	Vector_Vector_Activate_Batch<IFMChannels, ConvKernelDim*ConvKernelDim, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		 static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
		 weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Kernel_Stride_Padding_Batch_StreamWeights, when ConvKernelDim % Stride != 0 and Padding != 0. "weight" is a stream
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD,
  unsigned int PE,		
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Dw_Kernel_Stride_Padding_Batch_StreamWeights(hls::stream<ap_uint<InStreamW>> &in,
                                                       hls::stream<ap_uint<OutStreamW>> &out,
                                                       hls::stream<ap_uint<PE*Tw::width>> & weights,
                                                       TA const &activation,
                                                       unsigned const reps,
                                                       R const &r, int flag = 0, Profiler *profiler = nullptr)
{
  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE

	unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;

	WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_padded (in,  reps);
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
	hls::stream<ap_uint<SIMD*TSrcI::width>> padded_conv("WidthAdjustedInputStream.SameResize_Batch.padded_conv");
  hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_dws.Vector_Vector_Activate_Batch.convInp");

	SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(wa_padded, padded_conv, reps);	

	ConvolutionInputGenerator_kernel_stride_dws<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim+1, OFMDim, SIMD, Stride>
	(padded_conv, convInp, reps, ap_resource_dflt());
  
	Vector_Vector_Activate_Batch_StreamWeights<IFMChannels, ConvKernelDim*ConvKernelDim, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
		(static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
		 static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
		 weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Dw_Kernel_Stride_Padding_Batch_OffChipWeights, when ConvKernelDim % Stride != 0 and Padding != 0. "weight" is in off-chip memory
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD,
  unsigned int PE,		
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, int WeightMemW, typename TA, typename R
>
void Conv_Dw_Kernel_Stride_Padding_Batch_OffChipWeights(hls::stream<ap_uint<InStreamW>> &in_stream,
                                                       hls::stream<ap_uint<OutStreamW>> &out_stream,
                                                       ap_uint<WeightMemW> *weight_mem,
                                                       TA const &activation,
                                                       unsigned const reps,
                                                       R const &r, int flag = 0, Profiler *profiler = nullptr)
{
	CASSERT_DATAFLOW(WeightMemW % Tw::width == 0);
  CASSERT_DATAFLOW(IFMChannels == OFMChannels);
  CASSERT_DATAFLOW(SIMD == PE);

#pragma HLS INLINE

  unsigned const ReadsPerWeightMem = (OFMChannels * ConvKernelDim * ConvKernelDim * Tw::width + WeightMemW - 1) / WeightMemW;

  hls::stream<ap_uint<WeightMemW>> m2s_out("m2s_out");
  Mem2Stream_Batch<ReadsPerWeightMem>(weight_mem, m2s_out, OFMDim * OFMDim * reps);
  WidthAdjustedInputStream <WeightMemW, PE*Tw::width, ReadsPerWeightMem>  weight_stream (m2s_out, OFMDim * OFMDim * reps);	

	Conv_Dw_Kernel_Stride_Padding_Batch_StreamWeights
  <
  ConvKernelDim,
  IFMChannels,
  IFMDim,
  OFMChannels,
  OFMDim,
  Stride,
  Padding,
  SIMD,
  PE,
  Tw,
  TSrcI,
  TDstI
  >
  (in_stream,
  out_stream,
  weight_stream,
  activation,
  reps,
  r, flag, profiler);
}
/*
****************************************************************************************************************************************************************************************************************************************************************************************************************
* 
****************************************************************************************************************************************************************************************************************************************************************************************************************
*/
/*
********************************************************************************************************************************************************
* Conv_Kernel_Stride_Padding_Batch, when ConvKernelDim % Stride != 0 and Padding != 0
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD, 			
  unsigned int PE,				
  typename Tw,	
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Kernel_Stride_Padding_Batch(hls::stream<ap_uint<InStreamW>> &in,
                                      hls::stream<ap_uint<OutStreamW>> &out,
                                      TW const &weights,
                                      TA const &activation,
                                      unsigned const reps,
                                      R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE

  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;
  
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_padded (in,  reps); 
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);   
  hls::stream<ap_uint<SIMD*TSrcI::width>> padded_conv("WidthAdjustedInputStream.SameResize_Batch.padded_conv");
  hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_kernel_stride.Matrix_Vector_Activate_Batch.convInp");

  SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(wa_padded, padded_conv, reps);  
  
  ConvolutionInputGenerator_kernel_stride<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim+1, OFMDim, SIMD, Stride>
      (padded_conv, convInp, reps, ap_resource_dflt());
  
  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Kernel_Stride_Padding_Batch_StreamWeights, when ConvKernelDim % Stride != 0 and Padding != 0. "weights" is a stream
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD, 			
  unsigned int PE,				
  typename Tw,	
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Kernel_Stride_Padding_Batch_StreamWeights(hls::stream<ap_uint<InStreamW>> &in,
                                                    hls::stream<ap_uint<OutStreamW>> &out,
                                                    hls::stream<ap_uint<SIMD*PE*Tw::width>> & weights,
                                                    TA const &activation,
                                                    unsigned const reps,
                                                    R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE

  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;
  
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_padded (in,  reps); 
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);   
  hls::stream<ap_uint<SIMD*TSrcI::width>> padded_conv("WidthAdjustedInputStream.SameResize_Batch.padded_conv");
  hls::stream<ap_uint<SIMD*TSrcI::width>> convInp("ConvolutionInputGenerator_kernel_stride.Matrix_Vector_Activate_Batch.convInp");

  SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(wa_padded, padded_conv, reps);  
  
  ConvolutionInputGenerator_kernel_stride<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim+1, OFMDim, SIMD, Stride>
      (padded_conv, convInp, reps, ap_resource_dflt());
    
  Matrix_Vector_Activate_Batch_StreamWeights<MatrixW, MatrixH, SIMD, PE, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Kernel_Stride_Padding_Batch_OffChipWeights, when ConvKernelDim % Stride != 0 and Padding != 0. "weights" is in off-chip memory
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,		
  unsigned int Padding, // !!!: deducible	
  unsigned int SIMD, 			
  unsigned int PE,				
  typename Tw,	
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int WeightMemW, int InStreamW, int OutStreamW, typename TA, typename R
>
void Conv_Kernel_Stride_Padding_Batch_OffChipWeights(hls::stream<ap_uint<InStreamW>> &in_stream,
                                                     hls::stream<ap_uint<OutStreamW>> &out_stream,
                                                     ap_uint<WeightMemW> *weight_mem,
                                                     TA const &activation,
                                                     unsigned const reps,
                                                     R const &r, int flag = 0, Profiler *profiler = nullptr)
{
	
  CASSERT_DATAFLOW(WeightMemW % Tw::width == 0);

#pragma HLS INLINE

  unsigned const ReadsPerWeightMem = (IFMChannels * OFMChannels * ConvKernelDim * ConvKernelDim * Tw::width + WeightMemW - 1) / WeightMemW;

  hls::stream<ap_uint<WeightMemW>> m2s_out("m2s_out");
  Mem2Stream_Batch<ReadsPerWeightMem>(weight_mem, m2s_out, OFMDim * OFMDim * reps);
  WidthAdjustedInputStream <WeightMemW, SIMD*PE*Tw::width, ReadsPerWeightMem>  weight_stream (m2s_out, OFMDim * OFMDim * reps);	

	Conv_Kernel_Stride_Padding_Batch_StreamWeights
  <
  ConvKernelDim,
  IFMChannels,
  IFMDim,
  OFMChannels,
  OFMDim,
  Stride,
  Padding,
  SIMD,
  PE,
  Tw,
  TSrcI,
  TDstI
  >
  (in_stream,
  out_stream,
  weight_stream,
  activation,
  reps,
  r, flag, profiler);	
}
/*
********************************************************************************************************************************************************
* Conv_Kernel_Stride_Batch, when ConvKernelDim % Stride != 0 but Padding = 0
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,	
  unsigned int Padding, // !!!: always 0
  unsigned int SIMD,
  unsigned int PE,
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Kernel_Stride_Batch(hls::stream<ap_uint<InStreamW>> &in,
                              hls::stream<ap_uint<OutStreamW>> &out,
                              TW const &weights,
                              TA const &activation,
                              unsigned const reps,
                              R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;
  
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_in (in,  reps);
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");
  
  ConvolutionInputGenerator_kernel_stride<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim, SIMD, Stride>
      (wa_in, convInp, reps, ap_resource_dflt());
  
  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Batch, when ConvKernelDim % Stride = 0 and Padding = 0
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,		
  unsigned int Stride,	
  unsigned int Padding, // !!!: always = 0
  unsigned int SIMD,
  unsigned int PE,	
  typename Tw,		
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Batch(hls::stream<ap_uint<InStreamW>> & in,
                hls::stream<ap_uint<OutStreamW>> &out,
                TW const & weights,
                TA const & activation,
                unsigned const reps,
                R const &r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim*IFMDim*IFMChannels*TSrcI::width/InStreamW;
  unsigned const ReadsPerOutput = OFMDim*OFMDim*OFMChannels/PE;
  
  WidthAdjustedInputStream <InStreamW, SIMD*TSrcI::width, ReadsPerInput>  wa_in (in,  reps);
  WidthAdjustedOutputStream <PE*TDstI::width, OutStreamW, ReadsPerOutput>  mvOut (out,  reps);
  hls::stream<ap_uint<SIMD*TSrcI::width> > convInp("StreamingConvLayer_Batch.convInp");
  
  ConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim, SIMD,1>
      (wa_in, convInp, reps, ap_resource_dflt());
  
  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD*TSrcI::width>>&>(convInp),
     static_cast<hls::stream<ap_uint<PE*TDstI::width>>&>  (mvOut),
     weights, activation, reps* OFMDim * OFMDim, r, flag, profiler);
}
/*
********************************************************************************************************************************************************
* Conv_Padding_Batch, when ConvKernelDim % Stride = 0 but Padding != 0
********************************************************************************************************************************************************
*/
template
<
  unsigned int ConvKernelDim,		
  unsigned int IFMChannels,		
  unsigned int IFMDim,			
  unsigned int OFMChannels,		
  unsigned int OFMDim,			
  unsigned int Stride,
  unsigned int Padding, // !!!: deducible  
  unsigned int SIMD,
  unsigned int PE,
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,
  // safely deducible (stream width must be int though!)
  int InStreamW, int OutStreamW, typename TW, typename TA, typename R
>
void Conv_Padding_Batch(hls::stream<ap_uint<InStreamW>> & in,
                        hls::stream<ap_uint<OutStreamW>> & out,
                        TW const & weights,
                        TA const & activation,
                        unsigned const reps,
                        R const & r, int flag = 0, Profiler *profiler = nullptr)
{
#pragma HLS INLINE
  unsigned const MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
  unsigned const MatrixH = OFMChannels;
  unsigned const ReadsPerInput = IFMDim * IFMDim * IFMChannels * TSrcI::width / InStreamW;
  unsigned const ReadsPerOutput = OFMDim * OFMDim * OFMChannels / PE;
  unsigned const MvauTotalIters = reps * OFMDim * OFMDim;
  auto storg_r = ap_resource_dflt();

  hls::stream<ap_uint<SIMD * TSrcI::width>> waPadded("Conv_Padding_Batch.waPadded");
  hls::stream<ap_uint<SIMD * TSrcI::width>> paddedConv("Conv_Padding_Batch.SameResize_Batch.paddedConv");
  hls::stream<ap_uint<SIMD * TSrcI::width>> convInp("Conv_Padding_Batch.convInp");
  hls::stream<ap_uint<PE * TDstI::width>> mvOut("Conv_Padding_Batch.mvOut");

  StreamingDataWidthConverter_Batch<InStreamW, SIMD * TSrcI::width, ReadsPerInput>(in, waPadded, reps);
  SameResize_Batch<SIMD, IFMDim, ConvKernelDim, Stride, IFMChannels, TSrcI>(waPadded, paddedConv, reps);

  ConvolutionInputGenerator<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim + 2, OFMDim, SIMD, 1>(paddedConv, convInp, reps, storg_r);

  Matrix_Vector_Activate_Batch<MatrixW, MatrixH, SIMD, PE, 1, Tw, TSrcI, TDstI, TWeightI>(convInp, mvOut, weights, activation, MvauTotalIters, r, flag, profiler);
  StreamingDataWidthConverter_Batch<PE * TDstI::width, OutStreamW, ReadsPerOutput>(mvOut, out, reps);
}
/*
****************************************************************************************************************************************************************************************************************************************************************************************************************
* 
****************************************************************************************************************************************************************************************************************************************************************************************************************
*/
/*
********************************************************************************************************************************************************
* Depth_Wise: conv(g=1,k=1,s=1,p=0) + bn + relu + conv(g=!1,k=3,s=2,p=1) + bn + relu + conv(g=1,k=1,s=1,p=0) + bn
********************************************************************************************************************************************************
*/
template
<
  unsigned int conv_kernel_dim,
  unsigned int conv_ifmap_channels,
  unsigned int conv_ifmap_dim,
  unsigned int conv_ofmap_channels,
  unsigned int conv_ofmap_dim,
  unsigned int conv_stride,
  unsigned int conv_padding,
  unsigned int conv_simd,
  unsigned int conv_pe,
  int conv_input_stream_width,
  int conv_output_stream_width,
  typename conv_weight_dtype,
  typename conv_input_dtype,     
  typename conv_output_dtype,
  
  unsigned int conv_dw_kernel_dim,
  unsigned int conv_dw_ifmap_channels,
  unsigned int conv_dw_ifmap_dim,
  unsigned int conv_dw_ofmap_channels,
  unsigned int conv_dw_ofmap_dim,
  unsigned int conv_dw_stride,
  unsigned int conv_dw_padding,
  unsigned int conv_dw_simd,
  unsigned int conv_dw_pe,
  int conv_dw_input_stream_width,
  int conv_dw_output_stream_width,
  typename conv_dw_weight_dtype,
  typename conv_dw_input_dtype,     
  typename conv_dw_output_dtype,

  unsigned int conv_prj_kernel_dim,
  unsigned int conv_prj_ifmap_channels,
  unsigned int conv_prj_ifmap_dim,
  unsigned int conv_prj_ofmap_channels,
  unsigned int conv_prj_ofmap_dim,
  unsigned int conv_prj_stride,
  unsigned int conv_prj_padding,
  unsigned int conv_prj_simd,
  unsigned int conv_prj_pe,
  int conv_prj_input_stream_width,
  int conv_prj_output_stream_width,
  typename conv_prj_weight_dtype,
  typename conv_prj_input_dtype,     
  typename conv_prj_output_dtype,

  // safely deducible from the functions arguments
  //int conv_input_stream_width,
  //int conv_prj_output_stream_width,
  typename conv_weights_dtype,		
  typename conv_activation_dtype,
  typename conv_dw_weights_dtype,		
  typename conv_dw_activation_dtype,
  typename conv_prj_weights_dtype,
  typename conv_prj_activation_dtype
>
void Depth_Wise(hls::stream<ap_uint<conv_input_stream_width>> &in,
                hls::stream<ap_uint<conv_prj_output_stream_width>> &out,
                conv_weights_dtype const &conv_weights,
                conv_activation_dtype const &conv_activation,
                conv_dw_weights_dtype const &conv_dw_weights,
                conv_dw_activation_dtype const &conv_dw_activation,
                conv_prj_weights_dtype const &conv_prj_weights,
                conv_prj_activation_dtype const &conv_prj_activation,
                unsigned const reps, int flag = 0, Profiler *profiler_conv = nullptr, Profiler *profiler_dw = nullptr, Profiler *profiler_project = nullptr)
{
	
#pragma HLS DATAFLOW
 
  // Each of Conv_ layers has a converter inside converting from something* to simd*, this something* might be pe* of the previous layer
  hls::stream<ap_uint<conv_output_stream_width>> conv_conv_dw("Conv_Pw_Batch.Conv_Dw_Kernel_Stride_Padding_Batch.conv_conv_dw");
#pragma HLS STREAM variable = conv_conv_dw depth = 2
  hls::stream<ap_uint<conv_dw_output_stream_width>> conv_dw_conv_prj("Conv_Dw_Kernel_Stride_Padding_Batch.Conv_Pw_Batch.conv_dw_conv_prj");
#pragma HLS STREAM variable = conv_dw_conv_prj depth = 2

  Conv_Pw_Batch
  <
  conv_kernel_dim,
  conv_ifmap_channels,
  conv_ifmap_dim,
  conv_ofmap_channels,
  conv_ofmap_dim,
  conv_stride,
  conv_padding,
  conv_simd,
  conv_pe,
  conv_weight_dtype,
  Slice<conv_input_dtype>,
  Slice<conv_output_dtype>,
  Identity
  >
  (in,
  conv_conv_dw,
  conv_weights,
  conv_activation,
  reps,
  ap_resource_dflt(), flag, profiler_conv);

	Conv_Dw_Kernel_Stride_Padding_Batch
	<
	conv_dw_kernel_dim,
	conv_dw_ifmap_channels,
	conv_dw_ifmap_dim,
	conv_dw_ofmap_channels,
	conv_dw_ofmap_dim,
	conv_dw_stride,
  conv_dw_padding,
	conv_dw_simd,
	conv_dw_pe,
	conv_dw_weight_dtype,
	Slice<conv_dw_input_dtype>,
	Slice<conv_dw_output_dtype>
	>
	(conv_conv_dw,
	conv_dw_conv_prj,
	conv_dw_weights,
	conv_dw_activation,
	reps,
	ap_resource_dflt(), flag, profiler_dw);
	
	Conv_Pw_Batch
	<
	conv_prj_kernel_dim,
	conv_prj_ifmap_channels,
	conv_prj_ifmap_dim,
	conv_prj_ofmap_channels,
	conv_prj_ofmap_dim,
  conv_prj_stride,
  conv_prj_padding,
	conv_prj_simd,
	conv_prj_pe,
	conv_prj_weight_dtype,
	Slice<conv_prj_input_dtype>,
	Slice<conv_prj_output_dtype>
	>
	(conv_dw_conv_prj,
	out,
	conv_prj_weights,
	conv_prj_activation,
	reps,
	ap_resource_dflt(), flag, profiler_project);

}
/*
********************************************************************************************************************************************************
* Depth_Wise_StreamWeights: conv(g=1,k=1,s=1,p=0) + bn + relu + conv(g=!1,k=3,s=2,p=1) + bn + relu + conv(g=1,k=1,s=1,p=0) + bn, with streaming weights
********************************************************************************************************************************************************
*/
template
<
  unsigned int conv_kernel_dim,
  unsigned int conv_ifmap_channels,
  unsigned int conv_ifmap_dim,
  unsigned int conv_ofmap_channels,
  unsigned int conv_ofmap_dim,
  unsigned int conv_stride,
  unsigned int conv_padding,
  unsigned int conv_simd,
  unsigned int conv_pe,
  int conv_input_stream_width,
  int conv_output_stream_width,
  typename conv_weight_dtype,
  typename conv_input_dtype,     
  typename conv_output_dtype,

  unsigned int conv_dw_kernel_dim,
  unsigned int conv_dw_ifmap_channels,
  unsigned int conv_dw_ifmap_dim,
  unsigned int conv_dw_ofmap_channels,
  unsigned int conv_dw_ofmap_dim,
  unsigned int conv_dw_stride,
  unsigned int conv_dw_padding,
  unsigned int conv_dw_simd,
  unsigned int conv_dw_pe,
  int conv_dw_input_stream_width,
  int conv_dw_output_stream_width,
  typename conv_dw_weight_dtype,
  typename conv_dw_input_dtype,     
  typename conv_dw_output_dtype,

  unsigned int conv_prj_kernel_dim,
  unsigned int conv_prj_ifmap_channels,
  unsigned int conv_prj_ifmap_dim,
  unsigned int conv_prj_ofmap_channels,
  unsigned int conv_prj_ofmap_dim,
  unsigned int conv_prj_stride,
  unsigned int conv_prj_padding,
  unsigned int conv_prj_simd,
  unsigned int conv_prj_pe,
  int conv_prj_input_stream_width,
  int conv_prj_output_stream_width,
  typename conv_prj_weight_dtype,
  typename conv_prj_input_dtype,     
  typename conv_prj_output_dtype,

  // safely deducible from the functions arguments
  //int conv_input_stream_width,
  //int conv_prj_output_stream_width,	
  typename conv_activation_dtype,
  typename conv_dw_weights_dtype,		
  typename conv_dw_activation_dtype,
  typename conv_prj_activation_dtype
>
void Depth_Wise_StreamWeights(hls::stream<ap_uint<conv_input_stream_width>> &in,
                              hls::stream<ap_uint<conv_prj_output_stream_width>> &out,
                              hls::stream<ap_uint<conv_pe*conv_simd*conv_weight_dtype::width>> &conv_weights,
                              conv_activation_dtype const &conv_activation,
                              conv_dw_weights_dtype const &conv_dw_weights,
                              conv_dw_activation_dtype const &conv_dw_activation,
                              hls::stream<ap_uint<conv_prj_pe*conv_prj_simd*conv_prj_weight_dtype::width>> &conv_prj_weights,
                              conv_prj_activation_dtype const &conv_prj_activation,
                              unsigned const reps, int flag = 0, Profiler *profiler_conv = nullptr, Profiler *profiler_dw = nullptr, Profiler *profiler_project = nullptr)
{
	
#pragma HLS DATAFLOW
 
  // Each of Conv_ layer has a converter inside converting from something* to simd*, this something* might be pe* of the previous layer
  hls::stream<ap_uint<conv_output_stream_width>> conv_conv_dw("ConvLayer_Batch_Kernel1.ConvLayer_Batch_Kernel_Stride_DWS_Padding.conv_conv_dw");
#pragma HLS STREAM variable = conv_conv_dw depth = 2
  hls::stream<ap_uint<conv_dw_output_stream_width>> conv_dw_conv_prj("ConvLayer_Batch_Kernel_Stride_DWS_Padding.ConvLayer_Batch_Kernel1.conv_dw_conv_prj");
#pragma HLS STREAM variable = conv_dw_conv_prj depth = 2

	Conv_Pw_Batch_StreamWeights
	<
	conv_kernel_dim,
	conv_ifmap_channels,
	conv_ifmap_dim,
	conv_ofmap_channels,
	conv_ofmap_dim,
  conv_stride,
  conv_padding,
	conv_simd,
	conv_pe,
	conv_weight_dtype,
	Slice<conv_input_dtype>,
	Slice<conv_output_dtype>,
	Identity
	>
	(in,
	conv_conv_dw,
	conv_weights,
	conv_activation,
	reps,
	ap_resource_dflt(), flag, profiler_conv);
                          
	Conv_Dw_Kernel_Stride_Padding_Batch
	<
	conv_dw_kernel_dim,
	conv_dw_ifmap_channels,
	conv_dw_ifmap_dim,
	conv_dw_ofmap_channels,
	conv_dw_ofmap_dim,
	conv_dw_stride,
  conv_dw_padding,
	conv_dw_simd,
	conv_dw_pe,
	conv_dw_weight_dtype,
	Slice<conv_dw_input_dtype>,
	Slice<conv_dw_output_dtype>
	>
	(conv_conv_dw,
	conv_dw_conv_prj,
	conv_dw_weights,
	conv_dw_activation,
	reps,
	ap_resource_dflt(), flag, profiler_dw);
	
	Conv_Pw_Batch_StreamWeights
	<
	conv_prj_kernel_dim,
	conv_prj_ifmap_channels,
	conv_prj_ifmap_dim,
	conv_prj_ofmap_channels,
	conv_prj_ofmap_dim,
  conv_prj_stride,
  conv_prj_padding,
	conv_prj_simd,
	conv_prj_pe,
	conv_prj_weight_dtype,
	Slice<conv_prj_input_dtype>,
	Slice<conv_prj_output_dtype>
	>
	(conv_dw_conv_prj,
	out,
	conv_prj_weights,
	conv_prj_activation,
	reps,
	ap_resource_dflt(), flag, profiler_project);

}
/*
********************************************************************************************************************************************************
* Depth_Wise_Residual: conv(g=1,k=1,s=1,p=0) + bn + relu + conv(g=!1,k=3,s=1,p=1) + bn + relu + conv(g=1,k=1,s=1,p=0) + bn
********************************************************************************************************************************************************
*/
template
<
  unsigned int conv_kernel_dim,
  unsigned int conv_ifmap_channels,
  unsigned int conv_ifmap_dim,
  unsigned int conv_ofmap_channels,
  unsigned int conv_ofmap_dim,
  unsigned int conv_stride,
  unsigned int conv_padding,
  unsigned int conv_simd,
  unsigned int conv_pe,
  int conv_input_stream_width,
  int conv_output_stream_width,
  typename conv_weight_dtype,	
  typename conv_input_dtype,     
  typename conv_output_dtype,

  unsigned int conv_dw_kernel_dim,
  unsigned int conv_dw_ifmap_channels,
  unsigned int conv_dw_ifmap_dim,
  unsigned int conv_dw_ofmap_channels,
  unsigned int conv_dw_ofmap_dim,
  unsigned int conv_dw_stride,
  unsigned int conv_dw_padding,
  unsigned int conv_dw_simd,
  unsigned int conv_dw_pe,
  int conv_dw_input_stream_width,
  int conv_dw_output_stream_width,
  typename conv_dw_weight_dtype,
  typename conv_dw_input_dtype,     
  typename conv_dw_output_dtype,

  unsigned int conv_prj_kernel_dim,
  unsigned int conv_prj_ifmap_channels,
  unsigned int conv_prj_ifmap_dim,
  unsigned int conv_prj_ofmap_channels,
  unsigned int conv_prj_ofmap_dim,
  unsigned int conv_prj_stride,
  unsigned int conv_prj_padding,
  unsigned int conv_prj_simd,
  unsigned int conv_prj_pe,
  typename conv_prj_weight_dtype,
  typename conv_prj_input_dtype,     
  typename conv_prj_output_dtype,

  // safely deducible from the functions arguments
  //int conv_input_stream_width,
  int conv_prj_output_stream_width,
  typename conv_weights_dtype,		
  typename conv_activation_dtype,
  typename conv_dw_weights_dtype,		
  typename conv_dw_activation_dtype,
  typename conv_prj_weights_dtype,
  typename conv_prj_activation_dtype
>
void Depth_Wise_Residual(hls::stream<ap_uint<conv_input_stream_width>> &in,
                         hls::stream<ap_uint<conv_prj_output_stream_width>> &out,
                         conv_weights_dtype const &conv_weights,
                         conv_activation_dtype const &conv_activation,
                         conv_dw_weights_dtype const &conv_dw_weights,
                         conv_dw_activation_dtype const &conv_dw_activation,
                         conv_prj_weights_dtype const &conv_prj_weights,
                         conv_prj_activation_dtype const &conv_prj_activation,
                         unsigned const reps, int flag = 0, Profiler *profiler_conv = nullptr, Profiler *profiler_dw = nullptr, Profiler *profiler_project = nullptr)
{
	
#pragma HLS DATAFLOW	
  
  // Each of Conv_ layers has a converter inside converting from something* to simd*, this something* might be pe* of the previous layer
  hls::stream<ap_uint<conv_output_stream_width>> conv_conv_dw("ConvLayer_Batch_Kernel1.ConvLayer_Batch_Kernel_Stride_DWS_Padding.conv_conv_dw");
#pragma HLS STREAM variable = conv_conv_dw depth = 2
  hls::stream<ap_uint<conv_dw_output_stream_width>> conv_dw_conv_prj("ConvLayer_Batch_Kernel_Stride_DWS_Padding.ConvLayer_Batch_Kernel1.conv_dw_conv_prj");
#pragma HLS STREAM variable = conv_dw_conv_prj depth = 2

  Conv_Pw_Batch
  <
  conv_kernel_dim,
  conv_ifmap_channels,
  conv_ifmap_dim,
  conv_ofmap_channels,
  conv_ofmap_dim,
  conv_stride,
  conv_padding,
  conv_simd,
  conv_pe,
  conv_weight_dtype,
  Slice<conv_input_dtype>,
  Slice<conv_output_dtype>,
  Identity
  >
  (in,
  conv_conv_dw,
  conv_weights,
  conv_activation,
  reps,
  ap_resource_dflt(), flag, profiler_conv);

	Conv_Dw_Padding_Batch
	<
	conv_dw_kernel_dim,
	conv_dw_ifmap_channels,
	conv_dw_ifmap_dim,
	conv_dw_ofmap_channels,
	conv_dw_ofmap_dim,
	conv_dw_stride,
  conv_dw_padding,
	conv_dw_simd,
	conv_dw_pe,
	conv_dw_weight_dtype,
	Slice<conv_dw_input_dtype>,
	Slice<conv_dw_output_dtype>
	>
	(conv_conv_dw,
	conv_dw_conv_prj,
	conv_dw_weights,
	conv_dw_activation,
	reps,
	ap_resource_dflt(), flag, profiler_dw);

	Conv_Pw_Batch
	<
	conv_prj_kernel_dim,
	conv_prj_ifmap_channels,
	conv_prj_ifmap_dim,
	conv_prj_ofmap_channels,
	conv_prj_ofmap_dim,
  conv_prj_stride,
  conv_prj_padding,
	conv_prj_simd,
	conv_prj_pe,
	conv_prj_weight_dtype,
	Slice<conv_prj_input_dtype>,
	Slice<conv_prj_output_dtype>
	>
	(conv_dw_conv_prj,
	out,
	conv_prj_weights,
	conv_prj_activation,
	reps,
	ap_resource_dflt(), flag, profiler_project);

}
/*
********************************************************************************************************************************************************
* Depth_Wise_Residual_StreamWeights: conv(g=1,k=1,s=1,p=0) + bn + relu + conv(g=!1,k=3,s=1,p=1) + bn + relu + conv(g=1,k=1,s=1,p=0) + bn, with streaming weights
********************************************************************************************************************************************************
*/
template
<
  unsigned int conv_kernel_dim,
  unsigned int conv_ifmap_channels,
  unsigned int conv_ifmap_dim,
  unsigned int conv_ofmap_channels,
  unsigned int conv_ofmap_dim,
  unsigned int conv_stride,
  unsigned int conv_padding,
  unsigned int conv_simd,
  unsigned int conv_pe,
  int conv_input_stream_width,
  int conv_output_stream_width,
  typename conv_weight_dtype,	
  typename conv_input_dtype,     
  typename conv_output_dtype,

  unsigned int conv_dw_kernel_dim,
  unsigned int conv_dw_ifmap_channels,
  unsigned int conv_dw_ifmap_dim,
  unsigned int conv_dw_ofmap_channels,
  unsigned int conv_dw_ofmap_dim,
  unsigned int conv_dw_stride,
  unsigned int conv_dw_padding,
  unsigned int conv_dw_simd,
  unsigned int conv_dw_pe,
  int conv_dw_input_stream_width,
  int conv_dw_output_stream_width,
  typename conv_dw_weight_dtype,
  typename conv_dw_input_dtype,     
  typename conv_dw_output_dtype,

  unsigned int conv_prj_kernel_dim,
  unsigned int conv_prj_ifmap_channels,
  unsigned int conv_prj_ifmap_dim,
  unsigned int conv_prj_ofmap_channels,
  unsigned int conv_prj_ofmap_dim,
  unsigned int conv_prj_stride,
  unsigned int conv_prj_padding,
  unsigned int conv_prj_simd,
  unsigned int conv_prj_pe,
  typename conv_prj_weight_dtype,
  typename conv_prj_input_dtype,     
  typename conv_prj_output_dtype,

  // safely deducible from the functions arguments
  //int conv_input_stream_width,
  int conv_prj_output_stream_width,	
  typename conv_activation_dtype,
  typename conv_dw_weights_dtype,		
  typename conv_dw_activation_dtype,
  typename conv_prj_activation_dtype
>
void Depth_Wise_Residual_StreamWeights(hls::stream<ap_uint<conv_input_stream_width>> &in,
                                       hls::stream<ap_uint<conv_prj_output_stream_width>> &out,
                                       hls::stream<ap_uint<conv_pe*conv_simd*conv_weight_dtype::width>> &conv_weights,
                                       conv_activation_dtype const &conv_activation,
                                       conv_dw_weights_dtype const &conv_dw_weights,
                                       conv_dw_activation_dtype const &conv_dw_activation,
                                       hls::stream<ap_uint<conv_prj_pe*conv_prj_simd*conv_prj_weight_dtype::width>> &conv_prj_weights,
                                       conv_prj_activation_dtype const &conv_prj_activation,
                                       unsigned const reps, int flag = 0, Profiler *profiler_conv = nullptr, Profiler *profiler_dw = nullptr, Profiler *profiler_project = nullptr)
{
	
#pragma HLS DATAFLOW	
  
  // Each of Conv_ layer has a converter inside converting from something* to simd*, this something* might be pe* of the previous layer
  hls::stream<ap_uint<conv_output_stream_width>> conv_conv_dw("ConvLayer_Batch_Kernel1.ConvLayer_Batch_Kernel_Stride_DWS_Padding.conv_conv_dw");
#pragma HLS STREAM variable = conv_conv_dw depth = 2
  hls::stream<ap_uint<conv_dw_output_stream_width>> conv_dw_conv_prj("ConvLayer_Batch_Kernel_Stride_DWS_Padding.ConvLayer_Batch_Kernel1.conv_dw_conv_prj");
#pragma HLS STREAM variable = conv_dw_conv_prj depth = 2

  Conv_Pw_Batch_StreamWeights
  <
  conv_kernel_dim,
  conv_ifmap_channels,
  conv_ifmap_dim,
  conv_ofmap_channels,
  conv_ofmap_dim,
  conv_stride,
  conv_padding,
  conv_simd,
  conv_pe,
  conv_weight_dtype,
  Slice<conv_input_dtype>,
  Slice<conv_output_dtype>,
  Identity
  >
  (in,
  conv_conv_dw,
  conv_weights,
  conv_activation,
  reps,
  ap_resource_dflt(), flag, profiler_conv);

	Conv_Dw_Padding_Batch
	<
	conv_dw_kernel_dim,
	conv_dw_ifmap_channels,
	conv_dw_ifmap_dim,
	conv_dw_ofmap_channels,
	conv_dw_ofmap_dim,
	conv_dw_stride,
  conv_dw_padding,
	conv_dw_simd,
	conv_dw_pe,
	conv_dw_weight_dtype,
	Slice<conv_dw_input_dtype>,
	Slice<conv_dw_output_dtype>
	>
	(conv_conv_dw,
	conv_dw_conv_prj,
	conv_dw_weights,
	conv_dw_activation,
	reps,
	ap_resource_dflt(), flag, profiler_dw);

	Conv_Pw_Batch_StreamWeights
	<
	conv_prj_kernel_dim,
	conv_prj_ifmap_channels,
	conv_prj_ifmap_dim,
	conv_prj_ofmap_channels,
	conv_prj_ofmap_dim,
  conv_prj_stride,
  conv_prj_padding,
	conv_prj_simd,
	conv_prj_pe,
	conv_prj_weight_dtype,
	Slice<conv_prj_input_dtype>,
	Slice<conv_prj_output_dtype>
	>
	(conv_dw_conv_prj,
	out,
	conv_prj_weights,
	conv_prj_activation,
	reps,
	ap_resource_dflt(), flag, profiler_project);

}
/*
********************************************************************************************************************************************************
* Depth_Wise_Residual: conv(g=1,k=1,s=1,p=0) + bn + relu + conv(g=!1,k=3,s=1,p=1) + bn + relu + conv(g=1,k=1,s=1,p=0) + bn
* 
* DuplicateStreams_Batch -> FIFO -> WidthAdjustedInputStream -> 
*            ↓                                                 AddStreams_Batch ->
*        Depth_Wise ------------------------------------------>
********************************************************************************************************************************************************
*/
template
<
  unsigned int conv_depth,
  unsigned int conv_kernel_dim,
  unsigned int conv_ifmap_channels,
  unsigned int conv_ifmap_dim,
  unsigned int conv_ofmap_channels,
  unsigned int conv_ofmap_dim,
  unsigned int conv_stride,
  unsigned int conv_padding,
  unsigned int conv_simd,
  unsigned int conv_pe,
  int conv_input_stream_width,
  int conv_output_stream_width,
  typename conv_weight_dtype,
  typename conv_input_dtype,     
  typename conv_output_dtype,

  unsigned int conv_dw_kernel_dim,
  unsigned int conv_dw_ifmap_channels,
  unsigned int conv_dw_ifmap_dim,
  unsigned int conv_dw_ofmap_channels,
  unsigned int conv_dw_ofmap_dim,
  unsigned int conv_dw_stride,
  unsigned int conv_dw_padding,
  unsigned int conv_dw_simd,
  unsigned int conv_dw_pe,
  int conv_dw_input_stream_width,
  int conv_dw_output_stream_width,
  typename conv_dw_weight_dtype,
  typename conv_dw_input_dtype,     
  typename conv_dw_output_dtype,

  unsigned int conv_prj_kernel_dim,
  unsigned int conv_prj_ifmap_channels,
  unsigned int conv_prj_ifmap_dim,
  unsigned int conv_prj_ofmap_channels,
  unsigned int conv_prj_ofmap_dim,
  unsigned int conv_prj_stride,
  unsigned int conv_prj_padding,
  unsigned int conv_prj_simd,
  unsigned int conv_prj_pe,
  typename conv_prj_weight_dtype, 
  typename conv_prj_input_dtype,     
  typename conv_prj_output_dtype,

  typename add_output_dtype,

  // safely deducible from the functions arguments
  //int conv_input_stream_width,
  int add_ouput_stream_width,
  typename conv_weights_dtype,		
  typename conv_activation_dtype,
  typename conv_dw_weights_dtype,		
  typename conv_dw_activation_dtype,
  typename conv_prj_weights_dtype,
  typename conv_prj_activation_dtype
>
void Residual(hls::stream<ap_uint<conv_input_stream_width>> &in,
              hls::stream<ap_uint<add_ouput_stream_width>> &out,
              conv_weights_dtype const &conv_weights,
              conv_activation_dtype const &conv_activation,
              conv_dw_weights_dtype const &conv_dw_weights,
              conv_dw_activation_dtype const &conv_dw_activation,
              conv_prj_weights_dtype const &conv_prj_weights,
              conv_prj_activation_dtype const &conv_prj_activation,
              unsigned const reps, int flag = 0, Profiler *profiler_split = nullptr, Profiler *profiler_conv = nullptr, Profiler *profiler_dw = nullptr, Profiler *profiler_project = nullptr, Profiler *profiler_add = nullptr)
{
	
#pragma HLS DATAFLOW	
	
  unsigned const ReadsPerInput = conv_ifmap_dim * conv_ifmap_dim * conv_ifmap_channels  * conv_input_dtype::width / conv_input_stream_width ;
  
    hls::stream<ap_uint<conv_input_stream_width>> duplicate_depth_wise("DuplicateStreams_Batch.Depth_Wise.duplicate_depth_wise");
#pragma HLS STREAM variable = duplicate_depth_wise depth = 2

  hls::stream<ap_uint<conv_input_stream_width>> duplicate_wa("DuplicateStreams_Batch.WidthAdjustedInputStream.duplicate_wa");
#pragma HLS STREAM variable = duplicate_wa depth = conv_depth

  hls::stream<ap_uint<conv_prj_pe * conv_prj_output_dtype::width>> depth_wise_add("Depth_Wise.AddStreams.depth_wise_add");
#pragma HLS STREAM variable = depth_wise_add depth = 2

  WidthAdjustedInputStream
  <
  conv_input_stream_width,
  conv_prj_pe*conv_input_dtype::width,
  ReadsPerInput
  >  
  wa_add
  (duplicate_wa,
  reps);
  // wa_add(conv_prj_pe * conv_input_dtype::width) <- duplicate_wa(conv_input_stream_width)
 

  DuplicateStreams_Batch
  <
  ReadsPerInput,
  conv_input_dtype
  >
  (in,
  duplicate_depth_wise,
  duplicate_wa,
  reps, profiler_split);
 
  Depth_Wise_Residual
  <
  conv_kernel_dim,
  conv_ifmap_channels,
  conv_ifmap_dim,
  conv_ofmap_channels,
  conv_ofmap_dim,
  conv_stride,
  conv_padding,
  conv_simd,
  conv_pe,
  conv_input_stream_width,
  conv_output_stream_width,
  conv_weight_dtype,
  conv_input_dtype,     
  conv_output_dtype,
  conv_dw_kernel_dim,
  conv_dw_ifmap_channels,
  conv_dw_ifmap_dim,
  conv_dw_ofmap_channels,
  conv_dw_ofmap_dim,
  conv_dw_stride,
  conv_dw_padding,
  conv_dw_simd,
  conv_dw_pe,
  conv_dw_input_stream_width,
  conv_dw_output_stream_width,
  conv_dw_weight_dtype,
  conv_dw_input_dtype,     
  conv_dw_output_dtype,
  conv_prj_kernel_dim,
  conv_prj_ifmap_channels,
  conv_prj_ifmap_dim,
  conv_prj_ofmap_channels,
  conv_prj_ofmap_dim,
  conv_prj_stride,
  conv_prj_padding,
  conv_prj_simd,
  conv_prj_pe,
  conv_prj_weight_dtype,
  conv_prj_input_dtype,     
  conv_prj_output_dtype
  >
  (duplicate_depth_wise,
  depth_wise_add,
  conv_weights,
  conv_activation,
  conv_dw_weights,
  conv_dw_activation,
  conv_prj_weights,
  conv_prj_activation,
  reps, flag, profiler_conv, profiler_dw, profiler_project);

  unsigned const ReadsPerOutput = conv_prj_ofmap_dim * conv_prj_ofmap_dim * conv_prj_ofmap_channels / conv_prj_pe;
  WidthAdjustedOutputStream
  <
  conv_prj_pe*add_output_dtype::width,
  add_ouput_stream_width,
  ReadsPerOutput
  >
  add_wa
  (out,
  reps);
  // add_wa(conv_prj_pe * add_output_dtype::width) -> out(add_ouput_stream_width)

  AddStreams_Type_Batch
  <
  conv_prj_pe,
  conv_prj_output_dtype,
  conv_input_dtype,
  add_output_dtype,
  ReadsPerOutput
  >
  (depth_wise_add,
  wa_add,
  add_wa,
  reps, profiler_add);

}
/*
********************************************************************************************************************************************************
* Residual_StreamWeights: with streaming weights
********************************************************************************************************************************************************
*/
template
<
  unsigned int conv_depth,
  unsigned int conv_kernel_dim,
  unsigned int conv_ifmap_channels,
  unsigned int conv_ifmap_dim,
  unsigned int conv_ofmap_channels,
  unsigned int conv_ofmap_dim,
  unsigned int conv_stride,
  unsigned int conv_padding,
  unsigned int conv_simd,
  unsigned int conv_pe,
  int conv_input_stream_width,
  int conv_output_stream_width,
  typename conv_weight_dtype,
  typename conv_input_dtype,     
  typename conv_output_dtype,

  unsigned int conv_dw_kernel_dim,
  unsigned int conv_dw_ifmap_channels,
  unsigned int conv_dw_ifmap_dim,
  unsigned int conv_dw_ofmap_channels,
  unsigned int conv_dw_ofmap_dim,
  unsigned int conv_dw_stride,
  unsigned int conv_dw_padding,
  unsigned int conv_dw_simd,
  unsigned int conv_dw_pe,
  int conv_dw_input_stream_width,
  int conv_dw_output_stream_width,
  typename conv_dw_weight_dtype,
  typename conv_dw_input_dtype,     
  typename conv_dw_output_dtype,

  unsigned int conv_prj_kernel_dim,
  unsigned int conv_prj_ifmap_channels,
  unsigned int conv_prj_ifmap_dim,
  unsigned int conv_prj_ofmap_channels,
  unsigned int conv_prj_ofmap_dim,
  unsigned int conv_prj_stride,
  unsigned int conv_prj_padding,
  unsigned int conv_prj_simd,
  unsigned int conv_prj_pe,
  typename conv_prj_weight_dtype, 
  typename conv_prj_input_dtype,     
  typename conv_prj_output_dtype,

  typename add_streams_output_dtype,

  // safely deducible from the functions arguments
  //int conv_input_stream_width,
  int add_streams_ouput_stream_width,
  typename conv_activation_dtype,
  typename conv_dw_weights_dtype,		
  typename conv_dw_activation_dtype,
  typename conv_prj_activation_dtype
>
void Residual_StreamWeights(hls::stream<ap_uint<conv_input_stream_width>> &in,
                            hls::stream<ap_uint<add_streams_ouput_stream_width>> &out,
                            hls::stream<ap_uint<conv_pe*conv_simd*conv_weight_dtype::width>> &conv_weights,
                            conv_activation_dtype const &conv_activation,
                            conv_dw_weights_dtype const &conv_dw_weights,
                            conv_dw_activation_dtype const &conv_dw_activation,
                            hls::stream<ap_uint<conv_prj_pe*conv_prj_simd*conv_prj_weight_dtype::width>> &conv_prj_weights,
                            conv_prj_activation_dtype const &conv_prj_activation,
                            unsigned const reps, int flag = 0, Profiler *profiler_split = nullptr, Profiler *profiler_conv = nullptr, Profiler *profiler_dw = nullptr, Profiler *profiler_project = nullptr, Profiler *profiler_add = nullptr)
{
	
#pragma HLS DATAFLOW	
	
  unsigned const ReadsPerInput = conv_ifmap_dim * conv_ifmap_dim * conv_ifmap_channels  * conv_input_dtype::width / conv_input_stream_width ;
  unsigned const ReadsPerOutput = conv_prj_ofmap_dim * conv_prj_ofmap_dim * (conv_prj_ofmap_channels / conv_prj_pe);

    hls::stream<ap_uint<conv_input_stream_width>> duplicate_depth_wise("DuplicateStreams_Batch.Depth_Wise.duplicate_depth_wise");
#pragma HLS STREAM variable = duplicate_depth_wise depth = 2
  hls::stream<ap_uint<conv_input_stream_width>> duplicate_wa("DuplicateStreams_Batch.WidthAdjustedInputStream.duplicate_wa");
#pragma HLS STREAM variable = duplicate_wa depth = conv_depth
  hls::stream<ap_uint<conv_prj_pe * conv_prj_output_dtype::width>> depth_wise_add("Depth_Wise.AddStreams.depth_wise_add");
#pragma HLS STREAM variable = depth_wise_add depth = 2

  WidthAdjustedInputStream<conv_input_stream_width, conv_prj_pe * conv_input_dtype::width, ReadsPerInput>  wa_add (duplicate_wa,  reps);// wa_add(conv_prj_pe * conv_input_dtype::width) <- duplicate_wa(conv_input_stream_width)
  WidthAdjustedOutputStream <conv_prj_pe * add_streams_output_dtype::width, add_streams_ouput_stream_width, ReadsPerOutput>  add_wa (out,  reps);// add_wa(conv_prj_pe * add_streams_output_dtype::width) -> out(add_streams_ouput_stream_width)

  DuplicateStreams_Batch
  <
  ReadsPerInput,
  conv_input_dtype
  >
  (in,
  duplicate_depth_wise,
  duplicate_wa,
  reps, profiler_split);

	Depth_Wise_Residual_StreamWeights
	<
	conv_kernel_dim,
	conv_ifmap_channels,
	conv_ifmap_dim,
	conv_ofmap_channels,
	conv_ofmap_dim,
	conv_stride,
  conv_padding,
	conv_simd,
	conv_pe,
  conv_input_stream_width,
  conv_output_stream_width,
	conv_weight_dtype,
	conv_input_dtype,     
	conv_output_dtype,
	conv_dw_kernel_dim,
	conv_dw_ifmap_channels,
	conv_dw_ifmap_dim,
	conv_dw_ofmap_channels,
	conv_dw_ofmap_dim,
	conv_dw_stride,
  conv_dw_padding,
	conv_dw_simd,
	conv_dw_pe,
  conv_dw_input_stream_width,
  conv_dw_output_stream_width,
	conv_dw_weight_dtype,
	conv_dw_input_dtype,     
	conv_dw_output_dtype,
	conv_prj_kernel_dim,
	conv_prj_ifmap_channels,
	conv_prj_ifmap_dim,
	conv_prj_ofmap_channels,
	conv_prj_ofmap_dim,
	conv_prj_stride,
  conv_prj_padding,
	conv_prj_simd,
	conv_prj_pe,
	conv_prj_weight_dtype,
	conv_prj_input_dtype,     
	conv_prj_output_dtype
	>
	(duplicate_depth_wise,
	depth_wise_add,
	conv_weights,
	conv_activation,
	conv_dw_weights,
	conv_dw_activation,
	conv_prj_weights,
	conv_prj_activation,
	reps, flag, profiler_conv, profiler_dw, profiler_project);

  AddStreams_Type_Batch
  <
  conv_prj_pe,
  conv_prj_output_dtype,
  conv_input_dtype,
  add_streams_output_dtype,
  ReadsPerOutput
  >
  (depth_wise_add,
  wa_add,
  add_wa,
  reps, profiler_add);

}

#endif
