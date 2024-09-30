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
 *  \file transposed_convlayer.hpp
 *
 *  Library of templated HLS functions for DNN deployment.
 *  This file defines the transposed convolution layer.
 *
 *  The following code has been developed for a work package "Hardware Architectures
 *  for Low Power ML" contributing to project SustainML (Application Aware, Life-Cycle
 *  Oriented Model-Hardware Co-Design Framework for Sustainable, Energy Efficient ML
 *  Systems). The project has received funding from European Union’s Horizon Europe
 *  research and innovation programme (HORIZON-CL4-2021-HUMAN-01)
 *  under grant agreement No 101070408.
 * 
 ************************************************************************************/

#ifndef TRANSPOSED_CONV_H
#define TRANSPOSED_CONV_H

#include <ap_int.h>
#include <hls_stream.h>
#include "hls_math.h"
#include "streamtools.h"
#include "mvau.hpp"

/**
 * \brief Perform a transposed convolution operation on the input stream.
 *
 * This function performs a transposed convolution (also known as deconvolution) on the input stream
 * and outputs the result to the output stream. It utilizes multiple processing elements (PEs) and
 * single instruction, multiple data (SIMD) lanes for parallel processing.
 *
 * \tparam ConvKernelDim The dimension of the convolution kernel.
 * \tparam IFMChannels The number of input feature map channels.
 * \tparam IFMDim The dimension of the input feature map.
 * \tparam OFMChannels The number of output feature map channels.
 * \tparam SIMD The number of SIMD lanes.
 * \tparam PE The number of processing elements (PEs).
 * \tparam Tw The type of the weights.
 * \tparam TSrcI The type for input activations interpretation (default is Identity).
 * \tparam TDstI The type for output activations interpretation (default is Identity).
 * \tparam TWeightI The type for weights interpretation (default is Identity).
 * \tparam InStreamW The width of the input stream.
 * \tparam OutStreamW The width of the output stream.
 * \tparam TW The type of the weights.
 * \tparam TA The type of the activation.
 * \tparam R The type of the additional parameter.
 *
 * \param in The input stream of type `hls::stream<ap_uint<InStreamW>>`.
 * \param out The output stream of type `hls::stream<ap_uint<OutStreamW>>`.
 * \param weights The weights used for the transposed convolution.
 * \param activation The activation function to be applied.
 * \param reps The number of repetitions for the operation.
 * \param r An additional parameter of type `R`.
 * \param flag An optional flag parameter (default is 0).
 * \param profiler An optional profiler pointer (default is nullptr).
 */
template<
  unsigned int ConvKernelDim,
  unsigned int IFMChannels,
  unsigned int IFMDim,
  unsigned int OFMChannels,

  unsigned int SIMD,				// number of SIMD lanes
  unsigned int PE,				// number of PEs
  typename Tw,
  typename TSrcI = Identity,      // redefine I/O interpretation as needed for input activations
  typename TDstI = Identity,		// redefine I/O interpretation as needed for output activations
  typename TWeightI = Identity,	// redefine I/O interpretation as needed for weights

  int InStreamW, int OutStreamW,  // safely deducible (stream width must be int though!)
  typename TW, typename TA, typename R

>
void TransposeConv(hls::stream<ap_uint<InStreamW>>& in,
  hls::stream<ap_uint<OutStreamW>>& out,
  TW const& weights,
  TA const& activation,
  unsigned const   reps,
  R const& r, int flag = 0, Profiler *profiler = nullptr) {
  static_assert(IFMChannels == OFMChannels, "IFMChannels has to be equal to OFMChannels.");
#pragma HLS INLINE

  constexpr unsigned int OFMDim = ConvKernelDim * IFMDim;
  unsigned const InpPerImage = (IFMDim * IFMDim * IFMChannels * TSrcI::width) / InStreamW;
  unsigned const OutPerImage = OFMDim * OFMDim * (OFMChannels / PE);
  hls::stream<ap_uint<SIMD* TSrcI::width> > wa_in("TransposeConv.wa_in");
  hls::stream<ap_uint<SIMD* TSrcI::width> > calc_in("TransposeConv.calc_in");
  hls::stream<ap_uint<PE* TDstI::width> > mvOut("TransposeConv.mvOut");
  StreamingDataWidthConverter_Batch<InStreamW, SIMD* TSrcI::width, InpPerImage>(in, wa_in, reps);
  DeconvolutionInputGenerator_MMV<ConvKernelDim, IFMChannels, TSrcI::width, IFMDim, OFMDim, SIMD>(wa_in, calc_in);
  Matrix_Vector_Activate_Batch_Transpose<IFMChannels, OFMChannels, ConvKernelDim, OFMDim, SIMD, PE, Tw, TSrcI, TDstI, TWeightI>
    (static_cast<hls::stream<ap_uint<SIMD* TSrcI::width>>&>(calc_in),
      static_cast<hls::stream<ap_uint<PE* TDstI::width>>&>  (mvOut),
      weights, activation, reps * OFMDim * OFMDim, r, flag, profiler);
  StreamingDataWidthConverter_Batch<PE* TDstI::width, OutStreamW, OutPerImage>(mvOut, out, reps);
}
#endif