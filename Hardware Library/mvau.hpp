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
 *******************************************************************************/

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
 *  \file mvau.hpp
 *
 *  Library of templated HLS functions for DNN deployment.
 *  This file lists a templated funtion used to implement  
 *  Matrix-Vector-Activation Unit
 *
 *
 ************************************************************************************/


#ifndef MVAU_HPP
#define MVAU_HPP

#include "hls_stream.h"

#include "mac.hpp"
#include "interpret.hpp"

/**
 * \brief Matrix vector activate function
 *
 * The function performs the multiplication between a weigth matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 *
 * 
 * \tparam MatrixW    Width of the input matrix
 * \tparam MatrixH    Heigth of the input matrix
 * \tparam SIMD       Number of input columns computed in parallel
 * \tparam PE         Number of output rows computed in parallel
 * \tparam MMV        Number of output pixels computed in parallel
 * \tparam Tw         !!!: data type of a single weight
 * \tparam TSrcI      DataType of the input activation (as used in the MAC)
 * \tparam TDstI      DataType of the output activation (as generated by the activation)
 * \tparam TWeightI   DataType of the weights and how to access them in the array !!!: currently is not used as it slows down the synthesis
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters !!!: safely deducible
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters !!!: safely deducible
 * \tparam TW         DataType of the weights matrix - safely deducible from the paramaters !!!: safely deducible
 * \tparam TA         DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters !!!: safely deducible
 * \tparam R          Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the paramaters !!!: safely deducible
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param weights     Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation  Activation class
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r           Resource type for the hardware implementation of the MAC block
 * \param flag        !!!: Used during debugging to print values from the selected instance
 */
template
<
  unsigned MatrixW,
  unsigned MatrixH,
  unsigned SIMD,
  unsigned PE,
  unsigned MMV,  
  typename Tw,
  typename TSrcI = Identity,
  typename TDstI = Identity,  
  typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Matrix_Vector_Activate_Batch(hls::stream<TI> &in,
                                  hls::stream<TO> &out,
                                  TW const &weights,
                                  TA const &activation,
                                  int const reps,
                                  R const &r, int flag = 0, Profiler *profiler = nullptr)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;

  // input vector buffers
  TI  inputBuf[SF];
  // !!!: SIMD is a parallelism not SF
  //#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=0

  decltype(activation.init(0,0)) accu[MMV][PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++)
  {
  #pragma HLS PIPELINE II=1
    TI  inElem;
    if(nf == 0)
    {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else
    {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // Threshold Initialisation
    if(sf == 0)
    {
      for(unsigned  pe = 0; pe < PE; pe++)
      {
      #pragma HLS UNROLL
        for(unsigned mmv = 0; mmv < MMV; mmv++)
        {
        #pragma HLS UNROLL
          accu[mmv][pe] = activation.init(nf, pe);
        }
      }
    }
    // compute matrix-vector product for each processing element
    for(unsigned  pe = 0; pe < PE; pe++)
    {
    #pragma HLS UNROLL
      ap_uint<SIMD*Tw::width> wgt = weights[pe][tile];
      for (unsigned mmv = 0; mmv < MMV; mmv++)
      {
        auto const  act = TSrcI()(inElem, mmv);
        accu[mmv][pe] = mac_<SIMD,Tw>(accu[mmv][pe], wgt, act, r, mmv, 0, profiler);
      }
    }
    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF)
    {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++)
      {
      #pragma HLS UNROLL
        for (unsigned mmv = 0; mmv < MMV; mmv++)
        {
        #pragma HLS UNROLL
          outElem(pe,mmv,1) = activation.activate(nf, pe, accu[mmv][pe], flag, profiler);                
        }
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if(++nf == NF)
      {
        nf   = 0;
        tile = 0;
      }
    }
  }
}

template
<
  unsigned MatrixW,
  unsigned MatrixH,  
  unsigned SIMD,
  unsigned PE,  
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,  
  typename TI, typename TO, typename TA, typename R
>
void Matrix_Vector_Activate_Batch_StreamWeights(hls::stream<TI> &in,
                                                hls::stream<TO> &out,
                                                hls::stream<ap_uint<PE*SIMD*Tw::width>> &weight,
                                                TA  const &activation,
                                                int const  reps,
                                                R const &r, int flag = 0, Profiler *profiler = nullptr)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = MatrixH / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = MatrixW / SIMD;

  // input vector buffers
  TI  inputBuf[SF];
  // !!!: SIMD is a parallelism but not SF
  //#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
  
  // accumulators
  decltype(activation.init(0,0))  accu[1][PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

	//ap_uint<PE*SIMD*Tw::width> w_pe_simd;
	//ap_uint<SIMD*Tw::width>  w_simd;

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf
  
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++)
  {
  #pragma HLS PIPELINE II=1
    TI  inElem;

    if(nf == 0)
    {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else
    {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // read from the parameter stream
    ap_uint<PE*SIMD*Tw::width> w_pe_simd = weight.read();

    // Threshold Initialisation
    if(sf == 0)
    {
      for(unsigned pe = 0; pe < PE; pe++)
      {
      #pragma HLS UNROLL
        accu[0][pe] = activation.init(nf, pe);
      }
    }

    // compute matrix-vector product for each processing element
    for(unsigned  pe = 0; pe < PE; pe++)
    {
    #pragma HLS UNROLL
      auto const  act = TSrcI()(inElem, 0);
      ap_uint<SIMD*Tw::width> w_simd = w_pe_simd((pe+1)*SIMD*Tw::width-1,pe*SIMD*Tw::width);
      accu[0][pe] = mac_<SIMD,Tw>(accu[0][pe], w_simd, act, r, 0, flag, profiler);
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if(++sf == SF)
    {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator()<TO>();
      for (unsigned  pe = 0; pe < PE; pe++)
      {
      #pragma HLS UNROLL
        outElem(pe,0,1) = activation.activate(nf, pe, accu[0][pe], flag, profiler);
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if(++nf == NF)
      {
        nf   = 0;
        tile = 0;
      }
    }
  }
}

/**RPTU
 * \brief Matrix vector activate function for transposed convolution.
 *
 * The function performs the multiplication between a weight matrix and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 * This implementation is only valid for transposed convolution, hence it does not accumulate the results on a per kernel bases,
 * it accumulates the results on a channel wise bases, in other words it's like having a kernel of 1x1. Also, it assumes that the input,
 * is already expanded to the correct dimension so 1|2  should be inputted as 1|1|2|2 if the transpose kernel is 2x2 also it's currently only effective for Kernel of 2x2
 *                                                 3|4                        1|1|2|2
 *                                                                            3|3|4|4
 *                                                                            3|3|4|4
 *
 *
 * \tparam IFMChannels   Number of input feature map channels
 * \tparam OFMChannels   Number of output feature map channels
 * \tparam ConvKernelDim Dimension of the convolution kernel (assumed square) currently only 2 is supported
 * \tparam IFMDim 		 Width and Height of the Input Feature Map (assumed square)
 * \tparam SIMD          Number of input columns computed in parallel
 * \tparam PE            Number of output rows computed in parallel
 * \tparam TSrcI         DataType of the input activation (as used in the MAC)
 * \tparam TDstI         DataType of the output activation (as generated by the activation)
 * \tparam TWeightI      DataType of the weights and how to access them in the array
 * \tparam TI            DataType of the input stream - safely deducible from the parameters
 * \tparam TO            DataType of the output stream - safely deducible from the parameters
 * \tparam TW            DataType of the weights matrix - safely deducible from the parameters
 * \tparam TA            DataType of the activation class (e.g. thresholds) - safely deducible from the parameters
 * \tparam R             Datatype for the resource used for FPGA implementation of the MAC  - safely deducible from the parameters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param weights     Weights matrix (currently supports BinaryWeights or FixedPointWeights)
 * \param activation  Activation class
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 * \param r           Resource type for the hardware implementation of the MAC block
 */
template<
  unsigned IFMChannels, unsigned OFMChannels, unsigned ConvKernelDim, unsigned IFMDim, unsigned SIMD, unsigned PE, typename Tw,
  typename TSrcI = Identity, typename TDstI = Identity, typename TWeightI = Identity,
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Matrix_Vector_Activate_Batch_Transpose(hls::stream<TI>& in,
  hls::stream<TO>& out,
  TW  const& weights,
  TA  const& activation,
  int const  reps,
  R const& r, int flag = 0, Profiler *profiler = nullptr) {
  static_assert(OFMChannels == IFMChannels, "Code was only tested for equal input and output channels, comment and try :)");
  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = OFMChannels / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = IFMChannels / SIMD;
  // input vector buffers
  TI  inputBuf[SF];
// #pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=0

  decltype(activation.init(0,0)) accu[PE] = {0};
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf = 0;
  unsigned  sf = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelining the way we want
  unsigned const TOTAL_FOLD = NF * SF;
  unsigned const MAX_KER = ConvKernelDim * OFMChannels * IFMChannels; // Number of iterations required to calculate just one row of the kernel, used to control the tile index
  unsigned const ROW_ITERATIONS = OFMChannels * IFMChannels * IFMDim; // Number of iterations required to calculate whole row of the input, used to toggle Kernel_row
  unsigned TILE_BASE = 0; // For the first row of kernel it should be 0 otherwise it should be MAX_KER
  bool Kernel_row = 0; // 0:first row, 1:second row
  for (unsigned i = 0; i < reps * TOTAL_FOLD; i++) {
#pragma HLS pipeline style=stp II=1
    TI  inElem;
    if (nf == 0) {
      // read input from stream
      inElem = in.read();
      // store in appropriate buffer for reuse
      inputBuf[sf] = inElem;
    }
    else {
      // reuse buffered input
      inElem = inputBuf[sf];
    }

    // compute matrix-vector product for each processing element
    for (unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      ap_uint<SIMD*Tw::width> wgt = weights[pe][tile];
      auto const  act = TSrcI()(inElem, 0);
      accu[pe] = mac_<SIMD,Tw>(accu[pe], wgt, act, r, 0, flag, profiler);
    }

    // keep track of which folded synapse/neuron we are processing
    ++tile;
    if (++sf == SF) {
      // produce output and clear accumulators
      auto  outElem = TDstI().template operator() < TO > ();
      for (unsigned pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        outElem(pe,0,1) = activation.activate(nf, pe, accu[pe], flag, profiler); 
        accu[pe] = 0;
      }
      out.write(outElem);
      // next folded neuron or image
      sf = 0;
      if (++nf == NF) {
        nf = 0;
      }
    }
    if ((i + 1) % ROW_ITERATIONS == 0) { //reset every whole row
      Kernel_row = !Kernel_row;
      TILE_BASE = Kernel_row * MAX_KER;
    }
    if (tile % MAX_KER == 0) {
      tile = TILE_BASE;
    }
  }
}

#endif
