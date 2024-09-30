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
 *  \file vvau.hpp
 *
 *  Library of templated HLS functions for DNN deployment. 
 *  This file lists a templated funtion used to implement  
 *  Vector-Vector-Activation Unit (used for depthwise separable convolutions)
 *
 *
 ************************************************************************************/


#ifndef VVAU_HPP
#define VVAU_HPP

#include "hls_stream.h"

#include "mac.hpp"
#include "interpret.hpp"

/**
 * \brief Vector vector activate function
 *
 * The function performs the multiplication between a weigth vector and the input activation vector,
 * accumulating the results and then applying an activation function on the accumulated result.
 * It is used to implement depth-wise separable convolution
 * 
 * \tparam Channels   Number of channels
 * \tparam Kernel_2   Kernel * Kernel dimension (Kernel ^ 2 if square)
 * \tparam SIMD       Number of input columns computed in parallel !!!: does not make sense for depth-wise convolution
 * \tparam PE         Number of output rows computed in parallel
 * \tparam MMV        Number of output pixels computed in parallel
 * \tparam Tw         !!!: data type of a single weight
 * \tparam TSrcI      DataType of the input activation (as used in the MAC)
 * \tparam TDstI      DataType of the output activation (as generated by the activation)
 * \tparam TWeightI   DataType of the weights (as used in the MAC) !!!: currently is not used as it slows down the synthesis
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
  unsigned Channels,
  unsigned Kernel_2,
  unsigned SIMD, 
  unsigned PE,
  unsigned MMV,  
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,  
  typename TI, typename TO, typename TW, typename TA, typename R
>
void Vector_Vector_Activate_Batch(hls::stream<TI> &in,
                                  hls::stream<TO> &out,
                                  TW  const &weights,
                                  TA  const &activation,
                                  int const  reps,
                                  R const &r, int flag = 0, Profiler *profiler = nullptr)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const  NF = Channels / PE;

  // how many synapse groups each row is split into
  // alternatively: number of horizontal matrix chunks
  unsigned const  SF = (Channels * Kernel_2) / Channels;
  decltype(activation.init(0,0))  accu[MMV][PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

  unsigned  nf   = 0;
  unsigned  sf   = 0;
  unsigned  tile = 0; // invariant: tile = nf*SF + sf
  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  unsigned const TOTAL_FOLD = NF * SF ;//* Channels/SIMD;
  for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++)
  {
  #pragma HLS PIPELINE II=1
    TI  inElem;
    inElem = in.read();
    // Threshold Initialisation
    if(sf == 0)
    {
      for(unsigned  pe = 0; pe < PE; pe++)
      {
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
      ap_uint<Tw::width> wgt = weights[pe][tile];
      Tw weight = *reinterpret_cast<Tw*>(&wgt);
      for (unsigned mmv = 0; mmv < MMV; mmv++)
      {
        auto const  act = TSrcI()(inElem, mmv);

        #ifndef __SYNTHESIS__
          profiler->update_input(act(pe,mmv));
          profiler->update_weight(weight);
          profiler->update_acc(accu[mmv][pe]);
        #endif

		    auto mul_temp = mul(weight, act(pe,mmv), r);

         #ifndef __SYNTHESIS__
          profiler->update_acc(mul_temp);
        #endif

        accu[mmv][pe] += mul_temp;

        #ifndef __SYNTHESIS__
          profiler->update_acc(accu[mmv][pe]);
        #endif
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
  unsigned Channels,
  unsigned Kernel_2,
  unsigned SIMD, // !!!: should be equal to PE
  unsigned PE,  // !!!: should be equal to SIMD
  unsigned MMV,   
  typename Tw,  
  typename TSrcI = Identity,
  typename TDstI = Identity,
  typename TWeightI = Identity,  
  typename TI,
  typename TO,
  typename TA,
  typename R
>
void Vector_Vector_Activate_Batch_StreamWeights(hls::stream<TI> & in,
                                                hls::stream<TO> & out,
                                                hls::stream<ap_uint<PE*Tw::width>> & weights,
                                                TA const & activation,
                                                int const reps,
                                                R const & r, int flag = 0, Profiler *profiler = nullptr)
{

	CASSERT_DATAFLOW(SIMD == PE);
  
  // how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	unsigned const  NF = Channels / PE;

	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	unsigned const  SF = (Channels*Kernel_2) / Channels;
	decltype(activation.init(0,0))  accu[PE];
	#pragma HLS ARRAY_PARTITION variable=accu complete dim=0

	unsigned  nf   = 0;
	unsigned  sf   = 0;
	unsigned  tile = 0; // invariant: tile = nf*SF + sf
	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	unsigned const TOTAL_FOLD = NF * SF ;//* Channels/SIMD;
	for(unsigned  i = 0; i < reps * TOTAL_FOLD; i++)
	{
	#pragma HLS PIPELINE II=1
    TI  inElem;
    inElem = in.read();
    
    ap_uint<PE*Tw::width> w_pe= weights.read();
    
	// Threshold Initialisation
	if(sf == 0)
	{
		for(unsigned  pe = 0; pe < PE; pe++)
		{
		#pragma HLS UNROLL
			accu[pe] = activation.init(nf, pe);        
		}
	}

    // compute matrix-vector product for each processing element
    for(unsigned  pe = 0; pe < PE; pe++)
    {
	  #pragma HLS UNROLL
		  ap_uint<Tw::width> wgt = w_pe((pe+1)*Tw::width-1,pe*Tw::width);
		  Tw weight = *reinterpret_cast<Tw*>(&wgt);
		  auto const act = TSrcI()(inElem, 0);

      #ifndef __SYNTHESIS__
        profiler->update_input(act(pe,0));
        profiler->update_weight(weight);
        profiler->update_acc(accu[pe]);
      #endif

		  auto mul_temp = mul(weight, act(pe,0), r);

      #ifndef __SYNTHESIS__
        profiler->update_acc(mul_temp);
      #endif

      accu[pe] += mul_temp;

      #ifndef __SYNTHESIS__
        profiler->update_acc(accu[pe]);
      #endif

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
        outElem(pe,0,1) = activation.activate(nf, pe, accu[pe], flag, profiler);        
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
#endif
