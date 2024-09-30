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
 *  \file activations.hpp
 *
 *  Library of templated HLS classes for DNN deployment. 
 *  This file lists a set of classes used to implement  
 *  various activation functions in neural networks.
 *
 ************************************************************************************/


#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "interpret.hpp"

namespace comp{

  template<typename input_type = void>
    struct greater;

  template<typename input_type = void>
    struct less;

  template<typename input_type = void>
    struct greater_equal;

  template<typename input_type = void>
    struct less_equal;	

  template<typename input_type>
    struct greater : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a > b; }
    };

  template<typename input_type>
    struct less : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a < b; }
    };

  template<typename input_type>
    struct greater_equal : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a >= b; }
    };

  template<typename input_type>
    struct less_equal : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a <= b; }
    };
	
}

/**
 * General contract for activation functions.
 *
 * This class itself has no formal significance for the implementation
 * of the MVAU. Implementations of activation functions are encouraged
 * to implement it nonetheless to guarantee appropriate function
 * signatures.
 */
template<typename TA, typename TO>
class Activation {
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

  /**
   * Compute the activation of the passed accumulator value accu in row idx.
   */
  TO activate(unsigned const  nf, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const;
};

/**
 * A no-op activation that simply outputs the computed accumulator
 * output as the final result.
 */
template<typename TA, typename TO>
class PassThroughActivation : public Activation<TA, TO> {
public:
  ap_uint<TO::width> activate(unsigned const  nf, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
#pragma HLS inline
    TO result = (TO)accu;
    ap_uint<TO::width> result_int = *reinterpret_cast<ap_uint<TO::width>*>(&result);
    return result_int;
  }
};

/**
 * Use a simple global threshold comparison as activation function.
 *
 * The constant threshold is initialized at construction.
 * The default comparison returns true if the threshold value is
 * smaller than the passed accumulator value.
 */
template<typename TA, typename Compare = comp::less<TA>>
class ThresholdActivation : public Activation<TA, bool> {
  TA const  m_threshold;
public:
  ThresholdActivation(TA const &threshold) : m_threshold(threshold) {
#pragma HLS inline
  }

public:
  bool activate(unsigned const  nf, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
#pragma HLS inline
    return  Compare()(m_threshold, accu);
  }
};

/*!
 * Use a simple per-row threshold comparison as activation function.
 *
 * The thresholds are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 *
 * The default comparison returns true if the threshold value defined for
 * the indexed row is smaller than the passed accumulator value.
 */
template<unsigned NF, unsigned PE, unsigned NumTH, 
	 typename TA, typename TR, int ActVal = 0, typename Compare = comp::less<TA>>
class ThresholdsActivation {
public:
  TA m_thresholds[PE][NF][NumTH];
  
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
#pragma HLS inline
    TR result=ActVal;
	for(unsigned int i=0; i< NumTH; i++){
#pragma HLS unroll
      result+=Compare()(m_thresholds[pe][nf][i], accu);
    }
    return result;
  }
};


/*!
 * \brief Use a simple activation function with per-row parameters.
 *
 * The parameters are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 * 
 * \tparam NF    First dimension of the parameter matrix
 * \tparam PE    Second dimension of the parameter matrix
 * \tparam TI    DataType of input layer values
 * \tparam TP    DataType of parameters
 * \tparam TR    DataType of return values
 * \tparam Fxn   Function to be applied on the channel input value
 */

template<unsigned NF, unsigned PE,
   typename TI, typename TP, typename TR, typename Fxn = std::multiplies<TR>>
class ChannelWiseOperation {
public:
  TP parameters[PE][NF];
public:
  TI init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TI(0);
  }
public:
  TR activate(unsigned const  nf, unsigned const  pe,  TI const &in) const {
#pragma HLS inline
    TR result = Fxn()(parameters[pe][nf], in);
    return result;
  }
};

/*!
 * \brief Thresholding function for multiple images
 *
 * The function performs thresholds comparison with input activation vector, 
 * and generating output based on the comparison results
 *
 * \tparam ImgDimH        Heigth of the Input Feature Map
 * \tparam ImgDimW        Width of the Input Feature Map
 * \tparam NumChannels    Heigth of the input matrix
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param activation      Activation class
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned ImgDimH, unsigned ImgDimW, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    typename TI, typename TO, typename TA>
void Thresholding_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        TA const &activation,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const NF = NumChannels / PE;

  unsigned nf = 0;
  unsigned tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  for (unsigned i = 0; i < reps * ImgDimH * ImgDimW * NF; i++)
  {
    #pragma HLS PIPELINE II=1

    TI inElem;
    inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();
    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = activation.activate(nf, pe, act(pe,0));
    }
    out.write(outElem);
    if (++nf == NF)
    {
      nf = 0;
    }
  }
}

/*!
 * \brief Thresholding function for multiple images, with streaming thresholds
 *
 * The function performs thresholds comparison with input activation vector, 
 * and generating output based on the comparison results
 *
 * \tparam ImgDimH        Heigth of the Input Feature Map
 * \tparam ImgDimW        Width of the Input Feature Map
 * \tparam NumChannels    Heigth of the input matrix
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam ActVal         Initial value of activation at start of thresholding procedure
 * \tparam TT             DataType of the thresholds stream
 * \tparam NumSteps       Number of thresholds per activation
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param weight          Weight stream
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned ImgDimH, unsigned ImgDimW, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    int ActVal=0, typename TT, unsigned int NumSteps,
    typename TI, typename TO>
void Thresholding_Stream_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        hls::stream<ap_uint<PE*NumSteps*TT::width>> &weight,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const NF = NumChannels / PE;

  ThresholdsActivation<1, PE, NumSteps, TT, TO, ActVal, comp::less_equal<TT>> internal_thr;
  #pragma HLS ARRAY_PARTITION variable=internal_thr.m_thresholds complete dim=0

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  for (unsigned i = 0; i < reps * ImgDimH * ImgDimW * NF; i++)
  {
    #pragma HLS PIPELINE II=1

    ap_uint<PE*NumSteps*TT::width> packed_thr;
    packed_thr = weight.read();
    // slicer to get 1 PE's worth of thresholds
    auto const pe_slicer = Slice<ap_uint<NumSteps*TT::width>>()(packed_thr);

    TI inElem;
    inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();

    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      // slicer to get individual thresholds
      auto const thr_slicer = Slice<TT>()(pe_slicer(pe, 0));
      for (unsigned nt = 0; nt < NumSteps; nt++)
      {
      #pragma HLS UNROLL
        internal_thr.m_thresholds[pe][0][nt] = thr_slicer(nt, 0);
      }

      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = internal_thr.activate(0, pe, act(pe,0));
    }
    out.write(outElem);
  }
}

/*
********************************************************************************************************************************************************
* ReLU
********************************************************************************************************************************************************
*/

/**
 * Use a per-output channel/filter ReLU as activation function.
 *
 */
template<unsigned PE, unsigned TILES, typename TA, typename TO>
class ReLuActivation {
public:
  static const unsigned int MAX_OUT_VAL = (1 << (TO::iwidth - 1)) - 1;

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
   }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline

    #ifndef __SYNTHESIS__
      profiler->update_acc(accu);
    #endif

    TO res = (TO)0;
    if (accu < (TA)0)
        res = res;
    else
      if(accu > MAX_OUT_VAL){
        res = (TO)MAX_OUT_VAL;
      }else{
        res = (TO)accu;
      }

    #ifndef __SYNTHESIS__
      profiler->update_output(res);
    #endif

    return *reinterpret_cast<ap_uint<TO::width>*>(&res);
  }
};

template<unsigned PE, unsigned TILES, typename TA, typename TO>
class ReLu6Activation {
public:
  static const unsigned int MAX_OUT_VAL = 6;

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
   }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline

    #ifndef __SYNTHESIS__
      profiler->update_acc(accu);
    #endif

    TO res = (TO)0;
    if (accu < (TA)0)
        res = res;
    else
      if(accu > MAX_OUT_VAL){
        res = (TO)MAX_OUT_VAL;
      }else{
        res = (TO)accu;
      }

    #ifndef __SYNTHESIS__
      profiler->update_output(res);
    #endif

    return *reinterpret_cast<ap_uint<TO::width>*>(&res);
  }
};

/*
********************************************************************************************************************************************************
* Bias
********************************************************************************************************************************************************
*/

/**
 * Use a per-output bias as activation function.
 *
 *  \tparam TA            DataType of the conv or linear's accumulator
 *  \tparam TB            DataType of the bias (shift)
 *  \tparam TA            DataType of the activation function's accumulator
 *  \tparam TO            DataType of the activation function's output
 */
template<unsigned PE, unsigned TILES, typename TB, typename TA, typename TO>
class BiasActivation {
public:
    TB m_bias[PE][TILES];

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
  }
  
  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TA shifted = accu + (TA)m_bias[pe][tile];
    TO result = (TO)shifted;

    #ifndef __SYNTHESIS__
      profiler->update_bias(m_bias[pe][tile]);
      profiler->update_acc(m_bias[pe][tile]);
      profiler->update_acc(accu);
      profiler->update_acc(shifted);
      profiler->update_output(shifted);
    #endif

    if (flag == 1)
      printf("%.16f\n",(float)result);

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output bias as activation function.
 *
 *  \tparam TA            DataType of the conv or linear's accumulator
 *  \tparam TB            DataType of the bias (shift)
 *  \tparam TA            DataType of the activation function's accumulator
 *  \tparam TO            DataType of the activation function's output
 */
template<unsigned PE, unsigned TILES, typename TB, typename TA, typename TO>
class BiasSigmoidActivation {
public:
    TB m_bias[PE][TILES];

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
  }
  
  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TA shifted = accu + (TA)m_bias[pe][tile];
    TO result = (TO)shifted;

    #ifndef __SYNTHESIS__
      profiler->update_bias(m_bias[pe][tile]);
      profiler->update_acc(m_bias[pe][tile]);
      profiler->update_acc(accu);
      profiler->update_acc(shifted);
      profiler->update_output(shifted);
    #endif

    /*if (flag == 1)
      printf("%.16f\n",(float)acc);*/

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Bias + HardSwich Activation function.
 * HardSwich is a simplified version of SilU.
 *  \tparam PE            Number of output rows computed in parallel
 *  \tparam TILES         Number of input tiles
 *  \tparam TB            DataType of the bias (shift)
 *  \tparam TA            DataType of the activation function's accumulator
 *  \tparam TO            DataType of the activation function's output
 */
template<unsigned PE, unsigned TILES, typename TB, typename TA, typename TO>
class BiasHardSwichActivation {
public:
  TB m_bias[PE][TILES];

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
   }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TA sum = accu + (TA)m_bias[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_bias(m_bias[pe][tile]);
      profiler->update_acc(m_bias[pe][tile]);
      profiler->update_acc(sum);
    #endif

    TO result = (TO)0;
    if (sum <= (TA)-3)
        result = result;
    else if (sum >= (TA) 3)
        result = sum;
    else{
        result = (TO)(sum*(sum+3)/6);
    }

    #ifndef __SYNTHESIS__
      profiler->update_output(result);
    #endif

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output scale as activation function.
 *
 *  \tparam TCA           DataType of the conv or linear's accumulator
 *  \tparam TS            DataType of the scale
 *  \tparam TAA           DataType of the activation function's accumulator
 *  \tparam TO            DataType of the activation function's output
 */
template<unsigned PE, unsigned TILES, typename TCA, typename TS, typename TAA, typename TO>
class MulActivation {
public:
    TS m_scale;

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }
  
  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA scaled = (TAA)accu * (TAA)m_scale;
    TO result = (TO)scaled;

    #ifndef __SYNTHESIS__
      profiler->update_scale(m_scale);
      profiler->update_act_acc(accu);
      profiler->update_act_acc(m_scale);
      profiler->update_act_acc(scaled);
      profiler->update_output(result);
    #endif

    /*if (flag == 1)
      printf("%.16f\n",(float)acc);*/

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-row bias and scale as activation function.
 * 
 *  \tparam TCA            DataType of the conv or linear's accumulator
 *  \tparam TB             DataType of the bias (shift)
 *  \tparam TS             DataType of the scale
 *  \tparam TAA            DataType of the activation function's accumulator
 *  \tparam TO             DataType of the activation function's output
 *
 */
template<unsigned PE, unsigned TILES, typename TB, typename TCA, typename TS, typename TAA, typename TO>
class BiasMulActivation {
public:
  TB m_bias[PE][TILES];
  TS m_scale;

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &acc, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA shifted = (TAA)acc + (TAA)m_bias[pe][tile];
    TAA scaled = shifted * (TAA)m_scale;
    TO result = (TO)scaled;

    #ifndef __SYNTHESIS__
      profiler->update_bias(m_bias[pe][tile]);
      profiler->update_scale(m_scale);
      profiler->update_act_acc(m_bias[pe][tile]); 
      profiler->update_act_acc(m_scale);     
      profiler->update_act_acc(acc);            
      profiler->update_act_acc(scaled);      
      profiler->update_act_acc(shifted);
      profiler->update_output(scaled);
    #endif

    /*if (flag == 1)
      printf("%.16f\n",(float)acc);*/


    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output bias + ReLu as activation function.
 *
 *  \tparam TA             DataType of the conv or linear's accumulator
 *  \tparam TB             DataType of the bias (shift)
 *  \tparam TA             DataType of the activation function's accumulator
 *  \tparam TO             DataType of the activation function's output
 */
template<unsigned PE, unsigned TILES, typename TB, typename TA, typename TO>
class BiasReLuActivation {
public:
  TB m_bias[PE][TILES];
  static const unsigned int MAX_OUT_VAL = (1 << (TO::iwidth - 1)) - 1;

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
   }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TA shifted = accu + (TA)m_bias[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_bias(m_bias[pe][tile]);
      profiler->update_acc(m_bias[pe][tile]);
      profiler->update_acc(shifted);
    #endif

    TO result = (TO)0;
    if (shifted < (TA)0)
        result = result;
    else
      if(shifted > MAX_OUT_VAL){
        result = (TO)MAX_OUT_VAL;
      }else{
        result = (TO)shifted;
      }

    #ifndef __SYNTHESIS__
      profiler->update_output(shifted);
    #endif

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};
/**
 * Use a per-output bias + ReLu6 as activation function.
 *
 *  \tparam TA             DataType of the conv or linear's accumulator
 *  \tparam TB             DataType of the bias (shift)
 *  \tparam TA             DataType of the activation function's accumulator
 *  \tparam TO             DataType of the activation function's output
 */
template<unsigned PE, unsigned TILES, typename TB, typename TA, typename TO>
class BiasReLu6Activation {
public:
  TB m_bias[PE][TILES];
  static const unsigned int MAX_OUT_VAL = 6;

public:
  TA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TA(0);
   }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TA const &accu, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TA shifted = accu + (TA)m_bias[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_bias(m_bias[pe][tile]);
      profiler->update_acc(m_bias[pe][tile]);
      profiler->update_acc(shifted);
    #endif

    TO result = (TO)0;
    if (shifted < (TA)0)
        result = result;
    else
      if(shifted > MAX_OUT_VAL){
        result = (TO)MAX_OUT_VAL;
      }else{
        result = (TO)shifted;
      }

    #ifndef __SYNTHESIS__
      profiler->update_output(shifted);
    #endif

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/*
********************************************************************************************************************************************************
* Bias + Batchnorm; Bias + Batchnorm + ReLu
********************************************************************************************************************************************************
*/
/**
 * You do not need Bias, if you apply Batchnorm
*/
/*
********************************************************************************************************************************************************
* Batchnorm
********************************************************************************************************************************************************
*/

/**
 * Use a per-row batchnorm as activation function.
 * 
 *  \tparam TCA            DataType of the conv's accumulator
 *  \tparam TW             DataType of the scale (weight)
 *  \tparam TB             DataType of the shift (bias)
 *  \tparam TAA            DataType of the batchnorm's accumulator
 *  \tparam TO             DataType of the batchnorm's output
 *
 */
template<unsigned PE, unsigned TILES, typename TCA, typename TW, typename TB, typename TAA, typename TO>
class BatchNormActivation {
public:
  TW m_scale[PE][TILES];
  TB m_shift[PE][TILES];

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }

  ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &acc, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA scaled = (TAA)acc * (TAA)m_scale[pe][tile];
    TAA shifted = scaled - (TAA)m_shift[pe][tile];
    TO result = (TO)shifted;

    #ifndef __SYNTHESIS__
      profiler->update_scale(m_scale[pe][tile]);  
      profiler->update_shift(m_shift[pe][tile]);    
      profiler->update_act_acc(m_scale[pe][tile]);
      profiler->update_act_acc(m_shift[pe][tile]);
      profiler->update_act_acc(acc);
      profiler->update_act_acc(scaled);     
      profiler->update_act_acc(shifted);
      profiler->update_output(shifted);
    #endif

    /*if (flag == 1)
      printf("%.16f\n",(float)acc);*/

    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output channel/filter batchnorm + ReLu as activation function.
 * 
 *  \tparam TCA            DataType of the conv's accumulator
 *  \tparam TW             DataType of the scale (weight)
 *  \tparam TB             DataType of the shift (bias)
 *  \tparam TAA            DataType of the batchnorm's accumulator
 *  \tparam TO             DataType of the activation's output
 *
 */
template<unsigned PE, unsigned TILES, typename TCA, typename TW, typename TB, typename TAA, typename TO>
class BatchNormReLuActivation
{
public:
  TW m_scale[PE][TILES];
  TB m_shift[PE][TILES];
  static const unsigned int MAX_OUT_VAL = (1 << (TO::iwidth - 1)) - 1;

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }

	ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &acc, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA scaled = (TAA)acc * (TAA)m_scale[pe][tile];
    TAA shifted = scaled - (TAA)m_shift[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_scale(m_scale[pe][tile]); 
      profiler->update_shift(m_shift[pe][tile]);     
      profiler->update_act_acc(m_scale[pe][tile]);
      profiler->update_act_acc(m_shift[pe][tile]);
      profiler->update_act_acc(acc);
      profiler->update_act_acc(scaled);       
      profiler->update_act_acc(shifted);
    #endif

    TO result = (TO)0;
    if (shifted < (TAA)0)
        result = result;
    else
      if(shifted > MAX_OUT_VAL)
      {
        result = (TO)MAX_OUT_VAL;
      }
      else
      {
        result = (TO)shifted;
      }

      /*if (flag == 1)
        printf("%.16f\n",(float)result);*/

      #ifndef __SYNTHESIS__
        profiler->update_output(result);
      #endif
        
    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output channel/filter batchnorm + ReLu6 as activation function.
 * 
 *  \tparam TCA            DataType of the conv's accumulator
 *  \tparam TW             DataType of the scale (weight)
 *  \tparam TB             DataType of the shift (bias)
 *  \tparam TAA            DataType of the batchnorm's accumulator
 *  \tparam TO             DataType of the activation's output
 *
 */
template<unsigned PE, unsigned TILES, typename TCA, typename TW, typename TB, typename TAA, typename TO>
class BatchNormReLu6Activation
{
public:
  TW m_scale[PE][TILES];
  TB m_shift[PE][TILES];
  static const unsigned int MAX_OUT_VAL = 6;

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }

	ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &acc, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA scaled = (TAA)acc * (TAA)m_scale[pe][tile];
    TAA shifted = scaled - (TAA)m_shift[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_scale(m_scale[pe][tile]); 
      profiler->update_shift(m_shift[pe][tile]);     
      profiler->update_act_acc(m_scale[pe][tile]);
      profiler->update_act_acc(m_shift[pe][tile]);
      profiler->update_act_acc(acc);
      profiler->update_act_acc(scaled);       
      profiler->update_act_acc(shifted);
    #endif

    TO result = (TO)0;
    if (shifted < (TAA)0)
        result = result;
    else
      if(shifted > MAX_OUT_VAL)//if(shifted > (TAA)MAX_OUT_VAL)
      {
        result = (TO)MAX_OUT_VAL;
      }
      else
      {
        result = (TO)shifted;
      }

      /*if (flag == 1)
        printf("%.16f\n",(float)result);*/

      #ifndef __SYNTHESIS__
        profiler->update_output(result);
      #endif
        
    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output channel/filter InstanceNorm + ReLu as activation function.
 * 
 *  \tparam TCA            DataType of the conv's accumulator
 *  \tparam TW             DataType of the scale (weight)
 *  \tparam TB             DataType of the shift (bias)
 *  \tparam TAA            DataType of the batchnorm's accumulator
 *  \tparam TO             DataType of the activation's output
 *
 */
template<unsigned PE, unsigned TILES, typename TCA, typename TW, typename TB, typename TAA, typename TO>
class InstanceNormReLuActivation
{
public:
  TW m_weight[PE][TILES];
  TB m_bias[PE][TILES];
  static const unsigned int MAX_OUT_VAL = (1 << (TO::iwidth - 1)) - 1;

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }

	ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &acc, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA scaled = (TAA)acc * (TAA)m_weight[pe][tile];
    TAA shifted = scaled - (TAA)m_bias[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_scale(m_weight[pe][tile]); 
      profiler->update_shift(m_bias[pe][tile]);     
      profiler->update_act_acc(m_weight[pe][tile]);
      profiler->update_act_acc(m_bias[pe][tile]);
      profiler->update_act_acc(acc);
      profiler->update_act_acc(scaled);       
      profiler->update_act_acc(shifted);
    #endif

    TO result = (TO)0;
    if (shifted < (TAA)0)
        result = result;
    else
      if(shifted > MAX_OUT_VAL)
      {
        result = (TO)MAX_OUT_VAL;
      }
      else
      {
        result = (TO)shifted;
      }

      /*if (flag == 1)
        printf("%.16f\n",(float)result);*/

      #ifndef __SYNTHESIS__
        profiler->update_output(result);
      #endif
        
    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};

/**
 * Use a per-output channel/filter InstanceNorm + ReLu6 as activation function.
 * 
 *  \tparam TCA            DataType of the conv's accumulator
 *  \tparam TW             DataType of the scale (weight)
 *  \tparam TB             DataType of the shift (bias)
 *  \tparam TAA            DataType of the batchnorm's accumulator
 *  \tparam TO             DataType of the activation's output
 *
 */
template<unsigned PE, unsigned TILES, typename TCA, typename TW, typename TB, typename TAA, typename TO>
class InstanceNormReLu6Activation
{
public:
  TW m_weight[PE][TILES];
  TB m_bias[PE][TILES];
  static const unsigned int MAX_OUT_VAL = 6;

public:
  TCA init(unsigned const  tile, unsigned const  pe) const {
  #pragma HLS inline
    return  TCA(0);
  }

	ap_uint<TO::width> activate(unsigned const tile, unsigned const  pe, TCA const &acc, int flag = 0, Profiler *profiler = nullptr) const {
  #pragma HLS inline
    TAA scaled = (TAA)acc * (TAA)m_weight[pe][tile];
    TAA shifted = scaled - (TAA)m_bias[pe][tile];

    #ifndef __SYNTHESIS__
      profiler->update_scale(m_weight[pe][tile]); 
      profiler->update_shift(m_bias[pe][tile]);     
      profiler->update_act_acc(m_weight[pe][tile]);
      profiler->update_act_acc(m_bias[pe][tile]);
      profiler->update_act_acc(acc);
      profiler->update_act_acc(scaled);       
      profiler->update_act_acc(shifted);
    #endif

    TO result = (TO)0;
    if (shifted < (TAA)0)
        result = result;
    else
      if(shifted > MAX_OUT_VAL)
      {
        result = (TO)MAX_OUT_VAL;
      }
      else
      {
        result = (TO)shifted;
      }

      if (flag == 1)
        printf("%.16f\n",(float)shifted);

      #ifndef __SYNTHESIS__
        profiler->update_output(result);
      #endif
        
    return *reinterpret_cast<ap_uint<TO::width>*>(&result);
  }
};
#endif
