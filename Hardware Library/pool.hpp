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
 ******************************************************************************/
 
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
 *  \file pool.hpp
 * 
 *  Library of templated HLS functions for DNN deployment.
 *  This file defines the pool activations
 *
 ************************************************************************************/


#ifndef POOL_HPP
#define POOL_HPP

#include "interpret.hpp"

/*!
 * \brief PoolFunction: General contract for pool functions.
 *
 * This class itself has no formal significance for the implementation
 * of the pool function. It provides a guidence for specific pool function to be used in Pool_batch
 * 
 * \tparam TA Datatype of the internal accumulation in the pool function
 * \tparam TO Datatype of the output generated by the pool function
 * \tparam size Additional optional unsigned parameter to be used in pool or activate
 *
 */
template<typename TA, typename TO, unsigned size>
class PoolFunction {
public:
/*!
 * \brief init: initialization function returning the datatype for the accumulators
*/
  TA init(void) const {
#pragma HLS inline
    return  TA(0);
  }

/*!
 * \brief pool: computes the pooling algorithm (e.g., max, avg, sum)
 *
 * \param input Input value to be used in the pool function 
 * \param accu  Value already computed in previous iterations
*/
  TA pool(TA const &input, TA const &accu) const;
/*!
 * \brief activate: compute the output of pooling algorithm (e.g., max, avg, sum)
 *
 * \param accu Value already computed in previous iterations
*/
  TO activate(TA const &accu) const;
};

/*!
 * \brief MaxPoolFunction: Implementing max pool 
 *
 * This class inherits from the generic Poolfunction to implement Max Pool
 * 
 * \tparam T Datatype of the input value and the accu value containing the previously computed max
 * \tparam size Unused 
 *
 */
template<typename T, unsigned size>
class MaxPoolFunction : public PoolFunction<T, T, size> {
public:

T init(void) const {
#pragma HLS inline
    const T T_MIN_VAL = (T(-1)<0)? 1<<(T::width-1) : 0;
    return  T_MIN_VAL;
  }
/*!
 * \brief pool: computes the max value 
 *
 * \param input Input value to be used in the max pool function 
 * \param accu  Max value already computed in previous iterations
*/
  T pool(T const &input, T const &accu) const{
#pragma HLS inline
    return std::max(input,accu);
  }
/*!
 * \brief activate: compute the output of the max pooling algorithm
 *
 * \param accu Max value already computed and returned
*/  
  T activate(T const &accu) const {
#pragma HLS inline
    return  accu;
  }
};

/*!
 * \brief AvgPoolFunction: Implementing avg pool 
 *
 * This class inherits from the generic Poolfunction to implement Average Pool
 * 
 * \tparam TA Datatype of the internal accumulation in the avg pool function
 * \tparam TO Datatype of the output generated by the avg pool function
 * \tparam size Value used as divisor on the accumulator to generate output 
 *
 */
template<typename TA, typename TO, unsigned size>
class AvgPoolFunction : public PoolFunction<TA, TO, size> {
public:
/*!
 * \brief pool: computes the sum 
 *
 * \param input Input value to be used in the avg pool function 
 * \param accu  Accumulation value already computed in previous iterations
*/
  TA pool(TA const &input, TA const &accu) const{
#pragma HLS inline
    return std::plus<TA>()(input,accu);
  }
/*!
 * \brief activate: compute the output of the avg pooling algorithm
 *
 * \param accu Accumulation value already computed in previous iterations 
*/    
  TO activate(TA const &accu) const {
#pragma HLS inline
    return  (accu/size);
  }
};

/*!
 * \brief AccPoolFunction: Implementing accumulation pool 
 *
 * This class inherits from the generic Poolfunction to implement accumulation Pool
 * 
 * \tparam TA Datatype of the internal accumulation in the avg pool function

 * \tparam size Unused 
 *
 */
template<typename TA, unsigned size>
class AccPoolFunction : public PoolFunction<TA, TA, size> {
public:
/*!
 * \brief pool: computes the sum 
 *
 * \param input Input value to be used in the avg pool function 
 * \param accu  Accumulation value already computed in previous iterations
*/
  TA pool(TA const &input, TA const &accu) const{
#pragma HLS inline
    return std::plus<TA>()(input,accu);
  }
/*!
 * \brief activate: compute the output of the max pooling algorithm
 *
 * \param accu Accumulation value already computed in previous iterations 
*/   
  TA activate(TA const &accu) const {
#pragma HLS inline
    return  accu;
  }
};

/*!
 * \brief QuantAvgPoolFunction: Implementing avg pool with shift instead of 
 * division
 *
 * This class inherits from the generic Poolfunction to implement Average Pool  
 * with shift instead of division
 * 
 * \tparam TA Datatype of the internal accumulation in the quant avg pool function
 * \tparam TO Datatype of the output generated by the quant avg pool function
 * \tparam size Number of right shifts applied to generate output 
 *
 */
template<typename TA, typename TO, unsigned size>
class QuantAvgPoolFunction : public PoolFunction<TA, TO, size> {
public:
/*!
 * \brief pool: computes the sum 
 *
 * \param input Input value to be used in the avg pool function 
 * \param accu  Accumulation value already computed in previous iterations
*/
  TA pool(TA const &input, TA const &accu) const{
#pragma HLS inline
    return input + accu;
  }
/*!
 * \brief activate: compute the output of the quant avg pooling algorithm
 *
 * \param accu Accumulation value already computed in previous iterations
*/    
  TO activate(TA const &accu) const {
#pragma HLS inline
    return  TO(accu>>size);
  }
};


/**
 * \brief Adaptive_Avg_Pool_2d function
 *
 * The function performs an Adaptive_Avg_Pool_2d pool operation on the input stream. 
 *
 * \tparam Channels   Number of channels - input is equal to output -
 * \tparam PE         Number of channels in the pool layer computed in parallel - currently only 1 is supported -
 * \tparam IFM_Dim    Width of the input feature map
 * \tparam OFM_Dim    Width of the output feature map - (OFM) currently only 1 is supported
 * \tparam Act_dtype  DataType of the activation
 * \tparam Acc_dtype  DataType of the accumulator
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template<
  unsigned Channels, unsigned PE, unsigned IFM_Dim, unsigned OFM_Dim,
  typename Act_dtype,typename Acc_dtype,
  typename TI, typename TO
>
void Adaptive_Avg_Pool_2d(hls::stream<TI> &in,
                  hls::stream<TO> &out,
                  int const  reps) {
  assert(OFM_Dim == 1);
  assert(Channels%PE == 0);
  unsigned const  SF = IFM_Dim*IFM_Dim;

  Acc_dtype  accu[PE][Channels/PE];
#pragma HLS ARRAY_PARTITION variable=accu complete dim=0
  for(unsigned r = 0; r < reps; r++) {
  for(unsigned i = 0; i < SF; i++) {
  for(unsigned c = 0; c < Channels/PE; c++) {
#pragma HLS PIPELINE II=1
    TI  pixel_slice;

    // Threshold Initialisation
    if(i == 0) {
      for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
        accu[pe][c] = 0;
      }
    }

    for(unsigned  pe = 0; pe < PE; pe++) {
#pragma HLS UNROLL
      pixel_slice = in.read();
      Act_dtype const casted_slice = *reinterpret_cast<Act_dtype const *>(&pixel_slice);
      accu[pe][c] += casted_slice;

    // keep track of which folded synapse/neuron we are processing
    if(i == SF-1) {
      TO  outElem;
      accu[pe][c] /= SF;
      Act_dtype casted_res = accu[pe][c];
      outElem = *reinterpret_cast<TO *>(&casted_res);
      out.write(outElem);
    }
    }
  }
  }
  }
}

#endif
