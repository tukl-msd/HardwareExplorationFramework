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
 *  \file utils.hpp
 *
 *  Library of templated HLS functions for DNN deployment. 
 *  This file defines the utility functions.
 *
 *
 ************************************************************************************/


#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <cstddef>

//- Static Evaluation of ceil(log2(x)) ---------------------------------------
template<size_t N> struct clog2 {
  static unsigned const  value = 1 + ((N&1) == 0? clog2<N/2>::value : clog2<N/2+1>::value);
};
template<> struct clog2<0> {};
template<> struct clog2<1> { static unsigned const  value = 0; };
template<> struct clog2<2> { static unsigned const  value = 1; };

//- Helpers to get hold of types ---------------------------------------------
template<typename T> struct first_param {};
template<typename R, typename A, typename... Args>
struct first_param<R (*)(A, Args...)> { typedef A  type; };
template<typename C, typename R, typename A, typename... Args>
struct first_param<R (C::*)(A, Args...)> { typedef A  type; };

//- Resource Representatives -------------------------------------------------
class ap_resource_dflt {};
class ap_resource_lut {};
class ap_resource_dsp {};
//- Resource Representatives for sliding window-------------------------------
class ap_resource_lutram {};
class ap_resource_bram {};
class ap_resource_uram {};

/**
 * \brief   Stream logger - Logging call to dump on file - not synthezisable
 *
 *
 * \tparam     BitWidth    Width, in number of bits, of the input (and output) stream
 *
 * \param      layer_name   File name of the dump
 * \param      log          Input (and output) stream
 *
 */
template < unsigned int BitWidth >
void logStringStream(const char *layer_name, hls::stream<ap_uint<BitWidth> > &log){
    std::ofstream ofs(layer_name);
    hls::stream<ap_uint<BitWidth> > tmp_stream;
	
  while(!log.empty()){
    ap_uint<BitWidth> tmp = (ap_uint<BitWidth>) log.read();
    ofs << std::hex << tmp << std::endl;
    tmp_stream.write(tmp);
  }

  while(!tmp_stream.empty()){
    ap_uint<BitWidth> tmp = tmp_stream.read();
    log.write((ap_uint<BitWidth>) tmp);
  }

  ofs.close();
}

/**
 * \brief   Stream logger - Logging call to dump on file - not synthezisable
 *
 *
 * \tparam     T          Type of the value, e.g. ap_fixed<16,6>, the value will be reinterpret_cast to this type before writing to file
 *
 * \param      log_path   File name of the dump
 * \param      stream     Input (and output) stream
 *
 */
template < typename T, unsigned int NumChannels, unsigned int SIMD = 1,typename TI>
void logStringStreamDtype(const char *log_path, hls::stream<TI> &stream, bool append = false){
  std::ios_base::openmode mode = append ? std::ios::app : std::ios::out;
  std::ofstream ofs(log_path, mode);
  hls::stream<TI> tmp_stream("tmp_stream logStringStream");
	int i=0;
  while(!stream.empty()){
    ap_uint<TI::width> tmp = (ap_uint<TI::width>) stream.read();
    for (unsigned j=0; j < SIMD; j++){
      ap_uint<T::width> tmp2 = tmp((j+1)*T::width-1, j*T::width);
      ofs << *reinterpret_cast<T*>(&tmp2) << " ";
      if((i+1)%NumChannels==0){
        ofs << std::endl;}
      i++;
    }
    tmp_stream.write(tmp);
  }

  while(!tmp_stream.empty()){
    TI tmp = tmp_stream.read();
    stream.write((TI) tmp);
  }

  ofs.close();
}

#endif
