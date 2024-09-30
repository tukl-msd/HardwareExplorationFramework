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
 *  \file stream-tools.h
 *
 *  Library of templated HLS functions for DNN deployment. 
 *  This file lists a set of convenience funtions used to adapt stream size, 
 *  padding, and concating streams.
 *
 *
 ************************************************************************************/

#ifndef STREAMTOOLS_H
#define STREAMTOOLS_H

#include "ap_axi_sdata.h"

/**
 * \brief   Stream limiter - limits the number of stream packets
 *
 * The block only let the first NumAllowed elements of a stream to pass through, the remainder
 * (NumTotal-NumAllowed) are consumed from input but not re-emitted from the output. 
 * Useful to remove padding 
 *
 * \tparam     DataWidth    Width, in number of bits, of the input and output stream
 * \tparam     NumAllowed   Number of words to pass through
 * \tparam     NumTotal     Total number of words (NumAllowed+NumDropped)
 *
 * \param      in           Input stream
 * \param      out          Output stream
 *
 */
template<unsigned int DataWidth,    
		unsigned int NumAllowed, 	
		unsigned int NumTotal       
>
void StreamLimiter(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out) {
  CASSERT_DATAFLOW(NumTotal >= NumAllowed);
  unsigned int numLeft = NumAllowed;
  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II=1
    ap_uint<DataWidth> e = in.read();
    if (numLeft > 0) {
      out.write(e);
      numLeft--;
    }
  }
}

/**
 * \brief   Stream limiter batch - limits the number of stream packets multiple times
 *
 * The block only let the first NumAllowed elements of a stream to pass through, the remainder
 * (NumTotal-NumAllowed) are consumed from input but not re-emitted from the output. 
 * Useful to remove padding on multiple images (numReps)
 *
 * \tparam     DataWidth    Width, in number of bits, of the input and output stream
 * \tparam     NumAllowed   Number of words to pass through
 * \tparam     NumTotal     Total number of words (NumAllowed+NumDropped)
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the StreamLimiter function has to be called
 *
 */
template<unsigned int DataWidth,	
		unsigned int NumAllowed, 	
		unsigned int NumTotal       
>
void StreamLimiter_Batch(hls::stream<ap_uint<DataWidth> > & in,
		hls::stream<ap_uint<DataWidth> > & out, unsigned int numReps) {
  for (unsigned int rep = 0; rep < numReps; rep++) {
    StreamLimiter<DataWidth, NumAllowed, NumTotal>(in, out);
  }
}

/**
 * \brief   Stream Padding - Padds the input with zeroes for when the sliding window is
 *          centered on border pixels
 *
 * Used to add padding to the input with zeroes in case the sliding window is
 * centered on border pixels 
 *
 * \tparam     ImgDim          Size of the input feature map
 * \tparam     KernelDim       Size of the sliding window
 * \tparam     Stride          Stride of the sliding window
 * \tparam     NumChannels     Amount of channels of the input feature map
 * \tparam     In_t            Input datatype
 * \tparam     PaddingStyle    Type of padding that will be applied
 * 
 * \param      in              Input stream
 * \param      out             Output stream
 *
 */
/*template<	unsigned int ImgDim, 
			unsigned int KernelDim, 
			unsigned int Stride, 
			unsigned int NumChannels,
			typename In_t,
      unsigned int PaddingStyle=2>
void SameResize(hls::stream<ap_uint<NumChannels* In_t::width> > &in, 
				hls::stream<ap_uint<NumChannels* In_t::width> > &out){

	// Number of "same" windows over the input data
	constexpr unsigned int SameWindows = (ImgDim) / Stride + ((ImgDim % Stride) > 0);
	
	// Number of elements to generate as output per dimension
	constexpr unsigned int OutputDim = KernelDim + Stride * (SameWindows - 1);

	// Padding
	constexpr unsigned int Padding = OutputDim - ImgDim;

	// Padding Up and Left
  constexpr unsigned int PaddingUp = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);
  constexpr unsigned int PaddingLeft = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding - PaddingUp;
	constexpr unsigned int PaddingRight = Padding - PaddingLeft;
	ap_uint<NumChannels* In_t::width> outData, inData;

	for(unsigned int y = 0; y<OutputDim; y++){
		for(unsigned int x=0; x < OutputDim; x++){
#pragma HLS PIPELINE II=1				

			// Padding Rows
			if(y < PaddingUp || y >= (OutputDim - PaddingDown)){
				outData = 0;
			}
			// Padding Cols
			else if(x < PaddingLeft || x >= (OutputDim - PaddingRight)){
				outData = 0;
			}
			// No Padding
			else{
				inData = in.read();
				outData = inData;
			}

			out.write(outData);
		}
	}
}*/

template
<
unsigned int SIMD,
unsigned int ImgDim, 
unsigned int KernelDim, 
unsigned int Stride, 
unsigned int NumChannels,
typename In_t,
unsigned int PaddingStyle=2
>
void SameResize(hls::stream<ap_uint<SIMD*In_t::width>> &in, 
				hls::stream<ap_uint<SIMD*In_t::width>> &out){

	// Number of "same" windows over the input data
	constexpr unsigned int SameWindows = (ImgDim) / Stride + ((ImgDim % Stride) > 0);
	
	// Number of elements to generate as output per dimension
	constexpr unsigned int OutputDim = KernelDim + Stride * (SameWindows - 1);

	// Padding
	constexpr unsigned int Padding = OutputDim - ImgDim;

	// Padding Up and Left
  constexpr unsigned int PaddingUp = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);
  constexpr unsigned int PaddingLeft = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding - PaddingUp;
	constexpr unsigned int PaddingRight = Padding - PaddingLeft;
	ap_uint<SIMD*In_t::width> outData, inData;

#ifndef __SYNTHESIS__
	/*std::cout << "ImgDim: " << ImgDim << std::endl;
	std::cout << "KernelDim: " << KernelDim << std::endl;
	std::cout << "Stride: " << Stride << std::endl;
	std::cout << "OutputDim: " << OutputDim << std::endl;
	std::cout << "PaddingUp: " << PaddingUp << std::endl;
	std::cout << "PaddingLeft: " << PaddingLeft << std::endl;
	std::cout << "PaddingRight: " << PaddingRight << std::endl;
	std::cout << "PaddingDown: " << PaddingDown << std::endl;
	std::cout << "==========================================" << std::endl;*/
#endif

	for(unsigned int y = 0; y<OutputDim; y++){
		for(unsigned int x=0; x < OutputDim; x++){
			for(unsigned int c=0; c < NumChannels/SIMD; c++){
#pragma HLS PIPELINE II=1				

				// Padding Rows
				if(y < PaddingUp || y >= (OutputDim - PaddingDown)){
					outData = 0;
				}
				// Padding Cols
				else if(x < PaddingLeft || x >= (OutputDim - PaddingRight)){
					outData = 0;
				}
				// No Padding
				else{
					inData = in.read();
					outData = inData;
				}

				out.write(outData);
			}
		}
	}
}

/**
 * \brief   Stream Padding - Padds the input of multiple frames with zeroes
 *          for when the sliding window is centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels 
 *
 * \tparam     ImgDim          Size of the input feature map
 * \tparam     KernelDim       Size of the sliding window
 * \tparam     Stride          Stride of the sliding window
 * \tparam     NumChannels     Amount of channels of the input feature map
 * \tparam     In_t            Input datatype
 * \tparam     PaddingStyle    Type of padding that will be applied
 * 
 * \param      in              Input stream
 * \param      out             Output stream
 * \param      numReps         Amount of frames / images
 *
 */
/*template
<
unsigned int ImgDim, 
unsigned int KernelDim, 
unsigned int Stride, 
unsigned int NumChannels,
typename In_t,
unsigned int PaddingStyle=2
>
void SameResize_Batch(hls::stream<ap_uint<NumChannels* In_t::width> > &in, 
					  hls::stream<ap_uint<NumChannels* In_t::width> > &out, 
		const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		SameResize<ImgDim, KernelDim, Stride, NumChannels, In_t, PaddingStyle>(in, out);
	}

}*/

template
<
unsigned int SIMD,
unsigned int ImgDim, 
unsigned int KernelDim, 
unsigned int Stride, 
unsigned int NumChannels,
typename In_t,
unsigned int PaddingStyle=2
>
void SameResize_Batch(hls::stream<ap_uint<SIMD*In_t::width>> &in, 
					  hls::stream<ap_uint<SIMD*In_t::width>> &out, 
					  const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		SameResize<SIMD, ImgDim, KernelDim, Stride, NumChannels, In_t, PaddingStyle>(in, out);
	}
}



/**
 * \brief   Stream cast - Casts the input stream to a different datatype (OutT)
 *
 * Used to upscale or downscale a stream, enabling loss of information for downscaling or 
 * 0 padding for upscaling 
 *
 * \tparam     InT          Width, in number of bits, of the input and output stream
 * \tparam     OutT         Number of words to pass through
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the StreamLimiter function has to be called
 *
 */
template<typename InT, typename OutT>
void StreamingCast(hls::stream<InT> & in, hls::stream<OutT> & out, unsigned int numReps) {
  for(unsigned int i = 0; i < numReps; i++) {
#pragma HLS PIPELINE II=1
    out.write((OutT) in.read());
  }
}

/**
 * \brief   FM Padding - Padds the input with zeroes for when the sliding window is
 *          centered on border pixels
 *
 * Used to add padding to the input with zeroes in case the sliding window is
 * centered on border pixels
 *
 * \tparam	ImgDim		Size of the input feature map
 * \tparam	OutputDim		Size of the output feature map
 * \tparam	Padding		Amount of padding in total
 * \tparam	NumChannels	Amount of channels of the input feature map
 * \tparam	SIMD			Input parallelism 
 * \tparam	In_t			Input datatype
 * \tparam	PaddingStyle	Type of padding that will be applied
 *
 * \param		in			Input stream
 * \param		out			Output stream
 *
 */
template<	unsigned int ImgDim,
			unsigned int OutputDim,
			unsigned int Padding,
			unsigned int NumChannels,
			unsigned int SIMD,			
			typename In_t,
      unsigned int PaddingStyle=2>
void FMPadding(hls::stream<ap_uint<SIMD* In_t::width> > &in,
		hls::stream<ap_uint<SIMD* In_t::width> > &out){


	// Padding Up and Left
  constexpr unsigned int PaddingUp = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);
  constexpr unsigned int PaddingLeft = Padding/2 + ((PaddingStyle == 2) ? ((Padding % 2) > 0) : 0);

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding - PaddingUp;
	constexpr unsigned int PaddingRight = Padding - PaddingLeft;
	constexpr unsigned int Folding = NumChannels/SIMD;
	ap_uint<SIMD* In_t::width> outData, inData;

	for(unsigned int y = 0; y<OutputDim; y++){
		for(unsigned int x=0; x < OutputDim; x++){
			for(unsigned int simd=0; simd < Folding; simd++) {
#pragma HLS PIPELINE II=1

			// Padding Rows
			if(y < PaddingUp || y >= (OutputDim - PaddingDown)){
				outData = 0;
			}
			// Padding Cols
			else if(x < PaddingLeft || x >= (OutputDim - PaddingRight)){
				outData = 0;
			}
			// No Padding
			else{
				inData = in.read();
				outData = inData;
			}

			out.write(outData);
			}
		}
	}
}

/**
 * \brief   FM Padding - Padds the input of multiple frames with zeroes
 *          for when the sliding window is centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels
 *
 * \tparam	ImgDim		Size of the input feature map
 * \tparam	OutputDim		Size of the output feature map
 * \tparam	Padding		Amount of padding in total
 * \tparam	NumChannels	Amount of channels of the input feature map
 * \tparam	SIMD			Input parallelism 
 * \tparam	In_t			Input datatype
 * \tparam	PaddingStyle	Type of padding that will be applied
 *
 * \param		in			Input stream
 * \param		out			Output stream
 * \param		numReps		Amount of frames / images
 *
 */
template<	unsigned int ImgDim,
			unsigned int OutputDim,
			unsigned int Padding,
			unsigned int NumChannels,
			unsigned int SIMD,
			typename In_t,
      unsigned int PaddingStyle=2>
void FMPadding_Batch(hls::stream<ap_uint<SIMD* In_t::width> > &in,
		hls::stream<ap_uint<SIMD* In_t::width> > &out,
		const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		FMPadding<ImgDim, OutputDim, Padding, NumChannels, SIMD, In_t, PaddingStyle>(in, out);
	}

}


/**
 * \brief   FM Padding - Padds the input with zeroes for when the sliding window is
 *          centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels - working on non-square images and padding
 *
 * \tparam	OutputDim_x	Width of the output feature map (padded)
 * \tparam	OutputDim_y	Height of the output feature map (padded)
 * \tparam	Padding_x	Amount of padding over x axis
 * \tparam	Padding_y	Amount of padding over y axis
 * \tparam	NumChannels	Amount of channels of the input feature map
 * \tparam	SIMD			Input parallelism 
 * \tparam	In_t			Input datatype
 * \tparam	PaddingStyle	Type of padding that will be applied
 *
 * \param		in			Input stream
 * \param		out			Output stream
 *
 */
template<	unsigned int OutputDim_x,
			unsigned int OutputDim_y,
			unsigned int Padding_x,
			unsigned int Padding_y,
			unsigned int NumChannels,
			unsigned int SIMD,			
			typename In_t,
      unsigned int PaddingStyle=2>
void FMPadding_nonsquare(hls::stream<ap_uint<SIMD* In_t::width> > &in,
		hls::stream<ap_uint<SIMD* In_t::width> > &out){


	// Padding Up and Left
  constexpr unsigned int PaddingUp = Padding_y/2 + ((PaddingStyle == 2) ? ((Padding_y % 2) > 0) : 0);
  constexpr unsigned int PaddingLeft = Padding_x/2 + ((PaddingStyle == 2) ? ((Padding_x % 2) > 0) : 0);

	// Padding Down and Right (might be 1 element more than up and left in case of odd padding)
	constexpr unsigned int PaddingDown = Padding_y - PaddingUp;
	constexpr unsigned int PaddingRight = Padding_x - PaddingLeft;
	constexpr unsigned int Folding = NumChannels/SIMD;
	ap_uint<SIMD* In_t::width> outData, inData;

	for(unsigned int y = 0; y<OutputDim_y; y++){
		for(unsigned int x=0; x < OutputDim_x; x++){
			for(unsigned int simd=0; simd < Folding; simd++) {
#pragma HLS PIPELINE II=1

			// Padding Rows
			if(y < PaddingUp || y >= (OutputDim_y - PaddingDown)){
				outData = 0;
			}
			// Padding Cols
			else if(x < PaddingLeft || x >= (OutputDim_x - PaddingRight)){
				outData = 0;
			}
			// No Padding
			else{
				inData = in.read();
				outData = inData;
			}

			out.write(outData);
			}
		}
	}
}

/**
 * \brief   FM Padding Non Square - Padds the input of multiple frames with zeroes
 *          for when the sliding window is centered on border pixels
 *
 * Used to add padding with zeroes to multiple inputs in case the sliding window is
 * centered on border pixels - working on non-square images and padding
 *
 * \tparam	OutputDim_x	Width of the output feature map (padded)
 * \tparam	OutputDim_y	Height of the output feature map (padded)
 * \tparam	Padding_x	Amount of padding over x axis
 * \tparam	Padding_y	Amount of padding over y axis
 * \tparam	NumChannels	Amount of channels of the input feature map
 * \tparam	SIMD			Input parallelism 
 * \tparam	In_t			Input datatype
 * \tparam	PaddingStyle	Type of padding that will be applied
 *
 * \param		in			Input stream
 * \param		out			Output stream
 * \param		numReps		Amount of frames / images
 *
 */
template<	unsigned int OutputDim_x,
			unsigned int OutputDim_y,
			unsigned int Padding_x,
			unsigned int Padding_y,
			unsigned int NumChannels,
			unsigned int SIMD,
			typename In_t,
      unsigned int PaddingStyle=2>
void FMPadding_nonsquare_Batch(hls::stream<ap_uint<SIMD* In_t::width> > &in,
		hls::stream<ap_uint<SIMD* In_t::width> > &out,
		const unsigned int numReps) {
	for (unsigned int rep = 0; rep < numReps; rep++) {
		FMPadding_nonsquare<OutputDim_x, OutputDim_y, Padding_x, Padding_y, NumChannels, SIMD, In_t, PaddingStyle>(in, out);
	}

}


/**
 * \brief   Stream Data Width Converter - Converts the width of the input stream in the output stream
 *
 * Used to upscale or downscale a stream, without any loss of data in the procedure. 
 * For downscaling (InWidth > OutWidth), InWidth has to be a multiple of OutWidth.
 * For upscaling (InWidth < OutWidth), OutWidth has to be a multiple of InWidth.
 *
 * \tparam     InWidth      Width, in number of bits, of the input stream
 * \tparam     OutWidth     Width, in number of bits, of the output stream 
 * \tparam     NumInWords   Number of input words to process
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template
<
unsigned int InWidth,		
unsigned int OutWidth,		
unsigned int NumInWords		
>
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > & in,
									   hls::stream<ap_uint<OutWidth> > & out,
									   const unsigned int numReps)
{
	if (InWidth > OutWidth) {
    // emit multiple output words per input word read
    CASSERT_DATAFLOW(InWidth % OutWidth == 0);
    const unsigned int outPerIn = InWidth / OutWidth;
    const unsigned int totalIters = NumInWords * outPerIn * numReps;
    unsigned int o = 0;
    ap_uint<InWidth> ei = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
      // read new input word if current out count is zero
      if (o == 0) {
        ei = in.read();
	  }
      // pick output word from the rightmost position
      ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
      out.write(eo);
      // shift input to get new output word for next iteration
      ei = ei >> OutWidth;
      // increment written output count
      o++;
      // wraparound indices to recreate the nested loop structure
      if (o == outPerIn) {
        o = 0;
      }
    }
  } else if (InWidth == OutWidth) {
    // straight-through copy
    for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II=1
	  ap_uint<InWidth> e = in.read();
      out.write(e);
    }
  } else { // InWidth < OutWidth
    // read multiple input words per output word emitted
    CASSERT_DATAFLOW(OutWidth % InWidth == 0);
    const unsigned int inPerOut = OutWidth / InWidth;
    const unsigned int totalIters = NumInWords * numReps;
    unsigned int i = 0;
    ap_uint<OutWidth> eo = 0;
    for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
      // read input and shift into output buffer
      ap_uint<InWidth> ei = in.read();
      eo = eo >> InWidth;
      eo(OutWidth - 1, OutWidth - InWidth) = ei;
      // increment read input count
      i++;
      // wraparound logic to recreate nested loop functionality
      if (i == inPerOut) {
        i = 0;
        out.write(eo);
      }
    }
  }
}

/**
 * \brief   Stream Data Width Converter No Multiple - 
 *          Converts the width of the input stream in the output stream for no multiple dimensions
 *
 * Used to downscale a stream, without any loss of data in the procedure. 
 * For downscaling (InWidth > OutWidth), InWidth has to be a multiple of OutWidth.
 *
 * \tparam     InWidth      Width, in number of bits, of the input stream
 * \tparam     OutWidth     Width, in number of bits, of the output stream 
 *
 * \param      in           Input stream
 * \param      out          Output stream
 *
 */
template<
    unsigned int InWidth,    
    unsigned int OutWidth
>
void StreamingDataWidthConverterNoMultiple(
    hls::stream<ap_uint<InWidth> > & in,
    hls::stream<ap_uint<OutWidth> > & out) {
    CASSERT_DATAFLOW((InWidth % 2) == 0);
    CASSERT_DATAFLOW((OutWidth % 2) == 0);
    CASSERT_DATAFLOW(InWidth != OutWidth);
    static unsigned int      offset = 0; 

    if (InWidth > OutWidth){
     
      static ap_uint<OutWidth> remainder = 0;
      ap_uint<InWidth>  valueIn = in.read();
      
      if(offset !=0) {
        ap_uint<OutWidth>   valueOut = 0;
        valueOut = (valueIn(offset-1,0),remainder(OutWidth-offset-1,0));
        valueIn = valueIn(InWidth-1,offset); // leave the next part prepared 
        out.write(valueOut);
      }
      for (; offset <= (InWidth-OutWidth) ; offset+=OutWidth){
        ap_uint<OutWidth>   valueOut = valueIn(OutWidth-1,0);
        valueIn = valueIn(InWidth-1,OutWidth); // leave the next part prepared 
        out.write(valueOut);
      }
      remainder = valueIn;
      if (offset == InWidth)
        offset = 0;
      else
        offset = offset + OutWidth - InWidth;
    }
    else {
      /*OutWidth > InWidth*/
      static ap_uint<InWidth> remainder = 0;
      ap_uint<OutWidth> value = 0;
      if (offset !=0) {
        value(offset-1,0) = remainder(InWidth-1,InWidth-offset);
      }
      for (; offset <= (OutWidth-InWidth); offset+=InWidth){
        ap_uint<InWidth>   aux = in.read();
        value(offset+InWidth-1,offset) = aux;
      }
      if (offset != OutWidth){
        ap_uint<InWidth>   aux = in.read();
        value(OutWidth-1,offset) = aux(OutWidth-offset-1,0);
        remainder                   = aux;
        offset = offset + InWidth - OutWidth;
      }
      else
        offset = 0;
      out.write(value);
    }

}

/**
 * \brief   Stream Multi Chan Data Width Converter - Converts the width of the input stream in the output stream, working on multiple parallel streams
 *
 * Used to upscale or downscale a stream, without any loss of data in the procedure. 
 * For downscaling (InWidth > OutWidth), InWidth has to be a multiple of OutWidth.
 * For upscaling (InWidth < OutWidth), OutWidth has to be a multiple of InWidth.
 * This version works on the MMV structure, with multiple parallel streams
 *
 * \tparam     InWidth      Width, in number of bits, of the input stream
 * \tparam     OutWidth     Width, in number of bits, of the output stream 
 * \tparam     NumInWords   Number of input words to process
 * \tparam     NumVecs      Number of parallel vectors MMV
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords,		// number of input words to process
		unsigned int NumVecs
>
void MultiChanDataWidthConverter_Batch(
	hls::stream<MultiChanData<NumVecs, InWidth> > & in,
	hls::stream<MultiChanData<NumVecs, OutWidth> > & out,
	const unsigned int numReps) {
	if (InWidth > OutWidth) {
		// emit multiple output words per input word read
        CASSERT_DATAFLOW((InWidth % OutWidth) == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		const unsigned int totalIters = NumInWords * outPerIn * numReps;
		unsigned int o = 0;
		MultiChanData<NumVecs, InWidth> ei;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read new input word if current out count is zero
			if (o == 0)
				ei = in.read();
			// pick output word from the rightmost position
			MultiChanData<NumVecs, OutWidth> eo;
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				eo.data[v] = (ei.data[v])(OutWidth - 1, 0);
				// shift input to get new output word for next iteration
				ei.data[v] = ei.data[v] >> OutWidth;
			}
			out.write(eo);
			// increment written output count
			o++;
			// wraparound indices to recreate the nested loop structure
			if (o == outPerIn) {
				o = 0;
			}
		}
	} else if (InWidth == OutWidth) {
		// straight-through copy
		for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II=1
			MultiChanData<NumVecs, InWidth> e = in.read();
			MultiChanData<NumVecs, OutWidth> eo;
			// we don't support typecasting between templated types, so explicitly
			// transfer vector-by-vector here
			for(unsigned int v=0; v < NumVecs; v++) {
#pragma HLS UNROLL
				eo.data[v] = e.data[v];
			}
			out.write(eo);
		}
	} else { // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW((OutWidth % InWidth) == 0);
		const unsigned int inPerOut = OutWidth / InWidth;
		const unsigned int totalIters = NumInWords * numReps;
		unsigned int i = 0;
		MultiChanData<NumVecs, OutWidth> eo;
		for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II=1
			// read input and shift into output buffer
			MultiChanData<NumVecs, InWidth> ei = in.read();
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				eo.data[v] = eo.data[v] >> InWidth;
				(eo.data[v])(OutWidth - 1, OutWidth - InWidth) = ei.data[v];
			}
			// increment read input count
			i++;
			// wraparound logic to recreate nested loop functionality
			if (i == inPerOut) {
				i = 0;
				out.write(eo);
			}
		}
	}
}


/**
 * \brief   Flatten Multi Chan Data - Converts the parallel input stream in a flatten output stream
 *
 * Used to pach a flattened stream into a structure with multiple parallel streams
 *
 * \tparam     NumChannels  Number of channels flattened in the input stream
 * \tparam     DataWidth    Width, in number of bits, of each stream
 *
 * \param      in           Input parallel stream
 * \param      out          Output stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template <unsigned int NumChannels, unsigned int DataWidth>
void FlattenMultiChanData(
	hls::stream<MultiChanData<NumChannels, DataWidth> > & in,
	hls::stream<ap_uint<NumChannels*DataWidth> > & out,
	const unsigned int numReps
) {
	for(unsigned int r = 0; r < numReps; r++) {
#pragma HLS PIPELINE II=1
		MultiChanData<NumChannels, DataWidth> e = in.read();
		ap_uint<NumChannels*DataWidth> o = 0;
		for(unsigned int v = 0; v < NumChannels; v++) {
#pragma HLS UNROLL
			o(DataWidth*(v+1)-1, DataWidth*v) = e.data[v];
		}
		out.write(o);
	}
}

/**
 * \brief   Pack Multi Chan Data - Converts the flatten input stream into a parallel output stream
 *
 * Used to pach a flattened stream into a structure with multiple parallel streams
 *
 * \tparam     NumChannels  Number of channels flattened in the input stream
 * \tparam     DataWidth    Width, in number of bits, of each stream
 *
 * \param      in           Input stream
 * \param      out          Output parallel stream
 * \param      numReps      Number of times the function has to be called
 *
 */
template <unsigned int NumChannels, unsigned int DataWidth>
void PackMultiChanData(
	hls::stream<ap_uint<NumChannels*DataWidth> > & in,
	hls::stream<MultiChanData<NumChannels, DataWidth> > & out,
	const unsigned int numReps
) {
	for(unsigned int r = 0; r < numReps; r++) {
#pragma HLS PIPELINE II=1
		ap_uint<NumChannels*DataWidth> e = in.read();
		MultiChanData<NumChannels, DataWidth> o;
		for(unsigned int v = 0; v < NumChannels; v++) {
#pragma HLS UNROLL
			o.data[v] = e(DataWidth*(v+1)-1, DataWidth*v);
		}
		out.write(o);
	}
}


template<unsigned IW, unsigned OW, unsigned N>
 class WidthAdjustedInputStream {
  hls::stream<ap_uint<OW>>  m_target;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<IW> >&  source, unsigned const  reps) {
    StreamingDataWidthConverter_Batch<IW, OW, N>(source, m_target, reps);
  }
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<OW> >&() {
    return  m_target;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedInputStream<W, W, N> {

  hls::stream<ap_uint<W>> &m_source;

 public:
  WidthAdjustedInputStream(hls::stream<ap_uint<W> >&  source, unsigned const  reps) : m_source(source) {}
  ~WidthAdjustedInputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_source;
  }
};


template<unsigned IW, unsigned OW, unsigned N>
class WidthAdjustedOutputStream {
  hls::stream<ap_uint<IW>>  m_buffer;
  hls::stream<ap_uint<OW>> &m_target;
  unsigned const  m_reps;
  
 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<OW> >&  target, unsigned const  reps) : m_target(target), m_reps(reps) {}
  ~WidthAdjustedOutputStream() {
    StreamingDataWidthConverter_Batch<IW, OW, N>(m_buffer, m_target, m_reps);
  }

 public:
  operator hls::stream<ap_uint<IW> >&() {
    return  m_buffer;
  }
};
template<unsigned W, unsigned N>
 class WidthAdjustedOutputStream<W, W, N> {
  hls::stream<ap_uint<W>> &m_target;

 public:
  WidthAdjustedOutputStream(hls::stream<ap_uint<W> >&  target, unsigned const  reps)
    : m_target(target) {}
  ~WidthAdjustedOutputStream() {}

 public:
  operator hls::stream<ap_uint<W> >&() {
    return  m_target;
  }
};

/**
 * \brief   QDMA stream to normal stream conversion - Reads in a QDMA stream and strips metadata (TLAST, TKEEP)
 *
 * Used as an adapter when connecting blocks through top-level Vitis streams (kernel to kernel or host to plaform streaming)
 *
 * \tparam     DataWidth    Width, in number of bits, of the data on streams
 * \tparam     NumTotal     Total number of words in the input stream
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */
template<unsigned int DataWidth, unsigned int NumTotal>
void Qdma2Stream_Batch(hls::stream<qdma_axis<DataWidth,0,0,0> > & in, hls::stream<ap_uint<DataWidth> > & out, const unsigned int numReps){
	//TODO: CASSERT_DATAFLOW to ensure DataWidth is power of 2 between 8 and 512
	for (unsigned int image = 0; image < numReps; image++) {
		for (unsigned int word = 0; word < NumTotal; word++) {
#pragma HLS PIPELINE II=1
			out.write(in.read().get_data());
		}
	}
}

/**
 * \brief   Normal stream to QDMA stream conversion - Reads in a stream and outputs a QDMA stream including metadata (TLAST, TKEEP)
 *
 * Used as an adapter when connecting blocks through top-level Vitis streams (kernel to kernel or host to plaform streaming)
 *
 * \tparam     DataWidth    Width, in number of bits, of the data on streams
 * \tparam     NumTotal     Total number of words in the input stream
 *
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */
template<unsigned int DataWidth, unsigned int NumTotal>
void Stream2Qdma_Batch(hls::stream<ap_uint<DataWidth> > & in, hls::stream<qdma_axis<DataWidth,0,0,0> > & out, const unsigned int numReps){
	for (unsigned int image = 0; image < numReps; image++) {
		for (unsigned int word = 0; word < NumTotal; word++) {
#pragma HLS PIPELINE II=1
			qdma_axis<DataWidth,0,0,0> temp;
			temp.set_data(in.read());
			temp.set_keep(-1);
			temp.set_last(word == NumTotal-1);
			out.write(temp);
		}
	}
}

/**
 * \brief   Stream Duplicator - Reads in a stream and writes the data into two identical streams
 *
 * Used to generate the inputs to the bypass and convolutional branches in Resnet-50
 *
 * \tparam     DataWidth    Width, in number of bits, of the streams
 * \tparam     NumTotal     Total number of words in the input stream
 *
 * \param      in           Input stream
 * \param      out1         Output stream I
 * \param      out2         Output stream II
 *
 */
template
<
unsigned int NumTotal,
typename In_t,
int DataWidth
>
void DuplicateStreams(hls::stream<ap_uint<DataWidth> > & in,
					  hls::stream<ap_uint<DataWidth> > & out1,
					  hls::stream<ap_uint<DataWidth> > & out2, Profiler *profiler = nullptr)
{
	
	for (unsigned int i = 0; i < NumTotal; i++)
	{
	#pragma HLS PIPELINE II=1		
		ap_uint<DataWidth> e = in.read();		
		out1.write(e);
		out2.write(e);

		#ifndef __SYNTHESIS__
			In_t temp = *reinterpret_cast<In_t*>(&e);
			profiler->update_input(temp);
			profiler->update_output(temp);
		#endif
	}
}

template
<
unsigned int NumTotal,
typename In_t,
int DataWidth
>
void TriplicateStreams(hls::stream<ap_uint<DataWidth> > & in,
					  hls::stream<ap_uint<DataWidth> > & out1,
					  hls::stream<ap_uint<DataWidth> > & out2,
					  hls::stream<ap_uint<DataWidth> > & out3, Profiler *profiler = nullptr)
{
	
	for (unsigned int i = 0; i < NumTotal; i++)
	{
	#pragma HLS PIPELINE II=1		
		ap_uint<DataWidth> e = in.read();		
		out1.write(e);
		out2.write(e);
		out3.write(e);

		#ifndef __SYNTHESIS__
			In_t temp = *reinterpret_cast<In_t*>(&e);
			profiler->update_input(temp);
			profiler->update_output(temp);
		#endif
	}
}

/**
 * \brief   Batch Stream Duplicator - Reads in a stream multiple times and writes the data into two identical streams
 *
 * Used to generate the inputs to the bypass and convolutional branches in Resnet-50 when dealing with multiple 'frames'
 *
 * \tparam     DataWidth    Width, in number of bits, of the streams
 * \tparam     NumTotal     Total number of words in the input stream
 *
 * \param      in           Input stream
 * \param      out1         Output stream I
 * \param      out2         Output stream II
 * \param      reps      	Number of frames / images
 *
 */
template
<
unsigned int NumTotal,
typename In_t,
// safely deducible (stream width must be int though!)
int DataWidth
>
void DuplicateStreams_Batch(hls::stream<ap_uint<DataWidth> > & in,
						    hls::stream<ap_uint<DataWidth> > & out1,
						    hls::stream<ap_uint<DataWidth> > & out2, const unsigned int reps, Profiler *profiler = nullptr)
{	
	for (unsigned int image = 0; image < reps; image++)
	{
		DuplicateStreams<NumTotal, In_t>(in, out1, out2, profiler);
	}
}

template
<
unsigned int NumTotal,
typename In_t,
// safely deducible (stream width must be int though!)
int DataWidth
>
void TriplicateStreams_Batch(hls::stream<ap_uint<DataWidth> > & in,
						    hls::stream<ap_uint<DataWidth> > & out1,
						    hls::stream<ap_uint<DataWidth> > & out2,
							hls::stream<ap_uint<DataWidth> > & out3, const unsigned int reps, Profiler *profiler = nullptr)
{	
	for (unsigned int image = 0; image < reps; image++)
	{
		TriplicateStreams<NumTotal, In_t>(in, out1, out2, out3, profiler);
	}
}

template
<
unsigned int NumTotal,
typename In_t,
// safely deducible (stream width must be int though!)
int StreamW, int MemW
>
void DuplicateStreams_Batch_OffChipAct(hls::stream<ap_uint<StreamW> > & in_stream,
						    		   hls::stream<ap_uint<StreamW> > & out1_stream,
						    		   ap_uint<MemW> *out2_mem, const unsigned int reps, Profiler *profiler = nullptr)
{	

#pragma HLS INLINE
	
	hls::stream<ap_uint<MemW>> s2m_out2_stream;
	//hls::stream<ap_uint<StreamW>> out2_stream;

	WidthAdjustedOutputStream <StreamW, MemW, NumTotal>  out2_stream (s2m_out2_stream,  reps);// out2_stream -> s2m_out2_stream
	DuplicateStreams_Batch<NumTotal, In_t>(in_stream, out1_stream, static_cast<hls::stream<ap_uint<StreamW>>&> (out2_stream), reps, profiler);

	//DuplicateStreams_Batch<NumTotal, In_t>(in_stream, out1_stream, out2_stream, reps, profiler);

	//StreamingDataWidthConverter_Batch<StreamW, MemW, NumTotal>(out2_stream, s2m_out2_stream, reps);

	Stream2Mem_Batch<NumTotal>(s2m_out2_stream, out2_mem, reps);
	//Stream2Mem<NumTotal>(s2m_out2_stream, out2_mem);
	
}

/**
 * \brief   Element-Wise Addition - Reads in data elements from two streams and writes the sum of these elements to an output
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype 
 * \tparam     Out_t        Datatype of the accumulation output 
 * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     offset       Offset value for the accumulation
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 *
 */

template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal, 
          int offset = 0>
void AddStreams_Ch(hls::stream<ap_uint<NumChannels * In1_t::width>> &in1, hls::stream<ap_uint<NumChannels * In2_t::width>> &in2,
                hls::stream<ap_uint<NumChannels * Out_t::width>> &out) {

  for (unsigned int i = 0; i < NumTotal; i++) {
#pragma HLS PIPELINE II = 1
    ap_uint<NumChannels * In1_t::width> e1 = in1.read();
    ap_uint<NumChannels * In2_t::width> e2 = in2.read();
    ap_uint<NumChannels * Out_t::width> e;
    for (unsigned int j = 0; j < NumChannels; j++) {
#pragma HLS UNROLL
      In1_t op1 = e1((j + 1) * In1_t::width - 1, j * In1_t::width);
      In2_t op2 = e2((j + 1) * In2_t::width - 1, j * In2_t::width);
      Out_t sum = op1 + op2 + offset;
      e((j + 1) * Out_t::width - 1, j * Out_t::width) = sum;
    }
    out.write(e);
  }
}

/**
 * \brief   
 *
 * Used to implement point-wise addition in Resnet-50 for multiple images
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype 
 * \tparam     Out_t        Datatype of the accumulation output 
 * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     offset       Offset value for the accumulation
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */
template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          int offset = 0>
void AddStreams_Ch_Batch(hls::stream<ap_uint<NumChannels * In1_t::width>> &in1, hls::stream<ap_uint<NumChannels * In2_t::width>> &in2,
                hls::stream<ap_uint<NumChannels * Out_t::width>> &out, const unsigned int numReps) {
  for (unsigned int image = 0; image < numReps; image++) {
    AddStreams_Ch<NumChannels, In1_t, In2_t, Out_t, NumTotal, offset>(in1, in2, out);
  }
}

/**
 * 
 * With data conversion
 */

template <unsigned int SIMD,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          int offset = 0>
void AddStreams_Type(hls::stream<ap_uint<SIMD * In1_t::width>> &in1,
					 hls::stream<ap_uint<SIMD * In2_t::width>> &in2,
                	 hls::stream<ap_uint<SIMD * Out_t::width>> &out, int flag = 0, Profiler *profiler = nullptr)
{

  for (unsigned int i = 0; i < NumTotal; i++)
  {
	#pragma HLS PIPELINE II = 1
	ap_uint<SIMD * In1_t::width> e1 = in1.read();
    ap_uint<SIMD * In2_t::width> e2 = in2.read();
    ap_uint<SIMD * Out_t::width> e;
    for (unsigned int j = 0; j < SIMD; j++)
	{
	#pragma HLS UNROLL
    	ap_int <In1_t::width> op1_temp = e1((j + 1) * In1_t::width - 1, j * In1_t::width);
    	ap_int <In2_t::width> op2_temp = e2((j + 1) * In2_t::width - 1, j * In2_t::width);
		In1_t op1 = *reinterpret_cast<In1_t *>(&op1_temp);
		In2_t op2 = *reinterpret_cast<In2_t *>(&op2_temp);

      	Out_t sum = (Out_t)op1 + (Out_t)op2;

		#ifndef __SYNTHESIS__
			if (flag == 1)
			{
				Out_t op1_ = (Out_t)op1;
				Out_t op2_ = (Out_t)op2;
				printf("%.16f\n",(float)op1_, (float)op2_, (float)sum);
			}

			profiler->update_input(op1);
			profiler->update_input_dublicate(op2);
			profiler->update_output(sum);
		#endif

      	ap_uint <Out_t::width> sum_temp = *reinterpret_cast<ap_uint <Out_t::width> *>(&sum);
		e((j + 1) * Out_t::width - 1, j * Out_t::width) = sum_temp;
    }
    out.write(e);
  }
}

template <unsigned int SIMD,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          int offset = 0>
void AddStreams_Type_Batch(hls::stream<ap_uint<SIMD * In1_t::width>> &in1,
						   hls::stream<ap_uint<SIMD * In2_t::width>> &in2,
                		   hls::stream<ap_uint<SIMD * Out_t::width>> &out,
						   const unsigned int numReps, int flag = 0, Profiler *profiler = nullptr)
{
  for (unsigned int image = 0; image < numReps; image++)
  {
    AddStreams_Type<SIMD, In1_t, In2_t, Out_t, NumTotal, offset>(in1, in2, out, flag, profiler);
  }
}

/**
 * \brief   Addition Layer - Reads in two streams and writes the sum of these streams to an output
 *
 * Used to merge the outputs of the bypass and convolutional branches in Resnet-50
 *
 * \tparam     NumChannels  Amount of channels of the streams
 * \tparam     In1_t        First operand datatype
 * \tparam     In2_t        Second operand datatype 
 * \tparam     Out_t        Datatype of the accumulation output  * \tparam     NumTotal     Total number of words in the input streams
 * \tparam     PECount      Amount of processing elements working in parallel 
 * \tparam     offset       Offset value for the accumulation 
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 * \param      numReps      Number of frames / images
 *
 */
template <unsigned int NumChannels,
          typename In1_t,
          typename In2_t,
          typename Out_t,
          unsigned int NumTotal,
          unsigned int PECount, 
          int offset = 0>
void AddStreamsLayer_Batch(hls::stream<ap_uint<NumChannels * In1_t::width>> &in1,
						  hls::stream<ap_uint<NumChannels * In2_t::width>> &in2,
                          hls::stream<ap_uint<NumChannels * Out_t::width>> &out, const unsigned int numReps) {
#pragma HLS INLINE
  CASSERT_DATAFLOW(NumChannels % PECount == 0);
  hls::stream<ap_uint<PECount * In1_t::width>> in_folded1;
  hls::stream<ap_uint<PECount * In2_t::width>> in_folded2;
  hls::stream<ap_uint<PECount * Out_t::width>> out_folded;
  StreamingDataWidthConverter_Batch<NumChannels * In1_t::width, PECount * In1_t::width, NumTotal>(in1, in_folded1, numReps);
  StreamingDataWidthConverter_Batch<NumChannels * In2_t::width, PECount * In2_t::width, NumTotal>(in2, in_folded2, numReps);
  AddStreams_Ch_Batch<PECount, In1_t, In2_t, Out_t, NumTotal *(NumChannels / PECount),offset>(in_folded1, in_folded2, out_folded, numReps);
  StreamingDataWidthConverter_Batch<PECount * Out_t::width, NumChannels * Out_t::width, NumTotal *(NumChannels / PECount)>(out_folded, out, numReps);
}

/**
 * \brief   Addition Layer - Reads in two streams and writes the sum of these streams to an output
 *
 * Used to merge the outputs of the bypass and convolutional branches in Resnet-50
 *
 * \tparam     ReadsPerIn1  Number of the reads per In1 (direct input)
 * \tparam     ReadsPerIn2	Number of the reads per In2 (dublicate input)
 * \tparam     SIMD      	Amount of processing elements working in parallel (parallelism of the direct input)
 *
 * \param      in1          Input stream I
 * \param      in2          Input stream II
 * \param      out          Output stream
 * \param      reps      	Number of frames / images
 *
 */
template
<
unsigned int ReadsPerIn1,
unsigned int ReadsPerIn2,
unsigned int SIMD,	
typename In1_t,
typename In2_t,
typename Out_t,      
// safely deducible (stream width must be int though!)
int In1StreamW, int In2StreamW, int OutStreamW
>
void AddStreams_Batch(hls::stream<ap_uint<In1StreamW>> &in1,
					  hls::stream<ap_uint<In2StreamW>> &in2,
                      hls::stream<ap_uint<OutStreamW>> &out, const unsigned int reps)
{
#pragma HLS INLINE

	CASSERT_DATAFLOW(In1StreamW % SIMD == 0);

	WidthAdjustedInputStream <In1StreamW, SIMD * In1_t::width, ReadsPerIn1>  wa_in1 (in1,  reps);// wa_in1 <- in1  
	WidthAdjustedInputStream <In2StreamW, SIMD * In2_t::width, ReadsPerIn2>  wa_in2 (in2,  reps);// wa_in2 <- in2 
  	WidthAdjustedOutputStream <SIMD * Out_t::width, OutStreamW, ReadsPerIn1>  add_out (out,  reps);// add_out -> out


	AddStreams_Type_Batch<SIMD, In1_t, In2_t, Out_t, ReadsPerIn1>(wa_in1, wa_in2, add_out, reps);

}

template
<
unsigned int ReadsPerIn1,
unsigned int ReadsPerIn2,
unsigned int SIMD,	
typename In1_t,
typename In2_t,
typename Out_t,      
// safely deducible (stream width must be int though!)
int In1StreamW, int In2MemW, int OutStreamW
>
void AddStreams_Batch_OffChipAct(hls::stream<ap_uint<In1StreamW>> &in1_stream,
					  			ap_uint<In2MemW> *in2_mem,
                      			hls::stream<ap_uint<OutStreamW>> &out_stream, const unsigned int reps, Profiler *profiler = nullptr)
{
#pragma HLS INLINE

	CASSERT_DATAFLOW(In1StreamW % SIMD == 0);
	CASSERT_DATAFLOW(In2MemW % (In2_t::width) == 0);

	WidthAdjustedInputStream <In1StreamW, SIMD * In1_t::width, ReadsPerIn1>  wa_in1_stream (in1_stream,  reps);// wa_in1_stream <- in1_stream  
  	WidthAdjustedOutputStream <SIMD * Out_t::width, OutStreamW, ReadsPerIn1>  add_out_stream (out_stream,  reps);// add_out_stream -> out_stream

	hls::stream<ap_uint<In2MemW>> m2s_in2_stream("m2s_in2_stream");
	Mem2Stream_Batch<ReadsPerIn2>(in2_mem, m2s_in2_stream, reps);
	//Mem2Stream<ReadsPerIn2>(in2_mem, m2s_in2_stream);
	WidthAdjustedInputStream <In2MemW, SIMD*In2_t::width, ReadsPerIn2>  wa_in2_stream (m2s_in2_stream, reps);//wa_in2_stream <- m2s_in2_stream

	AddStreams_Type_Batch<SIMD, In1_t, In2_t, Out_t, ReadsPerIn1>(wa_in1_stream, wa_in2_stream, add_out_stream, reps, profiler);

}

/**
 * \brief   Expand Stream - Reads in a stream and duplicates the values to increase the size of the feautre map
 * 
 * \tparam     NumChannels  Number of channels or feature maps
 * \tparam	   InDim 	  	Input dimension -Only 1 is supported for now-
 * \tparam	   OutDim 	  	Output dimension
 * \tparam	   IA           Input stream data type
 * \tparam	   OA           Output stream data type
 * 
 * \param      in           Input stream
 * \param      out          Output stream
 * \param      reps      	Number of frames / images
 * 
 */
template
<
unsigned int NumChannels,
unsigned int InDim,
unsigned int OutDim,
typename IA,
typename OA
>
void ExpandStream_Batch(hls::stream<IA> &in, hls::stream<OA> &out, const unsigned int reps){
	#pragma HLS INLINE
	assert(InDim == 1);
	IA buffer[NumChannels];
	for (unsigned int r = 0; r < reps; r++){
		for (unsigned int i = 0; i < OutDim*OutDim; i++){
			for (unsigned int ch = 0; ch < NumChannels; ch++){
				#pragma HLS PIPELINE II=1
				if(i == 0){
					buffer[ch] = in.read();
				}
				out.write(buffer[ch]);
			}
		}
	}
}


/**
 * \brief Generic clip function
 *
 * The function performs a generic clip operation on the input stream values, used to implement Hardtahn.
 *
 * \tparam Channels   Number of channels - input is equal to output -
 * \tparam IFM_Dim    Width of the input feature map
 * \tparam PE         Number of channels in the pool layer computed in parallel - currently only 1 is supported -
 * \tparam Max        Maximum value
 * \tparam Min        Minimum value
 * \tparam Act_dtype  DataType of the activation
 * \tparam TI         DataType of the input stream - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters
 *
 * \param in          Input stream
 * \param out         Output stream
 * \param reps        Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned Channels, unsigned IFM_Dim, unsigned PE, int Max, int Min,
    typename Act_dtype, typename TI, typename TO>
void ClipStream_Batch(hls::stream<TI> &in, hls::stream<TO> &out, int const reps){
  assert(Channels % PE == 0);
  assert(TI::width == TO::width);
  assert(TI::width == PE * Act_dtype::width);
  for (unsigned r = 0; r < reps * (Channels/PE) * IFM_Dim * IFM_Dim; r++){
#pragma HLS PIPELINE II = 1
    TI in_slice;
    TO out_slice;
    in_slice = in.read();
    for (unsigned pe = 0; pe < PE; pe++){
#pragma HLS UNROLL
      ap_uint<Act_dtype::width> pe_slice = in_slice((pe + 1) * Act_dtype::width - 1, pe * Act_dtype::width);
      Act_dtype casted_slice = *reinterpret_cast<Act_dtype*>(&pe_slice);
      if (casted_slice > (Act_dtype)Max){
        casted_slice = (Act_dtype)Max;
      }
      else if (casted_slice < (Act_dtype)Min){
        casted_slice = (Act_dtype)Min;
      }
      ap_uint<Act_dtype::width> res = *reinterpret_cast<ap_uint<Act_dtype::width>*>(&casted_slice);
      out_slice((pe + 1) * Act_dtype::width - 1, pe * Act_dtype::width) = res;
    }
    out.write(out_slice);
  }
}

/**
 * \brief Concatenate four streams into one
 * 
 * The function reads in four streams and concatenates them into one output stream, pixle by pixel depth wise. used in use Yolo
 * 
 * \tparam Channels   Number of channels - input is equal to output -
 * \tparam IFM_Dim    Width of the input feature map
 * \tparam TI         DataType of the input streams - safely deducible from the paramaters
 * \tparam TO         DataType of the output stream - safely deducible from the paramaters 
 * 
 * \param in1         Input stream I
 * \param in2         Input stream II
 * \param in3         Input stream III
 * \param in4         Input stream IV
 * \param out         Output stream
 * \param reps        Number of frames / images
 *
*/
template <
	unsigned Channels, unsigned IFM_Dim,
	typename TI, typename TO>
void ConcatStreams_Batch(hls::stream<TI> &in1, hls::stream<TI> &in2, hls::stream<TI> &in3, hls::stream<TI> &in4, hls::stream<TO> &out, int const reps){
	assert(TI::width == TO::width);
	TI in_slice;
	for (unsigned r = 0; r < reps * IFM_Dim * IFM_Dim; r++){
		for (unsigned i = 0; i < Channels; i++){
			#pragma HLS PIPELINE II = 1
			in_slice = in1.read();
			out.write(in_slice);
		}
		for (unsigned i = 0; i < Channels; i++){
			#pragma HLS PIPELINE II = 1
			in_slice = in2.read();
			out.write(in_slice);
		}
		for (unsigned i = 0; i < Channels; i++){
			#pragma HLS PIPELINE II = 1
			in_slice = in3.read();
			out.write(in_slice);
		}
		for (unsigned i = 0; i < Channels; i++){
			#pragma HLS PIPELINE II = 1
			in_slice = in4.read();
			out.write(in_slice);
		}
	}
}

/**
 * \brief Concatenate two streams into one
 * 
 * The function reads in two streams and concatenates them into one output stream, pixel by pixel depth-wise.
 * 
 * \tparam Channels   Number of channels - input is equal to output -
 * \tparam IFM_Dim    Width of the input feature map
 * \tparam TI1        DataType of the first input stream
 * \tparam TI2        DataType of the second input stream
 * \tparam TO         DataType of the output stream
 * 
 * \param in1         Input stream I
 * \param in2         Input stream II
 * \param out         Output stream
 * \param reps        Number of frames / images
 *
*/
template <
	unsigned Channels, unsigned IFM_Dim,
	typename TI1, typename TI2, typename TO>
void ConcatStreams_Batch(hls::stream<TI1> &in1, hls::stream<TI2> &in2, hls::stream<TO> &out, int const reps){
	static_assert(TI1::width == TI2::width, "Input stream widths must be the same");
	static_assert(TI1::width <= TO::width, "Output stream width must be greater than or equal to input stream width");
	
	for (unsigned r = 0; r < reps * IFM_Dim * IFM_Dim; r++){
		for (unsigned i = 0; i < Channels; i++){
			#pragma HLS PIPELINE II = 1
			TI1 in1_slice = in1.read();
			TO out_slice = static_cast<TO>(in1_slice);
			out.write(out_slice);
		}
		for (unsigned i = 0; i < Channels; i++){
			#pragma HLS PIPELINE II = 1
			TI2 in2_slice = in2.read();
			TO out_slice = static_cast<TO>(in2_slice);
			out.write(out_slice);
		}
	}
}

#endif
