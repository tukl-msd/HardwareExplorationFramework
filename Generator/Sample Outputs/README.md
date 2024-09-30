#Sample output folder
This folder contains 4 sample outputs from the generator for the U-Net model
- **2SF4**
   * This model has 2 **S**tages of feature extraction and with 1/4 the number of **F**ilters in the original U-Net model
- **3SF4**
    * This model has 3 **S**tages of feature extraction and with 1/4 the number of **F**ilters in the original U-Net model
- **3SF2**
    * This model has 3 **S**tages of feature extraction and with 1/2 the number of **F**ilters in the original U-Net model
- **4SF4**
    * This model has 4 **S**tages of feature extraction and with 1/4 the number of **F**ilters in the original U-Net model

Each stage consists of an encoder and a decoder.

The encoder consists of the following sequence:
- Convolution (conv)
- Batch Normalization (bn)
- ReLU activation (relu)
- Convolution (conv)
- Batch Normalization (bn)
- ReLU activation (relu)

The output of the last ReLU activation is copied. One copy goes to a 2x2 Max Pooling layer and then to the next encoder. The other copy goes via a skip connection to the corresponding decoder in the same stage.

The decoder on the other side consists of the following sequence:
- Transposed Convolution (conv transpose)
- ReLU activation (relu)
- Concatenation (concat) with the output from the encoder coming from the skip connection
- Convolution (conv)
- Batch Normalization (bn)
- ReLU activation (relu)
- Convolution (conv)
- Batch Normalization (bn)
- ReLU activation (relu)