// **********************************************************************************
//
// BSD License.
// This file is part of a sobel convolution implementation.
//
// Copyright (c) 2017, Bruno Keymolen, email: bruno.keymolen@gmail.com
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this
// list of conditions and the following disclaimer in the documentation and/or
// other
// materials provided with the distribution.
// Neither the name of "Bruno Keymolen" nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// **********************************************************************************
#include "sobel.hpp"

#include <cmath>
#include <iostream>

#include <stdlib.h>
#include <string.h>
#include <cstdint>

namespace keymolen {

// clang-format off

    //Sobel
    const int8_t Gx[] = {-1, 0, 1,
                         -2, 0, 2,
                         -1, 0, 1};

    const int8_t Gy[] = { 1, 2, 1,
                          0, 0, 0,
                         -1,-2,-1};

    //Gausian blur
    //3 x 3 kernel
    const int8_t Gaus3x3[] = { 1, 2, 1,
                               2, 4, 2,   // * 1/16  
                               1, 2, 1};
    const int Gaus3x3Div = 16;


	const int8_t Gaus5x5[] = {  2,  4,  5,  4, 2,
                                4,  9, 12,  9, 4,
                                5, 12, 15, 12, 5, // * 1/159
                                4,  9, 12,  9, 4,
                                2,  4,  5,  4, 2 };
    const int Gaus5x5Div = 159;

// clang-format on

Sobel::Sobel(int w, int h) : w_(w), h_(h), size_(w * h) {
    // buffer_ = (unsigned char*)calloc(w_ * h_, sizeof(unsigned char));
    buffer_ = (double*)calloc(w_ * h_, sizeof(double));
    buffer2_ = (double*)calloc(w_ * h_, sizeof(double));
}

Sobel::~Sobel() {
    free(buffer_);
    free(buffer2_);
}

unsigned char* Sobel::edges(unsigned char* dst, const unsigned char* src,
                            Sobel::NoiseFilter kernel_size, bool normalize) {
    int offset_xy = 1;  // for kernel = 3
    int8_t* kernel = (int8_t*)Gaus3x3;
    int kernel_div = Gaus3x3Div;

    if (kernel_size == NoiseFilter::Gaus5x5) {
        offset_xy = 2;
        kernel = (int8_t*)Gaus5x5;
        kernel_div = Gaus5x5Div;
    }

    // gaussian filter
    for (int x = offset_xy; x < w_ - offset_xy; x++) {
        for (int y = offset_xy; y < h_ - offset_xy; y++) {
            int convolve = 0;
            int k = 0;
            int pos = x + (y * w_);
            for (int ky = -offset_xy; ky <= offset_xy; ky++) {
                for (int kx = -offset_xy; kx <= offset_xy; kx++) {
                    convolve += (src[pos + (kx + (ky * w_))] * kernel[k]);
                    k++;
                }
            }

            buffer_[pos] = ((double)convolve / (double)kernel_div);
        }
    }

    // apply sobel kernels
    offset_xy = 1;  // 3x3
    for (int x = offset_xy; x < w_ - offset_xy; x++) {
        for (int y = offset_xy; y < h_ - offset_xy; y++) {
            double convolve_X = 0.0;
            double convolve_Y = 0.0;
            int k = 0;
            int src_pos = x + (y * w_);

            for (int kx = -offset_xy; kx <= offset_xy; kx++) {
                for (int ky = -offset_xy; ky <= offset_xy; ky++) {
                    convolve_X += buffer_[src_pos + (kx + (ky * w_))] * Gx[k];
                    convolve_Y += buffer_[src_pos + (kx + (ky * w_))] * Gy[k];

                    buffer2_[src_pos] = sqrt((convolve_X * convolve_X) +
                                             (convolve_Y * convolve_Y));

                    k++;
                }
            }
        }
    }

    double normalize_factor = 0.236;

    if (normalize) {
        double max = 0.0;
        for (int p = 0; p < size_; p++) {
            if (buffer2_[p] > max) {
                max = buffer2_[p];
            }
        }
        normalize_factor = 255.0 / max;
        std::cout << "max: " << max << " normalize: " << normalize_factor
                  << std::endl;
    }

    for (int x = offset_xy; x < w_ - offset_xy; x++) {
        for (int y = offset_xy; y < h_ - offset_xy; y++) {
            int src_pos = x + (y * w_);
            dst[src_pos] =
                (unsigned char)(buffer2_[src_pos] * normalize_factor);
        }
    }

    return dst;
}
}
