#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    // since the original image size is not changed
    // we need to consider zero-paddings here
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    // im[channel, row, col]
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    // perverse the original dimension
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        // [c, h, w]
        // c = c * ksize * ksize + h * ksize + w
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        // input image channel
        int c_im = c / ksize / ksize;
        // for each pixel in the output matrix
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                // input pixel position
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                // output matrix index
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

