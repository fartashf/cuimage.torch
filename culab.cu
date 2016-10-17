#include "culab.h"
#include "common.h"

/*
 * Convert an sRGB color channel to a linear sRGB color channel.
 */
template <typename Dtype>
__host__ __device__ static inline Dtype culab_gamma_expand_sRGB(Dtype nonlinear)
{
  return (nonlinear <= 0.04045) ? (nonlinear / 12.92)
                                : (pow((nonlinear+0.055)/1.055, 2.4));
}

/*
 * Convert a linear sRGB color channel to a sRGB color channel.
 */
template <typename Dtype>
__host__ __device__ static inline Dtype culab_gamma_compress_sRGB(Dtype linear)
{
  return (linear <= 0.0031308) ? (12.92 * linear)
                               : (1.055 * pow(linear, 1.0/2.4) - 0.055);
}

template <typename Dtype>
__global__ void rgb2lab(const int nthreads, const Dtype* rgb, const int num,
        const int channels, const int rows, const int cols, Dtype* lab){
    // CIE Standard
    double epsilon = 216.0/24389.0;
    double k = 24389.0/27.0;
    // D65 white point
    double xn = 0.950456;
    double zn = 1.088754;

    CUDA_KERNEL_LOOP(index, nthreads) {
        Dtype r,g,b,l,a,_b;
        int x = index % cols;
        int y = (index / cols) % rows;
        int n = (index / cols) / rows;
        int offset = (n*channels*rows + y)*cols + x;
        int step = rows*cols;
        // get RGB
        // r = culab_gamma_expand_sRGB(rgb[offset + 0*step]);
        // g = culab_gamma_expand_sRGB(rgb[offset + 1*step]);
        // b = culab_gamma_expand_sRGB(rgb[offset + 2*step]);
        r = rgb[offset + 0*step];
        g = rgb[offset + 1*step];
        b = rgb[offset + 2*step];

        // sRGB to XYZ
        double X = 0.412453 * r + 0.357580 * g + 0.180423 * b;
        double Y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
        double Z = 0.019334 * r + 0.119193 * g + 0.950227 * b;

        // normalize for D65 white point
        X /= xn;
        Z /= zn;

        // XYZ normalized to CIE Lab
        double fx = X > epsilon ? pow(X, 1/3.0) : (k * X + 16)/116;
        double fy = Y > epsilon ? pow(Y, 1/3.0) : (k * Y + 16)/116;
        double fz = Z > epsilon ? pow(Z, 1/3.0) : (k * Z + 16)/116;
        l = 116 * fy - 16;
        a = 500 * (fx - fy);
        _b = 200 * (fy - fz);

        // set lab
        lab[offset + 0*step] = l;
        lab[offset + 1*step] = a;
        lab[offset + 2*step] = _b;
    }
}

/*
 * Converts an sRGB color value to LAB.
 * Based on http://www.brucelindbloom.com/index.html?Equations.html.
 * Assumes r, g, and b are contained in the set [0, 1].
 * LAB output is NOT restricted to [0, 1]!
 */
void culab_rgb2lab(THCState *state, THCudaTensor *input, THCudaTensor *output) {
    THCUNN_assertSameGPU(state, 2, input, output);
    THArgCheck(input->nDimension == 3 || input->nDimension == 4, 2,
            "3D or 4D (batch) tensor expected");

    long nInputCols, nInputRows, nInputPlane, batchSize;
    if (input->nDimension == 3) {
        nInputCols = input->size[2];
        nInputRows = input->size[1];
        nInputPlane = input->size[0];
        batchSize = 1;
    }
    else
    {
        nInputCols = input->size[3];
        nInputRows = input->size[2];
        nInputPlane = input->size[1];
        batchSize = input->size[0];
    }
    if(nInputPlane != 3)
        THError("Given input size: (%dx%dx%d). Expected 3 input channels",
                nInputPlane,nInputRows,nInputCols);
    THCudaTensor_resize4d(state, output, batchSize, nInputPlane, nInputRows,
            nInputCols);
    input = THCudaTensor_newContiguous(state, input);
    float* input_data = THCudaTensor_data(state, input);
    float* output_data = THCudaTensor_data(state, output);
    int count = batchSize*nInputRows*nInputCols;
    rgb2lab <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
            THCState_getCurrentStream(state) >>>
                (count, input_data, batchSize,
                 nInputPlane, nInputRows, nInputCols,
                 output_data);
    THCudaCheck(cudaGetLastError());

    if(input->nDimension == 3)
        THCudaTensor_resize3d(state, output, nInputPlane, nInputRows,
                nInputCols);

    THCudaTensor_free(state, input);
}
