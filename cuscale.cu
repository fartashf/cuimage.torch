/* from https://github.com/torch/image/blob/master/generic/image.c */
#include "cuimage.h"
#include "common.h"

template <typename Dtype>
__global__ void scaleLinear_rowcol(const int nthreads, const Dtype* src,
        Dtype* dst,
        long src_stride,
        long dst_stride,
        long src_len,
        long dst_len,
        long src_stridej,
        long dst_stridej,
        long src_stridek,
        long dst_stridek,
        long dst_len2) {

    CUDA_KERNEL_LOOP(index, nthreads){
        long di = index % dst_len;
        long j = (index / dst_len ) % dst_len2;
        long k = (index / dst_len ) / dst_len2;
        long src_start = j*src_stridej+k*src_stridek;
        long dst_start = j*dst_stridej+k*dst_stridek;
        if ( dst_len > src_len ){
            float si_f;
            long si_i;
            float scale = (float)(src_len - 1) / (dst_len - 1);

            if (di == dst_len - 1 ) {
                dst[ dst_start + (dst_len - 1) * dst_stride ] =
                    src[ src_start + (src_len - 1) * src_stride ];
            }else if ( src_len == 1 ) {
                long dst_pos = dst_start + di*dst_stride;
                dst[dst_pos] = src[ src_start ];
            } else {
                long dst_pos = dst_start + di*dst_stride;
                si_f = di * scale; si_i = (long)si_f; si_f -= si_i;

                dst[dst_pos] = 
                        (1 - si_f) * src[ src_start + si_i * src_stride ] +
                        si_f * src[ src_start + (si_i + 1) * src_stride ];
            }
        } else if ( dst_len < src_len ) {
            long si0_i = 0; float si0_f = 0;
            long si1_i; float si1_f;
            long si;
            float scale = (float)src_len / dst_len;
            float acc, n;
            si1_f = (di + 1) * scale; si1_i = (long)si1_f; si1_f -= si1_i;
            si0_f = di*scale; si0_i = (long)si0_f; si0_f -= si0_i;
            acc = (1 - si0_f) * src[ src_start + si0_i * src_stride ];
            n = 1 - si0_f;
            for( si = si0_i + 1; si < si1_i; si++ )
            {
                acc += src[ src_start + si * src_stride ];
                n += 1;
            }
            if( si1_i < src_len )
            {
                acc += si1_f * src[ src_start + si1_i*src_stride ];
                n += si1_f;
            }
            dst[ dst_start + di*dst_stride ] = acc / n;
        } else {
            dst[ dst_start + di*dst_stride ] = src[ src_start + di*src_stride ];
        }
    }

}

void cuimage_scaleBilinear(THCState *state, THCudaTensor *input,
        THCudaTensor *output, THCudaTensor *tmp,
        const int nOutputRows, const int nOutputCols) {
    THCUNN_assertSameGPU(state, 2, input, output);
    THArgCheck(nOutputRows > 0 && nOutputCols > 0, 2, "oheight <=0 || owidth <= 0");
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
    THCudaTensor_resize4d(state, output, batchSize, nInputPlane,
            nOutputRows, nOutputCols);
    THCudaTensor_resize4d(state, tmp, batchSize, nInputPlane,
            nInputRows, nOutputCols);
    input = THCudaTensor_newContiguous(state, input);
    float* input_data = THCudaTensor_data(state, input);
    float* output_data = THCudaTensor_data(state, output);
    float* tmp_data = THCudaTensor_data(state, tmp);

    /* compress/expand rows first */
    int count = THCudaTensor_nElement(state, tmp);
    scaleLinear_rowcol <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
                       THCState_getCurrentStream(state) >>>
                           (count, input_data, tmp_data,
                            1, 1, nInputCols, nOutputCols,
                            nInputCols, nOutputCols,
                            nInputRows*nInputCols, nInputRows*nOutputCols,
                            nInputRows);
    THCudaCheck(cudaGetLastError());

    /* then columns */
    count = THCudaTensor_nElement(state, output);
    scaleLinear_rowcol <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0,
                       THCState_getCurrentStream(state) >>>
                           (count, tmp_data, output_data,
                            nOutputCols, nOutputCols, nInputRows, nOutputRows,
                            1, 1,
                            nInputRows*nOutputCols, nOutputRows*nOutputCols,
                            nOutputCols);
    THCudaCheck(cudaGetLastError());

    if(input->nDimension == 3)
        THCudaTensor_resize3d(state, output, nInputPlane, nOutputRows,
                nOutputCols);

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, tmp);
}
