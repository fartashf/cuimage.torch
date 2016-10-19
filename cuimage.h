#ifndef CUIMAGE_CUH
#define CUIMAGE_CUH
#include <THC/THC.h>

extern "C" {
    void cuimage_rgb2lab(THCState *state, THCudaTensor *input,
            THCudaTensor *output, const bool do_gamma_expand);
    void cuimage_scaleBilinear(THCState *state, THCudaTensor *input,
            THCudaTensor *output, THCudaTensor *tmp,
            const int nOutputRows, const int nOutputCols);
}
#endif  /* CUIMAGE_CUH */
