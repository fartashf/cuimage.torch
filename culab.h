#ifndef CULAB_CUH
#define CULAB_CUH
#include <THC/THC.h>

extern "C" {
    void culab_rgb2lab(THCState *state, THCudaTensor *input, THCudaTensor *output);
}
#endif  /* CULAB_CUH */
