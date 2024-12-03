#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

void CUDAfunctionTemplate(size_t elementsNo);
void CUDA_QRfactorization(double* Gdata, size_t Grows, size_t Gcols, double* pdata, size_t prows,
                          size_t pcols, double* resultCoeff, size_t coefsNo);