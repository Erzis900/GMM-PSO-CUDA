#include "../include/CUDA_kernels.h"
#include "CUDA_kernels.h"

__host__ void WInitRNG(curandState *state, int noPoints)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    InitRNG<<<numBlocks, blockSize>>>(state, clock());
}

__host__ void calculateFitnessCUDA(double* d_centroids, double* d_widths, double* d_polyCoef, double *d_points, double *d_expectedOutput, double *d_fitnessResults, double *fitnessResults, int noParticles, int gaussiansNo, int dim, int noPoints, double bestFitness, double* bestPositionCentroids, double* bestPositionWidths)
{
    runKernel(noPoints, d_points, d_expectedOutput, d_polyCoef, gaussiansNo, d_fitnessResults, d_centroids, d_widths, noParticles, dim, bestFitness, bestPositionCentroids, bestPositionWidths);
    cudaMemcpy(fitnessResults, d_fitnessResults, noParticles * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void testKernel(double *points, double* centroids)
{
    printf("x[0] = %f\n", points[0]);
    printf("x[1] = %f\n", points[1]);
    printf("x[2] = %f\n", points[2]);
    printf("x[3] = %f\n", points[3]);
    printf("x[4] = %f\n", points[4]);
    printf("x[5] = %f\n", points[5]);

    printf("c[0] = %f\n", centroids[0]);
    printf("c[1] = %f\n", centroids[1]);
    printf("c[2] = %f\n", centroids[2]);
    printf("c[3] = %f\n", centroids[3]);
    printf("c[4] = %f\n", centroids[4]);
    printf("c[5] = %f\n", centroids[5]);
}

__global__ void evaluateKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double *centroids, double *widths, int noParticles, int dim, double bestFitness, double* bestPositionCentroids, double* bestPositionWidths)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // printf("noParticles %d, gaussiansNo %d, dimNo %d\n", noParticles, noCoef, dim);
    for (int particleNo = idx; particleNo < noParticles; particleNo += stride)
    {
        double sum = 0.0;
        for (int pointNo = 0; pointNo < noPoints; pointNo++)
        {
            // printf("point no: %d\n", pointNo);
            // printf("eval centroid[0]: %f\n", centroids[0]);
            double expectedValue = expectedOutput[pointNo];

            double computedValue = 0.0;
            for (int coefNo = 0; coefNo < noCoef; coefNo++) // gaussy
            {
                // printf("coefNo no: %d\n", coefNo);
                double c = polyCoef[particleNo * noCoef + coefNo];

                double resultDim = 0;
                for (int dimNo = 0; dimNo < dim; dimNo++)
                {
                    // printf("eval: noParticles %d, dim %d, noCoef %d\n", noParticles, dim, noCoef);
                    
                    double centroid = centroids[particleNo * noCoef * dim + coefNo * dim + dimNo];
                    if(idx == 0)
                    {
                        // printf("d_centroids: %f\n", centroids[particleNo * noCoef * dim + coefNo * dim + dimNo]);    
                        // printf("centroid: %f\n", centroid);
                    }
                    double width = widths[particleNo * noCoef * dim + coefNo * dim + dimNo];

                    // double centroid = centroids[dim * (particleNo * noCoef + coefNo * dimNo)];
                    // double width = widths[dim * (particleNo * noCoef + coefNo * dimNo)];

                    double x = points[pointNo * dim + dimNo];

                    double gaussianValue = (-width * pow((x - centroid), 2.0));

                    resultDim += gaussianValue;
                }

                resultDim /= double(dim);

                computedValue += c * exp(resultDim);
            }
            sum += fabs(expectedValue - computedValue);
        }
        fitnessResults[particleNo] = sum / noPoints;

        if (sum < bestFitness)
        {
            // printf("bestFitness: %f, sum: %f\n", bestFitness, sum / noPoints);
            // bestPositionCentroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo];
            // bestPositionWidths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo];
        }
    }
}

__host__ void runUpdateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* bestPositionWidths, double* inputDomains)
{
    int blockSize = 256;
    int numBlocks = (noParticles + blockSize - 1) / blockSize;

    // int blockSize = 1;
    // int numBlocks = 1;

    updateKernel<<<numBlocks, blockSize>>>(d_centroids, d_widths, state, dim, noPoints, noParticles, gaussianBoundaries, gaussiansNo, bestIndex, centroidChanges, widthChanges, bestPositionCentroids, bestPositionWidths, inputDomains);

    cudaDeviceSynchronize();
}

__host__ void runTestKernel(double* points, double* centroids)
{
    int blockSize = 1;
    int numBlocks = 1;

    testKernel<<<numBlocks, blockSize>>>(points, centroids);

    cudaDeviceSynchronize();
}

__host__ void runKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double *centroids, double *widths, int noParticles, int dim, double bestFitness, double* bestPositionCentroids, double* bestPositionWidths)
{
    // int deviceId;
    // int numberOfSMs;

    // cudaGetDevice(&deviceId);
    // cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // int numBlocks = 32 * numberOfSMs;

    // int numBlocks = 1;
    int blockSize = 256;
    int numBlocks = (noParticles + blockSize - 1) / blockSize;

    evaluateKernel<<<numBlocks, blockSize>>>(noPoints, points, expectedOutput, polyCoef, noCoef, fitnessResults, centroids, widths, noParticles, dim, bestFitness, bestPositionCentroids, bestPositionWidths);

    cudaDeviceSynchronize();
}

__global__ void InitRNG(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void updateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* bestPositionWidths, double* inputDomains)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double c1 = 2.f;
    double c2 = 2.f;
    double maxChange = 0.25;

    // printf("noParticles %d, gaussiansNo %d, dimNo %d\n", noParticles, gaussiansNo, dim);

    for (int particleNo = idx; particleNo < noParticles; particleNo += stride)
    {
        if(particleNo != bestIndex)
        {
            // printf("particleNo %d %d\n", particleNo, bestIndex);
            for (int gaussNo = 0; gaussNo < gaussiansNo; gaussNo++)
            {
                for (int dimNo = 0; dimNo < dim; dimNo++)
                {
                    double r1 = curand_uniform_double(&state[particleNo]);
                    double r2 = curand_uniform_double(&state[particleNo]);
                    // printf("coef 1 CUDA: %f\n", d_centroids[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    // printf("bpc CUDA: %f\n", bestPositionCentroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    // printf("d_centroids: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += c1 * r1 * (d_centroids[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) 
                    + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    // centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += c1 * (d_centroids[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) 
                    // + c2 * (bestPositionCentroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    //  printf("%d Centroid change CUDA: %f\n", gaussNo, centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    // if(idx == 0)
                        // printf("%d Centroid change CUDA: %f\n", particleNo, centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                        // printf("%d inputDomains CUDA %f %f\n", particleNo, inputDomains[4 * particleNo + dim * dimNo], inputDomains[4 * particleNo + dim * dimNo + 1]);

                    if(abs(centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) > ((inputDomains[dim * dimNo + 1] - inputDomains[dim * dimNo]) * maxChange))
                    {
                        if(centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] < 0)
                        {
                            centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = -(inputDomains[dim * dimNo + 1] - inputDomains[dim * dimNo]) * maxChange;
                        }
                        else
                        {
                            centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = (inputDomains[dim * dimNo + 1] - inputDomains[dim * dimNo]) * maxChange;
                        }
                    }

                    // if(idx == 0)
                    //     printf("%d Centroid change CUDA: %f\n", particleNo, centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    // // // printf("After Centroids changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    // // printf("Before d_centroids update: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo];
                    // // printf("upda centroid[0]: %f\n", d_centroids[0]);

                    // printf("After d_centroids update: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    // // printf("Before Width changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += c1 * r1 * (d_widths[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) 
                    + c2 * r2 * (bestPositionWidths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    // widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += c1 * (d_widths[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) 
                    // + c2 * (bestPositionWidths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                    
                    // printf("%d Width change CUDA: %f\n", gaussNo, widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    // if(idx == 0)
                        // printf("%d gaussianBoundaries CUDA %f %f\n", particleNo, gaussianBoundaries[4 * particleNo + dim * dimNo], gaussianBoundaries[4 * particleNo + dim * dimNo + 1]);

                    if(abs(widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) > ((gaussianBoundaries[4 * particleNo + dim * dimNo + 1] - gaussianBoundaries[4 * particleNo + dim * dimNo]) * maxChange))
                    {
                        if(widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] < 0)
                        {
                            widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = -(gaussianBoundaries[4 * particleNo + dim * dimNo + 1] - gaussianBoundaries[4 * particleNo + dim * dimNo]) * maxChange;
                        }
                        else
                        {
                            widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = (gaussianBoundaries[4 * particleNo + dim * dimNo + 1] - gaussianBoundaries[4 * particleNo + dim * dimNo]) * maxChange;
                        }
                    }

                    // // printf("After Width changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                    d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo];   
                    // // // d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = 0;

                    if(d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] < gaussianBoundaries[4 * particleNo + dim * dimNo])
                    {
                        d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = gaussianBoundaries[4 * particleNo + dim * dimNo];
                    }
                    if(d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] > gaussianBoundaries[4 * particleNo + dim * dimNo + 1])
                    {
                        d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = gaussianBoundaries[4 * particleNo + dim * dimNo + 1];
                    }
                }
            }
        }
    }
}
