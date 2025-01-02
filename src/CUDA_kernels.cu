#include "../include/CUDA_kernels.h"
#include "CUDA_kernels.h"

__host__ void WInitRNG(curandState *state, int noPoints)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    InitRNG<<<numBlocks, blockSize>>>(state, clock());
}

__host__ void calculateFitnessCUDA(double* d_centroids, double* d_widths, double* d_polyCoef, double *d_points, double *d_expectedOutput, double *d_fitnessResults, std::vector<Eigen::MatrixXd> &allCoefs, double *fitnessResults, std::vector<double> centroids, std::vector<double> &widths, int noParticles, int gaussiansNo, int dim, int noPoints)
{
    runKernel(noPoints, d_points, d_expectedOutput, d_polyCoef, gaussiansNo, d_fitnessResults, d_centroids, d_widths, noParticles, dim);
    cudaMemcpy(fitnessResults, d_fitnessResults, noPoints * sizeof(double), cudaMemcpyDeviceToHost);
}

__global__ void evaluateKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double *centroids, double *widths, int noParticles, int dim)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // printf("noParticles %d, gaussiansNo %d, dimNo %d\n", noParticles, noCoef, dim);
    for (int particleNo = idx; particleNo < noParticles; particleNo += stride)
    {
        double sum = 0.0;
        for (int pointNo = 0; pointNo < noPoints; pointNo++)
        {
            // printf("eval centroid[0]: %f\n", centroids[0]);
            double expectedValue = expectedOutput[pointNo];

            double computedValue = 0.0;
            for (int coefNo = 0; coefNo < noCoef; coefNo++) // gaussy
            {
                double c = polyCoef[particleNo * noCoef + coefNo];

                double resultDim = 0;
                for (int dimNo = 0; dimNo < dim; dimNo++)
                {
                    // printf("d_centroids: %f\n", centroids[particleNo * noCoef * dim + coefNo * dim + dimNo]);
                    double centroid = centroids[particleNo * noCoef * dim + coefNo * dim + dimNo];
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
    }
}

__host__ void runUpdateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* bestPositionWidths, double* inputDomains)
{
    // int blockSize = 256;
    // int numBlocks = (noParticles + blockSize - 1) / blockSize;

    int blockSize = 1;
    int numBlocks = 1;

    updateKernel<<<numBlocks, blockSize>>>(d_centroids, d_widths, state, dim, noPoints, noParticles, gaussianBoundaries, gaussiansNo, bestIndex, centroidChanges, widthChanges, bestPositionCentroids, bestPositionWidths, inputDomains);

    cudaDeviceSynchronize();
}

__host__ void runKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double *centroids, double *widths, int noParticles, int dim)
{
    // int deviceId;
    // int numberOfSMs;

    // cudaGetDevice(&deviceId);
    // cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // int numBlocks = 32 * numberOfSMs;

    // int numBlocks = 1;
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    evaluateKernel<<<numBlocks, blockSize>>>(noPoints, points, expectedOutput, polyCoef, noCoef, fitnessResults, centroids, widths, noParticles, dim);

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
        for (int gaussNo = 0; gaussNo < gaussiansNo; gaussNo++)
        {
            for (int dimNo = 0; dimNo < dim; dimNo++)
            {
                double r1 = curand_uniform_double(&state[particleNo]);
                double r2 = curand_uniform_double(&state[particleNo]);

                // printf("d_centroids: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += c1 * r1 * (d_centroids[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) 
                + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                // printf("Before Centroids changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                if(abs(centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) > ((inputDomains[4 * particleNo + dim * dimNo + 1] - inputDomains[4 * particleNo + dim * dimNo]) * maxChange))
                {
                    if(centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] < 0)
                    {
                        centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = -(inputDomains[4 * particleNo + dim * dimNo + 1] - inputDomains[4 * particleNo + dim * dimNo]) * maxChange;
                    }
                    else
                    {
                        centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = (inputDomains[4 * particleNo + dim * dimNo + 1] - inputDomains[4 * particleNo + dim * dimNo]) * maxChange;
                    }
                }

                // printf("After Centroids changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);
                // d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = 0;
                // printf("Before d_centroids update: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo];
                // printf("upda centroid[0]: %f\n", d_centroids[0]);

                // printf("After d_centroids update: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                // printf("Before Width changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += c1 * r1 * (d_widths[bestIndex * gaussiansNo * dim + gaussNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]) 
                + c2 * r2 * (bestPositionWidths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

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

                // printf("After Width changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo]);

                d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] += widthChanges[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo];   
                // d_widths[particleNo * gaussiansNo * dim + gaussNo * dim + dimNo] = 0;

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