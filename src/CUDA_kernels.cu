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
    std::vector<double> flattenedCoefs;

    size_t coefSize = 0;
    for (auto &coef : allCoefs)
    {
        coefSize += coef.rows() * coef.cols() * sizeof(double);
        flattenedCoefs.insert(flattenedCoefs.end(), coef.data(), coef.data() + coef.rows() * coef.cols());
    }

    cudaMalloc(&d_polyCoef, coefSize);
    cudaMemcpy(d_polyCoef, flattenedCoefs.data(), coefSize, cudaMemcpyHostToDevice);

    runKernel(noPoints, d_points, d_expectedOutput, d_polyCoef, gaussiansNo, d_fitnessResults, d_centroids, d_widths, noParticles, dim);
    cudaMemcpy(fitnessResults, d_fitnessResults, noPoints * sizeof(double), cudaMemcpyDeviceToHost);
}

__host__ void updateParticlesCUDA(curandState *state, int dim, int noPoints, int noParticles, std::vector<double> gaussianBoundaries, int gaussiansNo)
{
    //runUpdateKernel(state, dim, noPoints, noParticles, d_gaussianBoundaries, gaussiansNo);

    //cudaFree(d_gaussianBoundaries);
}

__global__ void evaluateKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double *centroids, double *widths, int noParticles, int dim)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int particleNo = idx; particleNo < noParticles; particleNo += stride)
    {
        double sum = 0.0;
        for (int pointNo = 0; pointNo < noPoints; pointNo++)
        {
            double expectedValue = expectedOutput[pointNo];

            double computedValue = 0.0;
            for (int coefNo = 0; coefNo < noCoef; coefNo++) // gaussy
            {
                double c = polyCoef[particleNo * noCoef + coefNo];

                double resultDim = 0;
                for (int dimNo = 0; dimNo < dim; dimNo++)
                {
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

__host__ void runUpdateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* inputDomains)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    // int blockSize = 1;
    // int numBlocks = 1;

    updateKernel<<<numBlocks, blockSize>>>(d_centroids, d_widths, state, dim, noPoints, noParticles, gaussianBoundaries, gaussiansNo, bestIndex, centroidChanges, widthChanges, bestPositionCentroids, inputDomains);

    cudaDeviceSynchronize();
}

__host__ void runKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double *centroids, double *widths, int noParticles, int dim)
{
    // int deviceId;
    // int numberOfSMs;

    // cudaGetDevice(&deviceId);
    // cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // int numBlocks = 32 * numberOfSMs;

    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    evaluateKernel<<<numBlocks, blockSize>>>(noPoints, points, expectedOutput, polyCoef, noCoef, fitnessResults, centroids, widths, noParticles, dim);
    // updateKernel<<<numBlocks, blockSize>>>(state, dim, noPoints, noParticles);

    cudaDeviceSynchronize();
}

__global__ void InitRNG(curandState *state, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void updateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* inputDomains)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double c1 = 2.f;
    double c2 = 2.f;
    double maxChange = 0.25;

    for (int particleNo = idx; particleNo < noParticles; particleNo += stride)
    {
        for (int gaussNo = 0; gaussNo < gaussiansNo; gaussNo++)
        {
            for (int dimNo = 0; dimNo < dim; dimNo++)
            {
                double r1 = curand_uniform_double(&state[particleNo]);
                double r2 = curand_uniform_double(&state[particleNo]);

                // double centroid = centroids[particleNo * noCoef * dim + coefNo * dim + dimNo];

                printf("r1 r2 %f %f\n", r1, r2);
                printf("d_centr: %f\n", d_centroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]);
                printf("Centroids changes: %f\n", centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]);
                centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] += c1 * r1 * (d_centroids[bestIndex * gaussiansNo + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] - d_centroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]);

                // if(abs(centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) > ((inputDomains[particleNo * gaussiansNo + dimNo + 1] - inputDomains[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) * maxChange))
                // {
                //     if(centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] < 0)
                //     {
                //         centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = -(inputDomains[particleNo * gaussiansNo + dimNo + 1] - inputDomains[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) * maxChange;
                //     }
                //     else 
                //     {
                //         centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = (inputDomains[particleNo * gaussiansNo + dimNo + 1] - inputDomains[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) * maxChange;
                //     }
                // }

                //d_centroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] += centroidChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo];
                d_centroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = 1;

                widthChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] += c1 * r1 * (d_widths[bestIndex * gaussiansNo + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] - d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]);

                // if(abs(widthChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) > ((gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1] - gaussianBoundaries[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) * maxChange))
                // {
                //     if(widthChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] < 0)
                //     {
                //         widthChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = -(gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1] - gaussianBoundaries[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) * maxChange;
                //     }
                //     else
                //     {
                //         widthChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = (gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1] - gaussianBoundaries[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo]) * maxChange;
                //     }
                // }

                d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] += widthChanges[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo];   

                // if(d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] < gaussianBoundaries[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo])
                // {
                //     d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = gaussianBoundaries[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo];
                // }
                // if(d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] > gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1])
                // {
                //     d_widths[particleNo * gaussiansNo * dim + gaussiansNo * dim + dimNo] = gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1];
                // }
            }
        }
    }
}