#include "../include/CUDA_kernels.h"
#include "CUDA_kernels.h"

__host__ void WInitRNG(curandState *state, int noPoints)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    InitRNG<<<numBlocks, blockSize>>>(state, clock());
}

__host__ void calculateFitnessCUDA(double *d_points, double *d_expectedOutput, double *d_fitnessResults, std::vector<Eigen::MatrixXd> &allCoefs, double *fitnessResults, std::vector<double> centroids, std::vector<double> &widths, int noParticles, int gaussiansNo, int dim, int noPoints)
{
    double *d_polyCoef;
    double *d_centroids;
    double *d_widths;

    size_t centroidsSize = centroids.size() * sizeof(double);
    size_t widthsSize = widths.size() * sizeof(double);

    std::vector<double> flattenedCoefs;

    size_t coefSize = 0;
    for (auto &coef : allCoefs)
    {
        coefSize += coef.rows() * coef.cols() * sizeof(double);
        flattenedCoefs.insert(flattenedCoefs.end(), coef.data(), coef.data() + coef.rows() * coef.cols());
    }

    cudaMalloc(&d_polyCoef, coefSize);
    cudaMalloc(&d_centroids, centroidsSize);
    cudaMalloc(&d_widths, widthsSize);

    cudaMemcpy(d_polyCoef, flattenedCoefs.data(), coefSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), centroidsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_widths, widths.data(), widthsSize, cudaMemcpyHostToDevice);

    runKernel(noPoints, d_points, d_expectedOutput, d_polyCoef, gaussiansNo, d_fitnessResults, d_centroids, d_widths, noParticles, dim);
    cudaMemcpy(fitnessResults, d_fitnessResults, noPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_polyCoef);
    cudaFree(d_fitnessResults);
}

__host__ void updateParticlesCUDA(curandState *state, int dim, int noPoints, int noParticles, std::vector<double> bestGaussianBoundaries, std::vector<double> pbestGaussianBoundaries, std::vector<double> gaussianBoundaries, std::vector<double> gaussianBoundariesChange)
{
    double* d_bestGaussianBoudaries;
    double* d_pbestGaussianBoudaries;
    double* d_gaussianBoundaries;
    double* d_gaussianBoundariesChange;

    size_t bestGaussianBoudariesSize = bestGaussianBoundaries.size() * sizeof(double);
    size_t pbestGaussianBoudariesSize = pbestGaussianBoundaries.size() * sizeof(double);
    size_t gaussianBoundariesSize = gaussianBoundaries.size() * sizeof(double);
    size_t gaussianBoundariesChangeSize = gaussianBoundariesChange.size() * sizeof(double);

    cudaMalloc(&d_bestGaussianBoudaries, bestGaussianBoudariesSize);
    cudaMalloc(&d_pbestGaussianBoudaries, pbestGaussianBoudariesSize);
    cudaMalloc(&d_gaussianBoundaries, gaussianBoundariesSize);
    cudaMalloc(&d_gaussianBoundariesChange, gaussianBoundariesChangeSize);

    cudaMemcpy(d_bestGaussianBoudaries, bestGaussianBoundaries.data(), bestGaussianBoudariesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pbestGaussianBoudaries, pbestGaussianBoundaries.data(), pbestGaussianBoudariesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussianBoundaries, gaussianBoundaries.data(), gaussianBoundariesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussianBoundariesChange, gaussianBoundariesChange.data(), gaussianBoundariesChangeSize, cudaMemcpyHostToDevice);

    runUpdateKernel(state, dim, noPoints, noParticles, d_bestGaussianBoudaries, d_pbestGaussianBoudaries, d_gaussianBoundaries, d_gaussianBoundariesChange);

    cudaFree(d_bestGaussianBoudaries);
    cudaFree(d_pbestGaussianBoudaries);
    cudaFree(d_gaussianBoundaries);
    cudaFree(d_gaussianBoundariesChange);
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
            for (int coefNo = 0; coefNo < noCoef; coefNo++)
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

__host__ void runUpdateKernel(curandState *state, int dim, int noPoints, int noParticles, double *bestGaussianBoundaries, double *pbestGaussianBoundaries, double *gaussianBoundaries, double *gaussianBoundariesChange)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    updateKernel<<<numBlocks, blockSize>>>(state, dim, noPoints, noParticles, bestGaussianBoundaries, pbestGaussianBoundaries, gaussianBoundaries, gaussianBoundariesChange);

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

__global__ void updateKernel(curandState *state, int dim, int noPoints, int noParticles, double *bestGaussianBoundaries, double *pbestGaussianBoundaries, double *gaussianBoundaries, double *gaussianBoundariesChange)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    double c1 = 2.f;
    double c2 = 2.f;
    double maxChange = 0.25;

    for (int particleNo = idx; particleNo < noParticles; particleNo += stride)
    {
        for (int dimNo = 0; dimNo < dim; dimNo++)
        {
            double r1 = curand_uniform_double(&state[particleNo]);
            double r2 = curand_uniform_double(&state[particleNo]);

            // j = particleNo * dim + dimNo

            gaussianBoundariesChange[particleNo * dim + dimNo] +=
            (c1 * r1 * (bestGaussianBoundaries[1] - gaussianBoundaries[particleNo * dim + 1])) 
            + (c2 * r2 * (pbestGaussianBoundaries[particleNo * dim + dimNo] - gaussianBoundaries[particleNo * dim + 1]));

            if (std::isinf(gaussianBoundariesChange[particleNo * dim + dimNo]))
                gaussianBoundariesChange[particleNo * dim + dimNo] = 0;
            gaussianBoundaries[particleNo * dim + 1] += gaussianBoundariesChange[particleNo * dim + dimNo];
            if (std::isinf(gaussianBoundaries[particleNo * dim + 1]))
                gaussianBoundaries[particleNo * dim + 1] = std::numeric_limits<double>::max();
            if (gaussianBoundaries[particleNo * dim + 1] < 0) // reset
                gaussianBoundaries[particleNo * dim + 1] = bestParticle.gaussianBoundaries[j].second;
            }
    }
}