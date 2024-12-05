#include "../include/CUDA_kernels.h"
#include "CUDA_kernels.h"

double *d_centroids;
double *d_widths;
double *d_polyCoef;

__host__ void WInitRNG(curandState *state, int noPoints)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    InitRNG<<<numBlocks, blockSize>>>(state, clock());
}

__host__ void calculateFitnessCUDA(double *d_points, double *d_expectedOutput, double *d_fitnessResults, std::vector<Eigen::MatrixXd> &allCoefs, double *fitnessResults, std::vector<double> centroids, std::vector<double> &widths, int noParticles, int gaussiansNo, int dim, int noPoints)
{
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

__host__ void runUpdateKernel(curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* inputDomains)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

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

                // centroidChanges[particleNo * gaussiansNo + dimNo] += c1 * r1 * (d_centroids[bestIndex * gaussiansNo + dimNo] - d_centroids[particleNo * gaussiansNo + dimNo]) + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo + dimNo] - d_centroids[particleNo * gaussiansNo + dimNo]);
                centroidChanges[particleNo * gaussiansNo + dimNo] += c1 * r1 * (d_centroids[bestIndex * gaussiansNo + dimNo] - d_centroids[particleNo * gaussiansNo + dimNo]) + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo + dimNo] - d_centroids[particleNo * gaussiansNo + dimNo]);

                if(abs(centroidChanges[particleNo * gaussiansNo + dimNo]) > ((inputDomains[particleNo * gaussiansNo + dimNo + 1] - inputDomains[particleNo * gaussiansNo + dimNo]) * maxChange))
                {
                    if(centroidChanges[particleNo * gaussiansNo + dimNo] < 0)
                    {
                        centroidChanges[particleNo * gaussiansNo + dimNo] = -(inputDomains[particleNo * gaussiansNo + dimNo + 1] - inputDomains[particleNo * gaussiansNo + dimNo]) * maxChange;
                    }
                    else 
                    {
                        centroidChanges[particleNo * gaussiansNo + dimNo] = (inputDomains[particleNo * gaussiansNo + dimNo + 1] - inputDomains[particleNo * gaussiansNo + dimNo]) * maxChange;
                    }
                }

                d_centroids[particleNo * gaussiansNo + dimNo] += centroidChanges[particleNo * gaussiansNo + dimNo];

                widthChanges[particleNo * gaussiansNo + dimNo] += c1 * r1 * (d_widths[bestIndex * gaussiansNo + dimNo] - d_widths[particleNo * gaussiansNo + dimNo]) + c2 * r2 * (bestPositionCentroids[particleNo * gaussiansNo + dimNo] - d_widths[particleNo * gaussiansNo + dimNo]);

                if(abs(widthChanges[particleNo * gaussiansNo + dimNo]) > ((gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1] - gaussianBoundaries[particleNo * gaussiansNo + dimNo]) * maxChange))
                {
                    if(widthChanges[particleNo * gaussiansNo + dimNo] < 0)
                    {
                        widthChanges[particleNo * gaussiansNo + dimNo] = -(gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1] - gaussianBoundaries[particleNo * gaussiansNo + dimNo]) * maxChange;
                    }
                    else
                    {
                        widthChanges[particleNo * gaussiansNo + dimNo] = (gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1] - gaussianBoundaries[particleNo * gaussiansNo + dimNo]) * maxChange;
                    }
                }

                d_widths[particleNo * gaussiansNo + dimNo] += widthChanges[particleNo * gaussiansNo + dimNo];   

                if(d_widths[particleNo * gaussiansNo + dimNo] < gaussianBoundaries[particleNo * gaussiansNo + dimNo])
                {
                    d_widths[particleNo * gaussiansNo + dimNo] = gaussianBoundaries[particleNo * gaussiansNo + dimNo];
                }
                if(d_widths[particleNo * gaussiansNo + dimNo] > gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1])
                {
                    d_widths[particleNo * gaussiansNo + dimNo] = gaussianBoundaries[particleNo * gaussiansNo + dimNo + 1];
                }
            }
        }

        /*std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < gaussiansNo; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                chromosome[i].setCentroidChange(j, chromosome[i].getCentroidChange(j) + c1 * distribution(generator) * (bestParticle.chromosome[i].getCentroid(j) - chromosome[i].getCentroid(j)) + c2 * distribution(generator) * (bestPosition[i].getCentroid(j) - chromosome[i].getCentroid(j)));
                if (abs(chromosome[i].getCentroidChange(j)) > ((inputDomain[j].second - inputDomain[j].first) * maxChange))
                {
                    if (chromosome[i].getCentroidChange(j) < 0)
                        chromosome[i].setCentroidChange(j, -(inputDomain[j].second - inputDomain[j].first) * maxChange);
                    else
                        chromosome[i].setCentroidChange(j, (inputDomain[j].second - inputDomain[j].first) * maxChange);
                }
                chromosome[i].setCentroid(j, chromosome[i].getCentroid(j) + chromosome[i].getCentroidChange(j));

                chromosome[i].setWidthChange(j, chromosome[i].getWidthChange(j) + c1 * distribution(generator) * (bestParticle.chromosome[i].getWidth(j) - chromosome[i].getWidth(j)) + c2 * distribution(generator) * (bestPosition[i].getWidth(j) - chromosome[i].getWidth(j)));
                if (abs(chromosome[i].getWidthChange(j)) > (gaussianBoundaries[j].second - gaussianBoundaries[j].first) * maxChange)
                {
                    if (chromosome[i].getWidthChange(j) < 0)
                        chromosome[i].setWidthChange(j, -(gaussianBoundaries[j].second - gaussianBoundaries[j].first) * maxChange);
                    else
                        chromosome[i].setWidthChange(j, (gaussianBoundaries[j].second - gaussianBoundaries[j].first) * maxChange);
                }
                chromosome[i].setWidth(j, chromosome[i].getWidth(j) + chromosome[i].getWidthChange(j));
                if (chromosome[i].getWidth(j) < (gaussianBoundaries[j].first))
                    chromosome[i].setWidth(j, gaussianBoundaries[j].first);
                if (chromosome[i].getWidth(j) > (gaussianBoundaries[j].second))
                    chromosome[i].setWidth(j, gaussianBoundaries[j].second);
            }
        }*/
    }
    //d_centroids = centroids;
}

__host__ void fitnessAgain(double *d_points, double *d_expectedOutput, double *d_fitnessResults, std::vector<Eigen::MatrixXd> &allCoefs, double *fitnessResults, int noParticles, int gaussiansNo, int dim, int noPoints)
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

    runFitnessKernelAgain(noPoints, d_points, d_expectedOutput, d_polyCoef, gaussiansNo, d_fitnessResults, noParticles, dim);
    cudaMemcpy(fitnessResults, d_fitnessResults, noPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_polyCoef);
    cudaFree(d_fitnessResults);
}

__host__ void runFitnessKernelAgain(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, int noParticles, int dim)
{
    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    evaluateKernel<<<numBlocks, blockSize>>>(noPoints, points, expectedOutput, polyCoef, noCoef, fitnessResults, d_centroids, d_widths, noParticles, dim);
    // updateKernel<<<numBlocks, blockSize>>>(state, dim, noPoints, noParticles);

    cudaDeviceSynchronize();
}
