#include "../include/CUDA_kernels.h"

__host__ void calculateFitnessCUDA(double* d_points, double* d_expectedOutput, double *d_fitnessResults,std::vector<Eigen::MatrixXd> &allCoefs, double *fitnessResults, std::vector<double> centroids, std::vector<double> &widths, int noParticles, int gaussiansNo, int dim, int noPoints)
{
    double *d_polyCoef;

    double *d_centroids;
    double *d_widths;

    //int *d_noCoef;

    //size_t noCoefSize = allCoefs.size() * sizeof(int);
    size_t centroidsSize = centroids.size() * sizeof(double);
    size_t widthsSize = widths.size() * sizeof(double);

    //std::vector<int> noCoef;
    std::vector<double> flattenedCoefs;

    //std::memcpy(flattenedPoints.data(), points.data(), inputSize);

    /*for(int i = 0; i < flattenedPoints.size(); ++i)
    {
        printf("flattenedPoints[%d] = %f\n", i, flattenedPoints[i]);
    }*/

    size_t coefSize = 0;
    for (auto &coef : allCoefs)
    {
        coefSize += coef.rows() * coef.cols() * sizeof(double);
        // printf("coef.rows() = %d\n", coef.rows());
        //noCoef.push_back(coef.rows());

        flattenedCoefs.insert(flattenedCoefs.end(), coef.data(), coef.data() + coef.rows() * coef.cols());
    }

    // printf("%d\n", coefSize); = 8400

    /*
    for (int i = 0; i < noCoef.size(); i++) {
        printf("noCoef[%d] = %d\n", i, noCoef[i]);
    }*/

    //nt *d_gaussiansNo;
    /*for (int i = 0; i < 5; i++)
        printf("centroids[%d] = %f\n", i, centroids[i]);*/

    cudaMalloc(&d_polyCoef, coefSize);
    //cudaMalloc(&d_noCoef, noCoefSize);
    //cudaMalloc(&d_gaussiansNo, sizeof(int));
    cudaMalloc(&d_centroids, centroidsSize);
    cudaMalloc(&d_widths, widthsSize);

    //cudaMemcpy(d_noCoef, noCoef.data(), noCoefSize, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_gaussiansNo, &gaussiansNo, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_polyCoef, flattenedCoefs.data(), coefSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), centroidsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_widths, widths.data(), widthsSize, cudaMemcpyHostToDevice);

    /*size_t offset = 0;
    for (auto& coef : allCoefs)
    {
        cudaMemcpy(d_polyCoef + offset, coef.data(), coef.rows() * coef.cols() * sizeof(double), cudaMemcpyHostToDevice);
        offset += sizeof(coef.data());

        noCoef.push_back(coef.rows());
    }*/

    /*for (int i = 0; i < centroids.size(); i++)
        printf("centroids[%d] = %f\n", i, centroids[i]);*/

    runKernel(noPoints, d_points, d_expectedOutput, d_polyCoef, gaussiansNo, d_fitnessResults, d_centroids, d_widths, noParticles, dim);

    // cudaMemcpy(fitnessResults, d_fitnessResults, fitnessResultsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(fitnessResults, d_fitnessResults, noPoints * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_polyCoef);
    cudaFree(d_fitnessResults);
}

__global__ void evaluateKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double* centroids, double* widths, int noParticles, int dim)
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

__host__ void runKernel(int noPoints, double *points, double *expectedOutput, double *polyCoef, int noCoef, double *fitnessResults, double* centroids, double* widths, int noParticles, int dim)
{
    int deviceId;
    int numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int blockSize = 256;
    int numBlocks = (noPoints + blockSize - 1) / blockSize;

    //int numBlocks = 32 * numberOfSMs;

    evaluateKernel<<<numBlocks, blockSize>>>(noPoints, points, expectedOutput, polyCoef, noCoef, fitnessResults, centroids, widths, noParticles, dim);

    cudaDeviceSynchronize();
}