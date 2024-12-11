#include "../3rdParty/Eigen/Geometry"
#include <curand_kernel.h>
#include <vector>

__host__ void calculateFitnessCUDA(double* d_centroids, double* d_widths, double* d_polyCoef, double *d_points, double *d_expectedOutput, double *d_fitnessResults, std::vector<Eigen::MatrixXd> &allCoefs, double *fitnessResults, std::vector<double> centroids, std::vector<double> &widths, int noParticles, int gaussiansNo, int dim, int noPoints);

__host__ void runUpdateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* inputDomains);
__host__ void runKernel(int noPoints, double* points, double* expectedOutput, double* polyCoef, int noCoef, double* fitnessResults, double* centroids, double* widths, int noParticles, int dim);
__host__ void WInitRNG(curandState* state, int noPoints);
__host__ void updateParticlesCUDA(curandState *state, int dim, int noPoints, int noParticles, std::vector<double> gaussianBoundaries, int gaussiansNo);

__global__ void InitRNG(curandState* state, unsigned long long seed);
__global__ void evaluateKernel(int noPoints, double* points, double* expectedOutput, double* polyCoef, int noCoef, double* fitnessResults, double* centroids, double* widths, int dim);
__global__ void updateKernel(double* d_centroids, double* d_widths, curandState *state, int dim, int noPoints, int noParticles, double *gaussianBoundaries, int gaussiansNo, int bestIndex, double* centroidChanges, double* widthChanges, double* bestPositionCentroids, double* inputDomains);