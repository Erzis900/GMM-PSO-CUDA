#include "../3rdParty/Eigen/Geometry"
#include <vector>

__host__ void calculateFitnessCUDA(double* points, double* expectedOutput, double* d_fitnessResults,
                                  std::vector<Eigen::MatrixXd>& allCoefs, double* fitnessResults, std::vector<double> centroids, std::vector<double>& widths, int noParticles, int gaussiansNo, int dim, int noPoints);

__global__ void evaluateKernel(int noPoints, double* points, double* expectedOutput, double* polyCoef, int noCoef, double* fitnessResults, double* centroids, double* widths, int dim);

__host__ void runKernel(int noPoints, double* points, double* expectedOutput, double* polyCoef, int noCoef, double* fitnessResults, double* centroids, double* widths, int noParticles, int dim);