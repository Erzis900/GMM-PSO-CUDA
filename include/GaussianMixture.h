//
//
//  Generated by StarUML(tm) C++ Add-In
//
//  @ Project : GaussianMixture library Untitled
//  @ File Name : GaussianMixture.h
//  @ Date : 2009-07-16
//  @ Author : Dominik Belter
//
//

#ifndef _GaussianMixture_H
#define _GaussianMixture_H

#include "../3rdParty/tinyXML/tinyxml2.h"
#include "../include/population.h"
#include "../include/individual.h"
#include "../include/approximation.h"
#include "recorder.h"
#include <memory>
#include <iostream>

#include "../include/CUDA_kernels.h"

using namespace approximator;

class GaussianMixture : public Approximation
{
public:
    /// Pointer
    typedef std::unique_ptr<GaussianMixture> Ptr;

    class Config
    {
    public:
        Config() {}
        Config(std::string configFilename);

    public:
        // input - number of features
        int inputsNo;

        // output - number of outputs
        int outputsNo;

        // PSO population size
        int populationSize;

        // Gaussians no
        int gaussiansNo;

        // max iterations
        int maxIterations;

        // length of the training vector
        int trainSize;

        // length of the testing vector
        int testSize;

        // output format
        int outputType;

        /// normalize output
        bool normalizeOutput;

        // boundaries for Gaussian width [min, max]
        std::vector<std::pair<double, double>> boundaries;

        // train dataset filename
        std::string trainFilename;

        // test dataset filename
        std::string testFilename;

        // verification dataset filename
        std::string verifFilename;
    };

    /// constructor
    GaussianMixture() = delete;
    /// constructor
    GaussianMixture(GaussianMixture::Config _config);
    /// destructor
    ~GaussianMixture();
    /// search for the best GaussianMixture function - PSO method
    void train();
    void trainCPU();

    float getMemcpyTime() { return memcpyTime; }

    std::vector<Eigen::MatrixXd> allCoefs;
    std::vector<double> centroids;
    std::vector<double> widths;

    std::vector<double> gaussianBoundariesV;
    std::vector<double> centroidChanges;
    std::vector<double> widthChanges;
    std::vector<double> bestPositionCentroids;
    std::vector<double> bestPositionWidths;
    std::vector<double> inputDomains;
private:
    double* d_centroidChanges;
    double* d_widthChanges;
    double* d_bestPositionCentroids;
    double* d_bestPositionWidths;
    double* d_gaussianBoundaries;
    double* d_inputDomains;

    double *d_points;
    double *d_expectedOutput;
    double *d_fitnessResults;

    double *d_centroids;
    double *d_widths;
    double *d_polyCoef;

    double *d_pBest;

    std::vector<double> flattenedCoefs;

    size_t centroidChangesSize;
    size_t widthChangesSize;
    size_t bestPositionCentroidsSize;
    size_t bestPositionWidthsSize;
    size_t gaussianBoundariesSize;
    size_t inputDomainsSize;

    size_t inputSize;
    size_t outputSize;
    size_t fitnessResultsSize;

    size_t centroidsSize;
    size_t widthsSize;

    curandState* state;
    std::vector<double> fitnessCPU;

    std::vector<float> fitnessTimesCPU;
    std::vector<float> fitnessTimesGPU;

    std::vector<float> updateTimesCPU;
    std::vector<float> updateTimesGPU;

    float memcpyTime;

    /// read train/test/verification data
    void readInOutData(const std::string &filename, Eigen::MatrixXd &_input, Eigen::MatrixXd &_output, int &vecLength);
    /// read train/test/verification data
    void readInOutSingleRow(const std::string &filename, Eigen::MatrixXd &_input, Eigen::MatrixXd &_output, int &vecLength);
    /// compute Average Percent Error
    double computeAPE(const Eigen::MatrixXd &input, const Eigen::MatrixXd &output, double &avError, double &maxError);
    /// set boundaries of function's parameters -run first
    void setBoundaries();
    /// compute n boundaries (width of Gaussian function)
    void computeBoundaries(std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain);
    /// sets parameters of polynomial and creates population for EA - run second
    void setParameters();
    /// read training data
    void readTrainingData(const std::string &filename);
    /// read verification data
    void readVerificationData(const std::string &filename);
    /// read test data
    void readTestData(const std::string &filename);
    /// initialize matrices
    void initializeMatrices();
    /// compute output for the best polynomial
    double computeOutput(const Eigen::MatrixXd &input, int outNo) const;
    /// compute coefficients
    bool computeCoef(unsigned int individualNo, Eigen::MatrixXd &result_coef);
    /// compute individual fitness
    double computeIndividualFitness(const Eigen::MatrixXd &_input, const Eigen::MatrixXd &_output, Eigen::MatrixXd &_coef, unsigned int individualNo);
    /// save plot file (test of GaussianMixture function)
    void savePlotFile(const std::string &filename, const Eigen::MatrixXd &input, const Eigen::MatrixXd &_output, const Eigen::MatrixXd &_testInput, const Eigen::MatrixXd &_testOutput);
    /// test GaussianMixture function
    void testResults(const std::string &filename, const Eigen::MatrixXd &_input, const Eigen::MatrixXd &_output);
    /// normalize output
    double normalizeValue(double value, const std::pair<double, double> &minMax);
    /// denormalize output
    double denormalizeValue(double value, const std::pair<double, double> &minMax);
    void loadVectors();

    bool isFitnessSame(double fitnessCPU, double fitnessGPU);

    /// population of individuals
    Population population;
    /// boundaries for Gaussian width
    Eigen::MatrixXd gaussianBoundaries;
    /// train_data
    Eigen::MatrixXd features;
    /// output - train_data
    Eigen::MatrixXd output;
    /// verification data - input
    Eigen::MatrixXd verifInput;
    /// verification data - output
    Eigen::MatrixXd verifOutput;
    /// test data - input
    Eigen::MatrixXd testInput;
    /// test data - output
    Eigen::MatrixXd testOutput;
    /// polynomial coefficients
    Eigen::MatrixXd polyCoef;
    /// the best polynomial
    Individual bestPolynomial;
    /// actual epoch
    unsigned int currentEpoch;
    /// best fitness
    double bestFitness;
    /// best individual number
    unsigned int bestIndividual;
    /// best fitness recorder
    Recorder bestFitnessRecorder;
    /// average fitness recorder
    Recorder averageFitnessRecorder;
    /// population size recorder
    Recorder populationSizeRecorder;
    /// epoch recorder
    Recorder epochRecorder;
    Eigen::MatrixXd sample;   // point (x1,x2,x3,x4,...)
    Eigen::MatrixXd tempCoef; // temp coef
    /// input boundaries
    std::vector<std::pair<double, double>> inputDomain;
    /// output boundaries
    std::vector<std::pair<double, double>> outputDomain;
    /// config
    Config config;
};

float calc_std_var(std::vector<float> times, float avg);

namespace approximator
{
    /// create a single Map
    // Approximation *createGaussianApproximation(GaussianMixture::Config config);
    std::unique_ptr<Approximation> createGaussianApproximation(GaussianMixture::Config config);
}

#endif //_GaussianMixture_H
