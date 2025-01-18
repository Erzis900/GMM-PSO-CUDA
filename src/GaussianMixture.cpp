//
//
//  @ Project : GaussianMixture library Untitled
//  @ File Name : GaussianMixture.cpp
//  @ Date : 2009-07-16
//  @ Author : Dominik Belter
//
//

#include "GaussianMixture.h"
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <iostream>
#include <chrono>

#include "../include/CUDA_decomposition.h"

#define CPU
// #define DEBUG_INFO
// #define SHOW_FIT

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/// A single instance of Gaussian Mixture
GaussianMixture::Ptr gaussianMixture;

/// constructor
GaussianMixture::GaussianMixture(GaussianMixture::Config _config) : config(_config), Approximation("Gaussian Mixture", GAUSSIANMIXTURE)
{
    currentEpoch = 0;
    bestFitness = double(1e34);
    bestFitnessRecorder.setDelay(1);
    averageFitnessRecorder.setDelay(1);
    populationSizeRecorder.setDelay(1);
    epochRecorder.setDelay(1);

    readTrainingData(config.trainFilename);     // read training data
    readVerificationData(config.verifFilename); // read verification data
    readTestData(config.testFilename);          // read test data
    setBoundaries();
    setParameters();
    initializeMatrices(); // initialize matrices

    auto t1 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&state, sizeof(curandState) * population.getPopulationSize());

    WInitRNG(state, population.getPopulationSize());

    loadVectors();

    centroidChangesSize = centroidChanges.size() * sizeof(double);
    widthChangesSize = widthChanges.size() * sizeof(double);
    bestPositionCentroidsSize = bestPositionCentroids.size() * sizeof(double);
    bestPositionWidthsSize = bestPositionCentroids.size() * sizeof(double);
    gaussianBoundariesSize = gaussianBoundariesV.size() * sizeof(double);
    inputDomainsSize = inputDomains.size() * sizeof(double);

    cudaMalloc(&d_centroidChanges, centroidChangesSize);
    cudaMalloc(&d_widthChanges, widthChangesSize);
    cudaMalloc(&d_bestPositionCentroids, bestPositionCentroidsSize);
    cudaMalloc(&d_bestPositionWidths, bestPositionWidthsSize);
    cudaMalloc(&d_gaussianBoundaries, gaussianBoundariesSize);
    cudaMalloc(&d_inputDomains, inputDomainsSize);

    inputSize = features.rows() * features.cols() * sizeof(double);
    outputSize = output.rows() * output.cols() * sizeof(double);
    fitnessResultsSize = population.getPopulationSize() * sizeof(double);

    gpuErrchk(cudaMalloc(&d_points, inputSize));
    cudaMalloc(&d_expectedOutput, outputSize);
    cudaMalloc(&d_fitnessResults, fitnessResultsSize);

    centroidsSize = centroids.size() * sizeof(double);
    // std::cout << centroids.size() << std::endl;
    widthsSize = widths.size() * sizeof(double);

    // std::cout << "Centroids size: " << centroids.size() << std::endl;

    cudaMalloc(&d_centroids, centroidsSize);
    cudaMalloc(&d_widths, widthsSize);

    std::vector<double> flattenedPoints;
    for (int i = 0; i < features.rows(); i++)
    {
        for (int j = 0; j < features.cols(); j++)
        {
            flattenedPoints.push_back(features(i, j));
        }
    }

    size_t coefSize = 0;
    for (auto &coef : allCoefs)
    {
        coefSize += coef.rows() * coef.cols() * sizeof(double);
        flattenedCoefs.insert(flattenedCoefs.end(), coef.data(), coef.data() + coef.rows() * coef.cols());
    }

    cudaMalloc(&d_polyCoef, coefSize);
    cudaMemcpy(d_polyCoef, flattenedCoefs.data(), coefSize, cudaMemcpyHostToDevice);

    // gpuErrchk(cudaMemcpy(d_points, flattenedPoints.data(), inputSize, cudaMemcpyHostToDevice));

    // cudaDeviceSynchronize();

    cudaMemcpy(d_expectedOutput, output.data(), outputSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_centroids, centroids.data(), centroidsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_widths, widths.data(), widthsSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_centroidChanges, centroidChanges.data(), centroidChangesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_widthChanges, widthChanges.data(), widthChangesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestPositionCentroids, bestPositionCentroids.data(), bestPositionCentroidsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bestPositionWidths, bestPositionWidths.data(), bestPositionWidthsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gaussianBoundaries, gaussianBoundariesV.data(), gaussianBoundariesSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputDomains, inputDomains.data(), inputDomainsSize, cudaMemcpyHostToDevice);

    auto t2 = std::chrono::high_resolution_clock::now();
    memcpyTime = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    std::cout << "memcpy time: " << memcpyTime << " ms\n";
    // runTestKernel(d_points, d_centroids);
}

/// destructor
GaussianMixture::~GaussianMixture()
{
}

/// config class construction
GaussianMixture::Config::Config(std::string configFilename)
{
    tinyxml2::XMLDocument config;
    std::string filename = "../../resources/" + configFilename;
    config.LoadFile(filename.c_str());
    if (config.ErrorID())
        std::cout << "unable to load Gaussian Misture config file.\n";
    tinyxml2::XMLElement *params = config.FirstChildElement("GaussianMixture");
    params->FirstChildElement("parameters")->QueryIntAttribute("inputsNo", &inputsNo);
    params->FirstChildElement("parameters")->QueryIntAttribute("outputsNo", &outputsNo);
    params->FirstChildElement("parameters")->QueryIntAttribute("populationSize", &populationSize);
    params->FirstChildElement("parameters")->QueryIntAttribute("gaussiansNo", &gaussiansNo);
    params->FirstChildElement("parameters")->QueryIntAttribute("maxIterations", &maxIterations);
    params->FirstChildElement("parameters")->QueryIntAttribute("trainSize", &trainSize);
    params->FirstChildElement("parameters")->QueryIntAttribute("testSize", &testSize);
    params->FirstChildElement("parameters")->QueryIntAttribute("outputType", &outputType);
    params->FirstChildElement("parameters")->QueryBoolAttribute("normalizeOutput", &normalizeOutput);
    // tinyxml2::XMLElement * bounds = config.FirstChildElement( "Boundaries" );
    for (int i = 0; i < inputsNo; i++)
    {
        // std::string paramName = "lambda" + std::to_string(i);
        // double min, max;
        // bounds->FirstChildElement( paramName.c_str() )->QueryDoubleAttribute("min", &min);
        // bounds->FirstChildElement( paramName.c_str() )->QueryDoubleAttribute("max", &max);
        boundaries.push_back(std::make_pair(0, 1));
    }
    params = config.FirstChildElement("trainingSet");
    trainFilename = params->GetText();
    params = config.FirstChildElement("testSet");
    testFilename = params->GetText();
    params = config.FirstChildElement("verificationSet");
    verifFilename = params->GetText();

    std::cout << "Configuration:\n";
    std::cout << "Inputs no (dim): " << inputsNo << "\n";
    std::cout << "Outputs no (dim): " << outputsNo << "\n";
    // std::cout << "Search method: " << searchMethod << "\n";
    std::cout << "Population size: " << populationSize << "\n";
    std::cout << "Gaussians no: " << gaussiansNo << "\n";
    std::cout << "Max iterations: " << maxIterations << "\n";
    std::cout << "Training vector size: " << trainSize << "\n";
    std::cout << "Test vector size: " << testSize << "\n";
    std::cout << "Output type: " << outputType << "\n";
    std::cout << "Normalize output: " << normalizeOutput << "\n";
    /*std::cout << "Gaussian boundaries: \n";
    int dimNo=0;
    for (auto it = boundaries.begin(); it!=boundaries.end(); it++){
        std::cout << "dim " << dimNo << ": min: " << it->first << ", max: " << it->second << "\n";
        dimNo++;
    }*/
}

/// set parameters of polynomial
void GaussianMixture::setParameters()
{
    polyCoef = Eigen::MatrixXd::Zero(config.gaussiansNo, config.outputsNo);
    population.setParameters(config.populationSize, config.gaussiansNo, config.inputsNo, config.outputsNo);
    bestPolynomial.setParameters(config.gaussiansNo, config.inputsNo, config.outputsNo);
}

/// set boundaries of function's parameters
void GaussianMixture::setBoundaries()
{
    computeBoundaries(config.boundaries, inputDomain);
    population.setBoundaries(config.boundaries, inputDomain);
    bestPolynomial.setBoundaries(config.boundaries, inputDomain);
}

/// compute n boundaries (width of Gaussian function)
void GaussianMixture::computeBoundaries(std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain)
{
    for (int i = 0; i < _gaussianBoundaries.size(); i++)
    {
        double width = _inputDomain[i].second - _inputDomain[i].first;
        if (width == 0)
            _gaussianBoundaries[i].second = std::numeric_limits<double>::max();
        else
            _gaussianBoundaries[i].second = 1 / (2 * pow(width / (10 * 4.2919), 2.0));
    }
}

/// read training data
void GaussianMixture::readTrainingData(const std::string &filename)
{
    int length;
    Eigen::MatrixXd featuresTmp = Eigen::MatrixXd::Zero(config.trainSize, config.inputsNo);
    Eigen::MatrixXd outputTmp = Eigen::MatrixXd::Zero(config.trainSize, config.outputsNo);
    // readInOutSingleRow(filename, featuresTmp, outputTmp, length);
    readInOutData(filename, featuresTmp, outputTmp, length);
    features = Eigen::MatrixXd::Zero(length, config.inputsNo);
    output = Eigen::MatrixXd::Zero(length, config.outputsNo);
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < config.inputsNo; j++)
        {
            features(i, j) = featuresTmp(i, j);
        }
        for (int j = 0; j < config.outputsNo; j++)
        {
            output(i, j) = outputTmp(i, j);
        }
    }
    if (length < config.trainSize)
    {
        config.trainSize = length;
        std::cout << "Dataset train size: " << config.trainSize << "\n";
    }

    inputDomain.resize(config.inputsNo);
    outputDomain.resize(config.outputsNo);
    // set a boundaries
    std::vector<double> values(config.trainSize, 0);
    for (int i = 0; i < config.inputsNo; i++)
    {
        for (int j = 0; j < config.trainSize; j++)
        {
            values[j] = features(j, i);
        }
        inputDomain[i].first = *std::min_element(values.begin(), values.end());
        inputDomain[i].second = *std::max_element(values.begin(), values.end());
    }
    for (int i = 0; i < config.outputsNo; i++)
    {
        for (int j = 0; j < config.trainSize; j++)
        {
            values[j] = output(j, i);
        }
        outputDomain[i].first = *std::min_element(values.begin(), values.end());
        outputDomain[i].second = *std::max_element(values.begin(), values.end());
    }
    if (config.normalizeOutput)
    {
        for (int i = 0; i < config.outputsNo; i++)
        {
            for (int j = 0; j < config.trainSize; j++)
            {
                output(j, i) = normalizeValue(output(j, i), outputDomain[i]);
            }
        }
    }
}

/// normalize output
double GaussianMixture::normalizeValue(double value, const std::pair<double, double> &minMax)
{
    return (value - minMax.first) / (minMax.second - minMax.first);
}

/// denormalize output
double GaussianMixture::denormalizeValue(double value, const std::pair<double, double> &minMax)
{
    return value * (minMax.second - minMax.first) + minMax.first;
}

/// read verification data
void GaussianMixture::readVerificationData(const std::string &filename)
{
    int length;
    Eigen::MatrixXd verifInputTmp = Eigen::MatrixXd::Zero(config.testSize, config.inputsNo);
    Eigen::MatrixXd verifOutputTmp = Eigen::MatrixXd::Zero(config.testSize, config.outputsNo);
    // readInOutSingleRow(filename, verifInputTmp, verifOutputTmp, length);
    readInOutData(filename, verifInputTmp, verifOutputTmp, length);
    //    std::cout << "legnth " << length << "\n";
    //    std::cout << "verifOutputTmp " << verifOutputTmp(0,0) << "\n";
    verifInput = Eigen::MatrixXd::Zero(length, config.inputsNo);
    verifOutput = Eigen::MatrixXd::Zero(length, config.outputsNo);
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < config.inputsNo; j++)
        {
            verifInput(i, j) = verifInputTmp(i, j);
            //            std::cout << "in " << verifInput(i,j) << ", ";
        }
        //        std::cout << "out \n";
        for (int j = 0; j < config.outputsNo; j++)
        {
            verifOutput(i, j) = verifOutputTmp(i, j);
            // std:cout << verifOutput(i,j) << "\n";
        }
        //        getchar();
    }
    if (length < config.testSize)
    {
        config.testSize = length;
        std::cout << "Dataset verification size: " << config.testSize << "\n";
    }
    if (config.normalizeOutput)
    {
        for (int i = 0; i < config.outputsNo; i++)
        {
            for (int j = 0; j < config.testSize; j++)
            {
                verifOutput(j, i) = normalizeValue(verifOutput(j, i), outputDomain[i]);
            }
        }
    }
}

/// read train/test/verification data
void GaussianMixture::readInOutData(const std::string &filename, Eigen::MatrixXd &_input, Eigen::MatrixXd &_output, int &vecLength)
{
    std::ifstream infile(filename);
    std::string line;
    vecLength = 0;
    int lineNo = 0;
    while (std::getline(infile, line))
    {
        vecLength++;
        if (lineNo >= _input.rows()) // divided by 2 because we increment for input and for output line
            break;
        std::istringstream iss(line); // input
        int colNo = 0;
        double value;
        while (iss >> value)
        {
            if (vecLength % 2 == 0)
            {
                _output(lineNo, colNo) = value;
                lineNo++;
            }
            else
            {
                _input(lineNo, colNo) = value;
            }
            if ((iss.peek() == ',') || (iss.peek() == ' '))
                iss.ignore();
            colNo++;
        }
    }
    vecLength = lineNo;
}

/// read train/test/verification data
void GaussianMixture::readInOutSingleRow(const std::string &filename, Eigen::MatrixXd &_input, Eigen::MatrixXd &_output, int &vecLength)
{
    std::ifstream infile(filename);
    std::string line;
    vecLength = 0;
    int lineNo = 0;
    while (std::getline(infile, line))
    {
        if (lineNo >= _input.rows())
            break;
        std::istringstream iss(line); // input
        int colNo = 0;
        double value;
        double isCorrect = false;
        while (iss >> value)
        {
            isCorrect = true;
            _input(lineNo, colNo) = value;
            //            if (colNo<520){
            //                if (value==100)
            //                    value=10;
            //                else
            //                    isCorrect=true;
            //                _input(lineNo,colNo) = value;
            // std::cout << "input" << colNo << " " << value <<"\n";
            // getchar();
            //            }
            //            else if (colNo==520||colNo==521){
            //                _output(lineNo,colNo-520) = value;
            ////                std::cout << "output" << colNo << " " << value <<"\n";
            ////                getchar();
            //            }
            //            else if (colNo==522){
            //                if (value!=2)
            //                    isCorrect=false;//ignore second floor
            //            }
            //            else if (colNo==523){
            //                if (value!=1)
            //                    isCorrect=false;//ignore building 1
            //            }
            if ((iss.peek() == ',') || (iss.peek() == ' '))
                iss.ignore();
            colNo++;
        }
        if (isCorrect)
        {
            vecLength++;
            lineNo++;
        }
    }
    std::cout << "Examples no: " << vecLength << "\n";
}

/// read test data
void GaussianMixture::readTestData(const std::string &filename)
{
    int length;
    Eigen::MatrixXd testInputTmp = Eigen::MatrixXd::Zero(config.testSize, config.inputsNo);
    Eigen::MatrixXd testOutputTmp = Eigen::MatrixXd::Zero(config.testSize, config.outputsNo);
    // readInOutSingleRow(filename, testInputTmp, testOutputTmp, length);
    readInOutData(filename, testInputTmp, testOutputTmp, length);
    testInput = Eigen::MatrixXd::Zero(length, config.inputsNo);
    testOutput = Eigen::MatrixXd::Zero(length, config.outputsNo);
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < config.inputsNo; j++)
            testInput(i, j) = testInputTmp(i, j);
        for (int j = 0; j < config.outputsNo; j++)
            testOutput(i, j) = testOutputTmp(i, j);
    }
    if (length < config.testSize)
    {
        config.testSize = length;
        std::cout << "Dataset test size: " << config.testSize << "\n";
    }
    if (config.normalizeOutput)
    {
        for (int i = 0; i < config.outputsNo; i++)
        {
            for (int j = 0; j < config.testSize; j++)
            {
                testOutput(j, i) = normalizeValue(testOutput(j, i), outputDomain[i]);
            }
        }
    }
}

/// initialize matrices
void GaussianMixture::initializeMatrices()
{
    sample = Eigen::MatrixXd::Zero(1, config.inputsNo);
}

/// compute output for the best polynomial
double GaussianMixture::computeOutput(const Eigen::MatrixXd &input, int outNo) const
{
    return bestPolynomial.computeValue(input, polyCoef, outNo);
}

/// compute individual fitness
double GaussianMixture::computeIndividualFitness(const Eigen::MatrixXd &_input, const Eigen::MatrixXd &_output, Eigen::MatrixXd &_coef, unsigned int individualNo)
{
    double fitness;
    if (!computeCoef(individualNo, _coef))
    {
        fitness = 1e34;
    }
    else
    {
        fitness = population.computeIndividualFitness(_input, _output, _coef, individualNo) / features.rows();
    }
    /*if (fitness < bestFitness)
    { // the best individual
        bestFitness = fitness;
        bestIndividual = individualNo;
        population.setBestIndividualNo(bestIndividual);
        polyCoef = _coef;
        bestPolynomial = population.getIndividual(individualNo);
        bestPolynomial.setGaussianBoundaries(population.getGaussianBoundaries(individualNo));
    }*/
    return fitness;
}

void GaussianMixture::loadVectors()
{
    for (int k = 0; k < config.inputsNo; k++)
    {
        inputDomains.push_back(population.getIndividual(0).getInputDomain(k).first);
        inputDomains.push_back(population.getIndividual(0).getInputDomain(k).second);
    }

    for (int i = 0; i < population.getPopulationSize(); i++)
    {
        // std::cout << population.getIndividual(i).getGaussianBoundaries().size() << std::endl;
        // 2
        
        tempCoef = Eigen::MatrixXd::Zero(population.getGaussiansNo(i), config.outputsNo);
        if(computeCoef(i, tempCoef))
        {
            allCoefs.push_back(tempCoef);
        }

        for (int k = 0; k < config.inputsNo; k++)
        {
            gaussianBoundariesV.push_back(population.getIndividual(i).getGaussianBoundaries().at(k).first);
            gaussianBoundariesV.push_back(population.getIndividual(i).getGaussianBoundaries().at(k).second);  
        }

        for (int j = 0; j < config.gaussiansNo; j++)
        {   
            for (int k = 0; k < config.inputsNo; k++)
            {
                centroids.push_back(population.getIndividual(i).getGauss(j).getCentroid(k));
                widths.push_back(population.getIndividual(i).getGauss(j).getWidth(k));

                centroidChanges.push_back(population.getIndividual(i).getGauss(j).getCentroidChange(k));
                widthChanges.push_back(population.getIndividual(i).getGauss(j).getWidthChange(k));

                bestPositionCentroids.push_back(population.getIndividual(i).getBestPosition(j).getCentroid(k));
                bestPositionWidths.push_back(population.getIndividual(i).getBestPosition(j).getWidth(k));
            }
        }
    }
}

bool GaussianMixture::isFitnessSame(double fitnessCPU, double fitnessGPU)
{
    double deviation = 0.001;
    return (fitnessGPU >= (fitnessCPU - deviation)) && (fitnessGPU <= (fitnessCPU + deviation));
}

float calc_std_var(std::vector<float> times, float avg)
{
    float sum = 0;
    for (auto &x : times)
    {
        sum += std::pow((x - avg), 2);
    }

    return std::sqrt(sum / times.size());
}

/// search for the best Approximation function - PSO method
void GaussianMixture::train()
{
    float totalFitnessTimeCPU = 0;
    float totalFitnessTimeGPU = 0;

    float totalUpdateTimeCPU = 0;
    float totalUpdateTimeGPU = 0;

    // runTestKernel(d_points, d_centroids);
    #ifdef CPU
        float CPUfitnessTime = 0;
        fitnessCPU.clear();
        for (int i = 0; i < (int)population.getPopulationSize(); i++)
        { // fitness evaluation for every individual
            tempCoef = Eigen::MatrixXd::Zero(population.getGaussiansNo(i), config.outputsNo);

            auto t1 = std::chrono::high_resolution_clock::now();
            double fit = computeIndividualFitness(features, output, tempCoef, i);
            auto t2 = std::chrono::high_resolution_clock::now();
            float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
            CPUfitnessTime += duration;

            fitnessCPU.push_back(fit);

            cout.precision(std::numeric_limits<double>::max_digits10);
            #ifdef SHOW_FIT
                std::cout << "[CPU] Individual " << i + 1 << " fitness: " << fit << "\n";
            #endif

            #ifdef DEBUG_INFO
                std::cout << "Duration: " << duration << "\n";
            #endif
        }
        // std::cout << "[CPU] Avg fitness time: " << CPUfitnessTime / population.getPopulationSize() << " ms\n";
        std::cout << "[CPU] Fitness calculated in: " << CPUfitnessTime << " ms\n";
        totalFitnessTimeCPU += CPUfitnessTime;
        fitnessTimesCPU.push_back(CPUfitnessTime);
    #endif

    std::vector<double> fitnessResults(population.getPopulationSize(), 0.0);

    auto t1 = std::chrono::high_resolution_clock::now();
    calculateFitnessCUDA(d_centroids, d_widths, d_polyCoef, d_points, d_expectedOutput, d_fitnessResults, fitnessResults.data(), population.getPopulationSize(), config.gaussiansNo, config.inputsNo, config.trainSize, bestFitness, d_bestPositionCentroids, d_bestPositionWidths);
    auto t2 = std::chrono::high_resolution_clock::now();
    float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    std::cout << "[CUDA] Fitness calculated in: " << duration + memcpyTime << " ms\n";
    totalFitnessTimeGPU += duration;
    fitnessTimesGPU.push_back(duration);

    #ifndef SHOW_FIT
        std::cout << std::endl;
    #endif

    for (int i = 0; i < population.getPopulationSize(); i++)
    {
        // global
        if (fitnessResults[i] < bestFitness)
        {
            bestFitness = fitnessResults[i];
            bestIndividual = i;
            population.setBestIndividualNo(bestIndividual);
            polyCoef = allCoefs.at(i);
            bestPolynomial = population.getIndividual(i);
            bestPolynomial.setGaussianBoundaries(population.getGaussianBoundaries(i));
        }
        // std::cout << "bestFitness: " << bestFitness << std::endl;

        // local
        if (fitnessResults[i] < population.getIndividual(i).getBestFitness())
        {
            population.getIndividual(i).changeBestPosition();
            population.getIndividual(i).setBestFitness(fitnessResults[i]);
        }

        cout.precision(std::numeric_limits<double>::max_digits10);
        #ifdef SHOW_FIT
            std::cout << "[CUDA] Individual " << i + 1 << " fitness: " << fitnessResults[i] << " ";
        #endif

        #ifdef CPU
            #ifdef SHOW_FIT 
                std::cout << " ";
                if(isFitnessSame(fitnessCPU[i], fitnessResults[i]))
                {
                    std::cout << "SAME\n";
                }
                else
                {
                    std::cout << "WRONG\n";
                }
            #endif
        #else
            std::cout << std::endl;
        #endif
    }

    population.computeAverageFitness();

    #ifdef SHOW_FIT 
        std::cout << "[CUDA] epoch: " << currentEpoch << ", best fitness: " << bestFitness << ", best ind: " << bestIndividual + 1 << "\n\n";
    #endif
    // save initial values
    bestFitnessRecorder.save(bestFitness);
    averageFitnessRecorder.save(population.getAverageFitness());
    
    // updateParticlesCUDA(state, config.inputsNo, config.trainSize, population.getPopulationSize(), gaussianBoundariesV, config.gaussiansNo); 

    while (currentEpoch < config.maxIterations)
    {
        allCoefs.clear();
        bestPositionCentroids.clear();
        bestPositionWidths.clear();

        #ifdef CPU
            fitnessCPU.clear();
            float CPUfitnessTime = 0;
            float CPUupdateTime = 0;
        #endif
        // centroids.clear();
        // widths.clear();

        for (int i = 0; i < (int)population.getPopulationSize(); i++)
        { // move individuals    
            #ifdef CPU
                if (i != bestIndividual)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    population.moveIndividual(i, bestPolynomial);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
                    CPUupdateTime += duration;
                }

                tempCoef = Eigen::MatrixXd::Zero(population.getGaussiansNo(i), config.outputsNo);

                auto t1 = std::chrono::high_resolution_clock::now();
                double fit = computeIndividualFitness(features, output, tempCoef, i);
                auto t2 = std::chrono::high_resolution_clock::now();
                float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
                CPUfitnessTime += duration;

                fitnessCPU.push_back(fit);

                #ifdef SHOW_FIT
                    std::cout << "[CPU] Individual " << i + 1 << " fitness: " << fit << "\n";
                #endif

                #ifdef DEBUG_INFO
                    std::cout << "Duration: " << duration << "\n";
                #endif
            #endif

            for (int j = 0; j < config.gaussiansNo; j++)
            {   
                for (int k = 0; k < config.inputsNo; k++)
                {
                    bestPositionCentroids.push_back(population.getIndividual(i).getBestPosition(j).getCentroid(k));
                    bestPositionWidths.push_back(population.getIndividual(i).getBestPosition(j).getWidth(k));
                }
            }

            tempCoef = Eigen::MatrixXd::Zero(population.getGaussiansNo(i), config.outputsNo);
            if(computeCoef(i, tempCoef))
            {
                allCoefs.push_back(tempCoef);
            }
            else
            {
                std::cout << "computeCoef failed!\n";
                return;
            }
        }
        #ifdef CPU
            // std::cout << "[CPU] Avg fitness time: " << CPUfitnessTime / population.getPopulationSize() << " ms\n";
            std::cout << "[CPU] Fitness calculated in: " << CPUfitnessTime << " ms\n";
            totalFitnessTimeCPU += CPUfitnessTime;
            totalUpdateTimeCPU += CPUupdateTime;

            fitnessTimesCPU.push_back(CPUfitnessTime);
            updateTimesCPU.push_back(CPUupdateTime);
        #endif
        
        flattenedCoefs.clear();

        size_t coefSize = 0;
        for (auto &coef : allCoefs)
        {
            coefSize += coef.rows() * coef.cols() * sizeof(double);
            flattenedCoefs.insert(flattenedCoefs.end(), coef.data(), coef.data() + coef.rows() * coef.cols());
        }

        // cudaMalloc(&d_polyCoef, coefSize);
        cudaMemcpy(d_polyCoef, flattenedCoefs.data(), coefSize, cudaMemcpyHostToDevice);

        cudaMemcpy(d_bestPositionCentroids, bestPositionCentroids.data(), bestPositionCentroidsSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bestPositionWidths, bestPositionWidths.data(), bestPositionWidthsSize, cudaMemcpyHostToDevice);

        auto t1Update = std::chrono::high_resolution_clock::now();
        runUpdateKernel(d_centroids, d_widths, state, config.inputsNo, config.trainSize, population.getPopulationSize(), d_gaussianBoundaries, config.gaussiansNo, bestIndividual, d_centroidChanges, d_widthChanges, d_bestPositionCentroids, d_bestPositionWidths, d_inputDomains);
        auto t2Update = std::chrono::high_resolution_clock::now();
        float durationUpdate = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(t2Update - t1Update).count());
        totalUpdateTimeGPU += durationUpdate;
        updateTimesGPU.push_back(durationUpdate);

        fitnessResults.clear();

        auto t1 = std::chrono::high_resolution_clock::now();
        calculateFitnessCUDA(d_centroids, d_widths, d_polyCoef, d_points, d_expectedOutput, d_fitnessResults, fitnessResults.data(), population.getPopulationSize(), config.gaussiansNo, config.inputsNo, config.trainSize, bestFitness, d_bestPositionCentroids, d_bestPositionWidths);
        auto t2 = std::chrono::high_resolution_clock::now();
        float duration = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
        std::cout << "[CUDA] Fitness calculated in: " << duration << " ms\n";
        totalFitnessTimeGPU += duration;
        fitnessTimesGPU.push_back(duration);

        for (int i = 0; i < population.getPopulationSize(); i++)
        {
            // std::cout << fitnessResults[i] << std::endl;
            // std::cout << "bestFitness: " << bestFitness << std::endl;
            if (fitnessResults[i] < bestFitness)
            {
                bestFitness = fitnessResults[i];
                bestIndividual = i;
                population.setBestIndividualNo(bestIndividual);
                polyCoef = allCoefs.at(i);
                bestPolynomial = population.getIndividual(i);
                bestPolynomial.setGaussianBoundaries(population.getGaussianBoundaries(i));
            }

            // local
            // std::cout << "Before: " << population.getIndividual(i).getBestFitness() << std::endl;
            if (fitnessResults[i] < population.getIndividual(i).getBestFitness())
            {
                // std::cout << "fitness: " << fitnessResults[i] << std::endl;
                // std::cout << population.getIndividual(i).getBestFitness() << std::endl;
                population.getIndividual(i).setBestFitness(fitnessResults[i]);
                population.getIndividual(i).changeBestPosition();
            }

            #ifdef SHOW_FIT
                std::cout << "[CUDA] Individual " << i + 1 << " fitness: " << fitnessResults[i];
            #endif

            #ifdef CPU
                #ifdef SHOW_FIT 
                    std::cout << " ";
                    if(isFitnessSame(fitnessCPU[i], fitnessResults[i]))
                    {
                        std::cout << "SAME\n";
                    }
                    else
                    {
                        std::cout << "WRONG\n";
                    }
                #endif
            #else
                std::cout << std::endl;
            #endif
        }

            //tempCoef = Eigen::MatrixXd::Zero(population.getGaussiansNo(i), 1);
            //double fit = computeIndividualFitness(features, output, tempCoef, i); // and fitness value
            //std::cout << "Individual " << i << " fitness: " << fit << "\n";
        currentEpoch++;
        // save
        // if ((currentEpoch%(config.maxIterations/10))==0) {
        //std::cout << "epoch: " << currentEpoch << ", best fitness: " << bestFitness << "\n";

        //}
        population.computeAverageFitness();
        #ifdef SHOW_FIT
            std::cout << "[CUDA] epoch: " << currentEpoch << ", best fitness: " << bestFitness << ", best ind: " << bestIndividual + 1 << "\n";
        #endif
        bestFitnessRecorder.save(bestFitness);
        averageFitnessRecorder.save(population.getAverageFitness());

        std::cout << "\n";
    }
    // save initial values
    bestFitnessRecorder.save2file("bestFitness.txt", "bestFitness");
    averageFitnessRecorder.save2file("average_fitness.txt", "average_fitness");

    double maxerr, averr;
    computeAPE(features, output, averr, maxerr);
    std::cout << "Error for training data: average " << averr << ", max " << maxerr << "\n";
    computeAPE(testInput, testOutput, averr, maxerr);
    std::cout << "Error for test data: average " << averr << ", max " << maxerr << "%f\n";
    if (config.inputsNo < 3)
        savePlotFile("plot_figure.m", features, output, testInput, testOutput);
    testResults("final_results.txt", features, output);
    testResults("final_results_test.txt", testInput, testOutput);

    std::cout << "Total CPU fitness time: " << totalFitnessTimeCPU << " ms\n";
    std::cout << "Total GPU fitness time: " << totalFitnessTimeGPU << " ms\n";

    float avgCPUTime = totalFitnessTimeCPU / (config.maxIterations + 1);
    float avgGPUTime = totalFitnessTimeGPU / (config.maxIterations + 1);
    std::cout << "Avg CPU fitness time: " << avgCPUTime << " ms\n";
    std::cout << "Avg GPU fitness time: " << avgGPUTime << " ms\n";

    // for (auto &v : updateTimesCPU)
    // {
    //     std::cout << v << std::endl;
    // }

    // for (auto &v : updateTimesGPU)
    // {
    //     std::cout << v << std::endl;
    // }

    std::cout << "std dev fitness CPU: " << calc_std_var(fitnessTimesCPU, avgCPUTime) << "\n";
    std::cout << "std dev fitness GPU: " << calc_std_var(fitnessTimesGPU, avgGPUTime) << "\n";

    std::cout << std::endl;

    std::cout << "Total CPU update time: " << totalUpdateTimeCPU << " micro\n";
    std::cout << "Total GPU update time: " << totalUpdateTimeGPU << " micro\n";

    float avgCPUupdate = totalUpdateTimeCPU / (config.maxIterations + 1);
    float avgGPUupdate = totalUpdateTimeGPU / (config.maxIterations + 1);
    std::cout << "Avg CPU update time: " << avgCPUupdate << " micro\n";
    std::cout << "Avg GPU update time: " << avgGPUupdate << " micro\n";

    std::cout << "std dev fitness CPU: " << calc_std_var(updateTimesCPU, avgCPUupdate) << "\n";
    std::cout << "std dev fitness GPU: " << calc_std_var(updateTimesGPU, avgGPUupdate) << "\n";

    std::ofstream CPUtimeFile("../../cpu_fitness_times.csv", std::ios::app);
    CPUtimeFile << config.populationSize << "," << avgCPUTime << "," << calc_std_var(fitnessTimesCPU, avgCPUTime) << "\n";

    std::ofstream GPUtimeFile("../../gpu_fitness_times.csv", std::ios::app);
    GPUtimeFile << config.populationSize << "," << avgGPUTime << "," << calc_std_var(fitnessTimesGPU, avgGPUTime) << "\n";

    std::ofstream memcpyTimeFile("../../memcpy_times.csv", std::ios::app);
    memcpyTimeFile << config.populationSize << "," << memcpyTime << "\n";

    std::ofstream updateTimeFileCPU("../../cpu_update_times.csv", std::ios::app);
    updateTimeFileCPU << config.populationSize << "," << avgCPUupdate << "," << calc_std_var(updateTimesCPU, avgCPUupdate) << "\n";

    std::ofstream updateTimeFileGPU("../../gpu_update_times.csv", std::ios::app);
    updateTimeFileGPU << config.populationSize << "," << avgGPUupdate << "," << calc_std_var(fitnessTimesGPU, avgGPUTime)  << "\n";

    memcpyTimeFile.close();
    CPUtimeFile.close();
    GPUtimeFile.close();
    updateTimeFileCPU.close();
    updateTimeFileGPU.close();

    cudaFree(d_points);
    cudaFree(d_expectedOutput);
    cudaFree(d_fitnessResults);
    cudaFree(d_polyCoef);
    cudaFree(d_centroids);
    cudaFree(d_widths);
    cudaFree(d_centroidChanges);
    cudaFree(d_widthChanges);
    cudaFree(d_bestPositionCentroids);
    cudaFree(d_gaussianBoundaries);
    cudaFree(d_inputDomains);
}

/// compute coefficients
bool GaussianMixture::computeCoef(unsigned int individual_no, Eigen::MatrixXd &resultCoef)
{
    Individual individual = population.getIndividual(individual_no);
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(features.rows(), individual.getGaussiansNo());
    Eigen::MatrixXd Vprim = Eigen::MatrixXd::Zero(individual.getGaussiansNo(), individual.getGaussiansNo());
    Eigen::MatrixXd p = Eigen::MatrixXd::Zero(individual.getGaussiansNo(), config.outputsNo);
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(individual.getGaussiansNo(), individual.getGaussiansNo());
    Eigen::MatrixXd Ginv = Eigen::MatrixXd::Zero(individual.getGaussiansNo(), individual.getGaussiansNo());
    for (int i = 0; i < features.rows(); i++)
    {
        sample = features.row(i);
        // std::cout << "sample " << sample << "\n";
        // getchar();
        for (int j = 0; j < individual.getGaussiansNo(); j++)
        {
            V(i, j) = individual.computeValue(j, sample);
            // std::cout << "V(i,j) " << V(i,j) << ", ";
        }
        //(*r)(i,0) = (*output)(i,0);
    }
    // std::cout << "\n";
    // getchar();
    Vprim = V.transpose();
    G = Vprim * V;
    p = Vprim * output;
    // cout << "G: " << G << endl <<" p: " <<p << endl <<" V: " << V << endl <<" Vprim: " << Vprim <<endl;
    // if (G.lu().isInvertible()){ // inversion successfull

    // using inverse of Gramm matrix
    // Ginv = G.inverse();
    // resultCoef = Ginv*p;
    // using LDLT decomposition
    //(*G).ldlt().solve(*p,result_coef);
    //(*G).llt().solve(*p,result_coef);
    //  a little bit faster llt
    G.llt().solveInPlace(p);
    // CUDA_QRfactorization(G.data(), G.rows(), G.cols(), p.data(), p.rows(), p.cols(), resultCoef.data(), resultCoef.rows());
    resultCoef = p;
    // LU
    //(*G).lu().solve(*p,result_coef);
    // SVD
    //(*G).svd().solve(*->gaussiansNo,p,result_coef);//so slooow

    // cout << "WEE"<< (*result_coef).rows()<<endl;
    /*for (int i=0;i<resultCoef.rows();i++){
      cout << "Result coeff " << resultCoef(i,0) << " ";
    }*/
    return true;
    //}
    // else
    //	return false; // inversion failed
}

/// save plot file (test of GaussianMixture function)
void GaussianMixture::savePlotFile(const std::string &filename, const Eigen::MatrixXd &_input, const Eigen::MatrixXd &_output, const Eigen::MatrixXd &_testInput, const Eigen::MatrixXd &_testOutput)
{
    std::ofstream ofstr;
    ofstr.open(filename);
    ofstr << "close all;\n";
    unsigned int iter = 0;
    for (int j = 0; j < config.outputsNo; j++)
    {
        iter = 0;
        for (int i = 0; i < _input.rows(); i++)
        {
            sample = _input.row(iter);
            if (config.inputsNo == 1)
            {
                ofstr << "x=" << _input(iter, 0) << "; y=" << _output(iter, j) << ";\n";
                ofstr << "plot(x,y,'*r'); hold on;\n";
                ofstr << "x=" << _input(iter, 0) << "; y=" << computeOutput(sample, j) << ";\n";
                ofstr << "plot(x,y,'ob'); hold on;\n";
            }
            else if (config.inputsNo == 2)
            {
                ofstr << "x=" << _input(iter, 0) << "; y=" << _input(iter, 1) << "; z=" << _output(iter, j) << ";\n";
                ofstr << "plot3(x,y,z,'*r'); hold on;\n";
                ofstr << "x=" << _input(iter, 0) << "; y=" << _input(iter, 1) << "; z=" << computeOutput(sample, j) << ";\n";
                ofstr << "plot3(x,y,z,'ob'); hold on;\n";
            }
            iter++;
        }
    }
    iter = 0;
    Eigen::MatrixXd sample;
    sample = Eigen::MatrixXd::Zero(1, config.inputsNo);
    for (int j = 0; j < config.outputsNo; j++)
    {
        iter = 0;
        for (int i = 0; i < _testInput.rows(); i++)
        {
            sample = _testInput.row(iter);
            if (config.inputsNo == 1)
            {
                ofstr << "x=" << _testInput(iter, 0) << "; y=" << _testOutput(iter, j) << ";\n";
                ofstr << "plot(x,y,'*g'); hold on;\n";
                ofstr << "x=" << _testInput(iter, 0) << "; y=" << computeOutput(sample, j) << ";\n";
                ofstr << "plot(x,y,'ok'); hold on;\n";
            }
            else if (config.inputsNo == 2)
            {
                ofstr << "x=" << _testInput(iter, 0) << "; y=" << _testInput(iter, 1) << "; z=" << _testOutput(iter, j) << ";\n";
                ofstr << "plot3(x,y,z,'*g'); hold on;\n";
                ofstr << "x=" << _testInput(iter, 0) << "; y=" << _testInput(iter, 1) << "; z=" << computeOutput(sample, j) << ";\n";
                ofstr << "plot3(x,y,z,'ok'); hold on;\n";
            }
            iter++;
        }
    }
    double arg[2];
    if (config.inputsNo == 2)
        ofstr << "[X,Y]=meshgrid([" << inputDomain[0].first << ":" << (inputDomain[0].second - inputDomain[0].first) / 100 << ":" << inputDomain[0].second << "],[" << inputDomain[1].first << ":" << (inputDomain[1].second - inputDomain[1].first) / 100 << ":" << inputDomain[1].second << "]);\n";

    for (int j = 0; j < config.outputsNo; j++)
    {
        for (int i = 0; i < 100; i++)
        {
            arg[0] = inputDomain[0].first + i * ((inputDomain[0].second - inputDomain[0].first) / 99);
            if (config.inputsNo == 1)
            {
                sample(0, 0) = arg[0];
                ofstr << "x(" << i + 1 << ")=" << arg[0] << "; y(" << i + 1 << ")=" << computeOutput(sample, j) << ";\n";
            }
            else if (config.inputsNo == 2)
            { // jakis mesh lub surf
                for (int k = 0; k < 100; k++)
                {
                    arg[0] = inputDomain[0].first + i * ((inputDomain[0].second - inputDomain[0].first) / 99);
                    arg[1] = inputDomain[1].first + k * ((inputDomain[1].second - inputDomain[1].first) / 99);
                    sample(0, 0) = arg[0];
                    sample(0, 1) = arg[1];
                    ofstr << "Z(" << k + 1 << "," << i + 1 << ")=" << computeOutput(sample, j) << ";\n";
                }
            }
        }
        ofstr << "plot(x,y,'b'); hold on;\n";
    }
    ofstr << "%Input domain: " << inputDomain[0].first << ", " << inputDomain[0].second << "\n";

    if (config.inputsNo == 2)
    {
        ofstr << "X=X(1:1:100,1:1:100);\n";
        ofstr << "Y=Y(1:1:100,1:1:100);\n";
        ofstr << "mesh(X,Y,Z);\n";
    }
    ofstr.close();
}

/// test GaussianMixture function
void GaussianMixture::testResults(const std::string &filename, const Eigen::MatrixXd &_input, const Eigen::MatrixXd &_output)
{
    Eigen::MatrixXd sample = Eigen::MatrixXd::Zero(1, config.inputsNo);
    int iter = 0;
    std::ofstream ofstr;
    ofstr.open(filename);
    int correct = 0;
    for (int i = 0; i < _input.rows(); i++)
    {
        sample = _input.row(iter);

        ofstr << "f" << i << "(";
        for (int j = 0; j < sample.cols(); j++)
            ofstr << sample(0, j) << ",";
        ofstr << ") = (";
        for (int k = 0; k < config.outputsNo; k++)
        {
            if (config.normalizeOutput)
            {
                ofstr << denormalizeValue(computeOutput(sample, k), outputDomain[k]) << ", ";
            }
            else
                ofstr << computeOutput(sample, k) << ", ";
        }
        ofstr << ") should be (";
        for (int k = 0; k < config.outputsNo; k++)
        {
            if (config.normalizeOutput)
            {
                ofstr << denormalizeValue(_output(iter, k), outputDomain[k]) << ", ";
            }
            else
                ofstr << _output(iter, k) << ", ";
        }
        ofstr << ") error = (";
        for (int k = 0; k < config.outputsNo; k++)
        {
            if (config.normalizeOutput)
            {
                ofstr << denormalizeValue(computeOutput(sample, k), outputDomain[k]) - denormalizeValue((_output)(iter, k), outputDomain[k]) << ", ";
            }
            else
                ofstr << computeOutput(sample, k) - (_output)(iter, k) << ", ";
        }
        ofstr << ")\n";
        if (fabs(computeOutput(sample, 0) - (_output)(iter, 0)) < 0.5)
            correct++;
        iter++;
    }
    ofstr << "correct = " << (double(correct) / double(iter)) * 100 << "%\n";
    for (int i = 0; i < config.inputsNo; i++)
    {
        ofstr << "nbound[" << i << "] = (" << bestPolynomial.getGaussianBoundaries()[i].first << "," << bestPolynomial.getGaussianBoundaries()[i].second << ")\n";
    }
    double maxerr, averr;
    computeAPE(_input, _output, averr, maxerr);
    ofstr << "Error: average " << averr << ", max " << maxerr << "\n";
    ofstr << "Poly elements number: " << bestPolynomial.getGaussiansNo() << "\n";
    ofstr.close();
    bestPolynomial.save2file(filename, polyCoef, config.outputType);
}

/// test GaussianMixture function
double GaussianMixture::computeAPE(const Eigen::MatrixXd &_input, const Eigen::MatrixXd &_output, double &avError, double &maxError)
{
    sample = Eigen::MatrixXd::Zero(1, config.inputsNo);
    int iter = 0;
    double sum = 0;
    avError = 0;
    maxError = 0;
    for (int i = 0; i < _input.rows(); i++)
    {
        sample = _input.row(iter);
        for (int k = 0; k < config.outputsNo; k++)
        {
            double error;
            if (config.normalizeOutput)
            {
                error = fabs(denormalizeValue(_output(iter, k), outputDomain[k]) - denormalizeValue(computeOutput(sample, k), outputDomain[k]));
            }
            else
                error = fabs(_output(iter, k) - computeOutput(sample, k));
            sum += error; /// abs(_output(iter,k));
            if (error > maxError)
                maxError = error;
            avError += error;
        }
        iter++;
    }
    avError = avError / double(iter);
    std::cout << "average_error=" << avError << " max_error=" << maxError << "\n";
    return (sum / double(iter)) * 100.0;
}

approximator::Approximation *approximator::createGaussianApproximation(GaussianMixture::Config config)
{
    gaussianMixture.reset(new GaussianMixture(config));
    return gaussianMixture.get();
}
