//
//  @ Project : Gaussian Mixture Regression library
//  @ File Name : MultivariateGaussian.cpp
//  @ Date : 2009-07-16
//  @ Author : Dominik Belter
//
//

#include "../include/multivariateGaussian.h"
#include <chrono>
#include <fstream>

/// default constructor
MultivariateGaussian::MultivariateGaussian()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}

/// default destructor
MultivariateGaussian::~MultivariateGaussian()
{
}

/// initialize
void MultivariateGaussian::setParameters(unsigned int _dim)
{
    dim = _dim;
    // gaussianBoundariesChange.resize(dim);
}

/// set boundaries of function's parameters
void MultivariateGaussian::setBoundaries(const std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain)
{
    // gaussianBoundaries = _gaussianBoundaries;
    // inputDomain = _inputDomain;
    initializeGauss(_gaussianBoundaries, _inputDomain);
}

/// initialize 1D Gaussian
void MultivariateGaussian::initializeGauss(const std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain)
{
    gaussians.resize(dim);
    for (int i = 0; i < dim; i++)
    {
        std::uniform_real_distribution<double> distributionCentroid(_inputDomain[i].first, _inputDomain[i].second);
        std::uniform_real_distribution<double> distributionWidth(_gaussianBoundaries[i].first, _gaussianBoundaries[i].second);
        gaussians[i].setParameters(distributionCentroid(generator), distributionWidth(generator));

        // PSO
        gaussians[i].modifyPositionChange(0, 0);
        // gaussianBoundariesChange[i]=0;
    }
}

/// compue value of gene point must be row vector
double MultivariateGaussian::computeValue(const Eigen::MatrixXd &point) const
{
    double result = 0;
    for (int i = 0; i < dim; i++)
    {
        double cval = gaussians[i].computeValue(point(0, i));
        result += cval;
        // std::cout << "point(0,i) " << point(0,i) << "\n";
        // std::cout << "gaussians[i].computeValue(point(0,i)) " << gaussians[i].computeValue(point(0,i)) << "\n";
    }
    result /= double(dim);
    //    std::cout << "result " << result << "\n";
    //    std::cout << "exp(result) " << exp(result) << "\n";
    //    getchar();
    return exp(result);
}

/// save function to file
void MultivariateGaussian::save2file(std::ofstream &ofstr, int type)
{
    ofstr << "exp(-(";
    for (int i = 0; i < (int)dim; i++)
    {
        gaussians[i].save2file(ofstr, i, type);
        if (i < (dim - 1))
        {
            ofstr << "+";
        }
    }
    ofstr << "))";
}
