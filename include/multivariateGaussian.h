//
//  @ Project : Gaussian Mixture Regression library
//  @ File Name : multivariateGaussian.h
//  @ Date : 2009-07-16
//  @ Author : Dominik Belter

#ifndef _MULTIVARIATEGAUSSIAN_H
#define _MULTIVARIATEGAUSSIAN_H

#include "function.h"
#include "../3rdParty/Eigen/Geometry"

#include <iostream>
#include <vector>
#include <random>

class MultivariateGaussian
{
public:
    /// default constructor
    MultivariateGaussian();
    /// default destructor
    ~MultivariateGaussian();
    /// initialize
    void setParameters(unsigned int _dim);
    /// set boundaries of function's parameters
    void setBoundaries(const std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain);
    /// create multivariate Gaussian
    void createMultivariateGaussian();
    /// compue value of gene point must be column vector
    double computeValue(const Eigen::MatrixXd &point) const;
    /// save function to file
    void save2file(std::ofstream &ofstr, int type);
    /// initialize 1D Gaussian
    void initializeGauss(const std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain);
    /// Get Gaussian width
    inline double getWidth(int gaussianNo) const { return gaussians[gaussianNo].getWidth(); }
    /// Get centroid
    inline double getCentroid(int gaussianNo) const { return gaussians[gaussianNo].getCentroid(); }
    /// Get width change
    inline double getWidthChange(int gaussianNo) const { return gaussians[gaussianNo].getWidthChange(); }
    /// Get centroid change
    inline double getCentroidChange(int gaussianNo) const { return gaussians[gaussianNo].getCentroidChange(); }
    /// Set width
    inline void setWidth(int gaussianNo, double _width) { gaussians[gaussianNo].setWidth(_width); }
    /// Set centroid
    inline void setCentroid(int gaussianNo, double _centroid) { gaussians[gaussianNo].setCentroid(_centroid); }
    /// Set width change
    inline void setWidthChange(int gaussianNo, double _widthChange) { gaussians[gaussianNo].setWidthChange(_widthChange); }
    /// Set centroid change
    inline void setCentroidChange(int gaussianNo, double _centroidChange) { gaussians[gaussianNo].setCentroidChange(_centroidChange); }

    inline unsigned int getDim() const { return dim;}
private:
    /// random number generator
    std::default_random_engine generator;
    /// functions in gene
    std::vector<Function> gaussians;
    /// dimension of the approximation space
    unsigned int dim;
    /// boundaries n-coefficient for functions - two-column vector [min, max]
    // std::vector<std::pair<double,double>> gaussianBoundaries;
    /// boundaries a-coefficient for functions - two-column vector [min, max]
    // std::vector<std::pair<double,double>> inputDomain;
    /// n_bound change
    // std::vector<double> gaussianBoundariesChange;
};

#endif //_MULTIVARIATEGAUSSIAN_H
