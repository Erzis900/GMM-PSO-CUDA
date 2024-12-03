//
//
//
//  @ Project : Approximation library
//  @ File Name : cgene.cpp
//  @ Date : 2009-07-16
//  @ Author : Dominik Belter
//
//

#include "../include/function.h"
#include <cmath>
#include <fstream>
#include <iostream>

/// compute value of function
double Function::computeValue(double x) const
{
    //    std::cout << "width " << width << "\n";
    //    std::cout << "centroid " << centroid << "\n";
    //    std::cout << "(-width*pow((x-centroid),2.0)) " << (-width*pow((x-centroid),2.0)) << "\n";
    double cval = (-width * pow((x - centroid), 2.0)); // gaussian element
    return cval;
}

/// set Gaussian parameters
void Function::setParameters(double _centroid, double _width)
{
    centroid = _centroid;
    width = _width;
}

/// PSO -- modify position change
void Function::modifyPositionChange(double _centroidChange, double _widthChange)
{
    centroidChange = _centroidChange;
    widthChange = _widthChange;
}

/// save function to file
void Function::save2file(std::ofstream &ofstr, unsigned int dim, int type) const
{
    if (type == 1)
    {
        ofstr << "exp(-(" << width << ")*(input(" << dim + 1 << ")-(" << centroid << ")))";
    }
    if (type == 2)
    {
        ofstr << "exp(-(" << width << ")*(in[" << dim << "]-(" << centroid << ")))";
    }
}
