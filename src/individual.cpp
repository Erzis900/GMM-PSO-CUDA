//
//
//  @ Project : Approximation library Untitled
//  @ File Name : Individual.cpp
//  @ Date : 2009-07-16
//  @ Author : Dominik Belter
//
//

#include "../include/individual.h"
#include <chrono>
#include <random>
#include <fstream>

/// default constructor
Individual::Individual()
{
    fitnessValue = double(3.40282e+038);
    maxChange = 0.25;
    bestFitnessValue = double(3.40282e+038);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);
}

/// default destructor
Individual::~Individual()
{
}

/// set parameters
void Individual::setParameters(unsigned int _gaussiansNo, unsigned int _dim, unsigned int _outputsNo)
{
    gaussiansNo = _gaussiansNo;
    dim = _dim;
    outputsNo = _outputsNo;
    createChromosome();
    c1 = 2;
    c2 = 2;
    maxChange = 0.25;
}

/// set boundaries of function's parameters
void Individual::setBoundaries(const std::vector<std::pair<double, double>> &_gaussianBoundaries, const std::vector<std::pair<double, double>> &_inputDomain)
{
    gaussianBoundaries = _gaussianBoundaries;
    inputDomain = _inputDomain;
    gaussianBoundariesChange.resize(gaussianBoundaries.size());
    bestGaussianBoundaries.resize(gaussianBoundaries.size());
    for (int i = 0; i < gaussianBoundaries.size(); i++)
    {
        gaussianBoundariesChange[i] = 0;
        std::uniform_real_distribution<double> distribution(gaussianBoundaries[i].first, gaussianBoundaries[i].second);
        gaussianBoundaries[i].second = distribution(generator);
        bestGaussianBoundaries[i] = gaussianBoundaries[i].second;
    }
}

/// set boundaries of function's parameters
void Individual::setGaussianBoundaries(const std::vector<std::pair<double, double>> &_gaussianBoundaries)
{
    gaussianBoundaries = _gaussianBoundaries;
}

/// create gene
void Individual::initializeGaussian(int gaussianNo)
{
    chromosome[gaussianNo].setParameters(dim);
    chromosome[gaussianNo].setBoundaries(gaussianBoundaries, inputDomain);

    bestPosition[gaussianNo].setParameters(dim);
    bestPosition[gaussianNo].setBoundaries(gaussianBoundaries, inputDomain);
}

/// create chromosome
void Individual::createChromosome()
{
    chromosome.resize(gaussiansNo);
    bestPosition.resize(gaussiansNo);
    for (int i = 0; i < (int)gaussiansNo; i++)
    {
        initializeGaussian(i);
        // chromosome[i].setBoundaries(gaussianBoundaries, inputDomain);
        // bestPosition[i].setBoundaries(gaussianBoundaries, inputDomain);
    }

    for (int i = 0; i < (int)gaussiansNo; i++)
    {
        bestPosition[i] = chromosome[i];
    }
}

// PSO change the best position
void Individual::changeBestPosition()
{
    for (int i = 0; i < (int)gaussiansNo; i++)
    {
        bestPosition[i] = chromosome[i];
    }
    for (int i = 0; i < dim; i++)
    {
        bestGaussianBoundaries[i] = gaussianBoundaries[i].second;
    }
}

/// compute value of polynomial represented by individual; point must be a column vector
double Individual::computeValue(const Eigen::MatrixXd &point, const Eigen::MatrixXd &coefficient, int outNo) const
{
    double result = 0;
    for (int i = 0; i < coefficient.rows(); i++)
    {
        double c = coefficient(i, outNo);
        //std::cout << "CPU c = " << c << std::endl;
        /*		if ((_isnan(c))||(_finite(c))) c=0;
                char ch1[10],ch2[10];
                double dd = sqrt(-1.0);
                sprintf(ch1, "%g", dd);
                sprintf(ch2, "%g", c);
                if (strcmp(ch1,ch2)==0)
                        c=0;*/
        double comp = chromosome[i].computeValue(point);
        result += c * chromosome[i].computeValue(point);
    }
    return result;
}

/// compute value of polynomial represented by individual; point must be a column vector
double Individual::computeValue(int gaussNo, const Eigen::MatrixXd &point) const
{
    return chromosome[gaussNo].computeValue(point);
}

/// move individual
void Individual::moveIndividual(const Individual &bestParticle)
{
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < gaussiansNo; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            // chromosome[i].setCentroidChange(j, chromosome[i].getCentroidChange(j) + c1 * distribution(generator) * (bestParticle.chromosome[i].getCentroid(j) - chromosome[i].getCentroid(j)) + c2 * distribution(generator) * (bestPosition[i].getCentroid(j) - chromosome[i].getCentroid(j)));
            chromosome[i].setCentroidChange(j, chromosome[i].getCentroidChange(j) + c1 * (bestParticle.chromosome[i].getCentroid(j) - chromosome[i].getCentroid(j)) + c2 * (bestPosition[i].getCentroid(j) - chromosome[i].getCentroid(j)));
            // std::cout << "Centroid change: " << i << " " << chromosome[i].getCentroidChange(j) << std::endl;

            // std::cout << "CPU inputDomain " << inputDomain[j].first << " " << inputDomain[j].second << std::endl;

            if (abs(chromosome[i].getCentroidChange(j)) > ((inputDomain[j].second - inputDomain[j].first) * maxChange))
            {
                if (chromosome[i].getCentroidChange(j) < 0)
                    chromosome[i].setCentroidChange(j, -(inputDomain[j].second - inputDomain[j].first) * maxChange);
                else
                    chromosome[i].setCentroidChange(j, (inputDomain[j].second - inputDomain[j].first) * maxChange);
            }

            chromosome[i].setCentroid(j, chromosome[i].getCentroid(j) + chromosome[i].getCentroidChange(j));

            // chromosome[i].setWidthChange(j, chromosome[i].getWidthChange(j) + c1 * distribution(generator) * (bestParticle.chromosome[i].getWidth(j) - chromosome[i].getWidth(j)) + c2 * distribution(generator) * (bestPosition[i].getWidth(j) - chromosome[i].getWidth(j)));
             chromosome[i].setWidthChange(j, chromosome[i].getWidthChange(j) + c1 * (bestParticle.chromosome[i].getWidth(j) - chromosome[i].getWidth(j)) + c2 * (bestPosition[i].getWidth(j) - chromosome[i].getWidth(j)));
            // std::cout << "Width change CPU: " << i << " " << chromosome[i].getWidthChange(j) << std::endl;
            // std::cout << "CPU gaussianBoundaries " << gaussianBoundaries[j].first << " " << gaussianBoundaries[j].second << std::endl;
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
    }
}

/// compute fitness value
double Individual::computeFitness(const Eigen::MatrixXd &points, const Eigen::MatrixXd &expectedOutput, const Eigen::MatrixXd &polyCoef)
{
    double sum = 0;
    for (int i = 0; i < points.rows(); i++)
    {
        for (int j = 0; j < outputsNo; j++)
        {
            // point = (*points).block(i,0,1,(*points).cols())
            // sum += pow((*expected_output)(i,j)-computeValue(point,polyCoef,j),2.0);
            Eigen::MatrixXd point = points.row(i);
            double cval = computeValue(point, polyCoef, j);
            sum += fabs(expectedOutput(i, j) - cval);
        }
    }
    char ch1[20], ch2[20];
    double dd = sqrt(-1.0);
    sprintf(ch1, "%g", dd);
    sprintf(ch2, "%g", sum);
    if ((strcmp(ch1, ch2) == 0) || (std::isnan(sum)) || (sum != sum))
        sum = 1e10;
    if (sum < bestFitnessValue)
    {
        changeBestPosition();
        bestFitnessValue = sum;
    }
    fitnessValue = sum;
    return sum;
}

/// get gene value
MultivariateGaussian Individual::getGeneValue(unsigned int gene_no)
{
    return chromosome[gene_no];
}

/// set gene value
void Individual::setGeneValue(const MultivariateGaussian &gene, unsigned int geneNo)
{
    chromosome[geneNo] = gene;
}

/// save function to file
///  1 - octave/matlab style
///  2 - c++ style
void Individual::save2file(const std::string &filename, const Eigen::MatrixXd &coefficient, int type)
{
    std::ofstream ofstr;
    ofstr.open(filename, std::ofstream::out | std::ofstream::app);
    if (type == 1)
    {
        for (int k = 0; k < outputsNo; k++)
        {
            ofstr << "\nf" << k << "(...) = ";
            for (int i = 0; i < (int)gaussiansNo; i++)
            {
                double c = coefficient(i, k);
                if (c < 0)
                    ofstr << c << "*";
                else
                    ofstr << "+" << c << "*";
                chromosome[i].save2file(ofstr, type);
            }
            ofstr << "\n";
        }
    }
    if (type == 2)
    {
        for (int k = 0; k < outputsNo; k++)
        {
            ofstr << "\nf" << k << "(...) = ";
            for (int i = 0; i < (int)gaussiansNo; i++)
            {
                double c = coefficient(i, k);
                if (c < 0)
                    ofstr << c << "*";
                else
                    ofstr << "+" << c << "*";
                chromosome[i].save2file(ofstr, type);
            }
            ofstr << "\n";
        }
    }
    ofstr.close();
}
