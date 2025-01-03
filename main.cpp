//
//
//  @ Project : GaussianMixture library (Gaussian Mixture)
//  @ File Name : main.cpp
//  @ Date : 2011-01-20
//  @ Author : Dominik Belter
//

#include "include/GaussianMixture.h"
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>
#include <iostream>

using namespace std;

int main()
{
    try
    {
        srand((unsigned int)time(NULL)); // initialize random number generator

        /// set all required fields in config manualy or read them from file
        //GaussianMixture::Config config("configGlobal.xml");
        GaussianMixture::Config config("configGlobal.xml");

        Approximation *gaussianMixture = createGaussianApproximation(config);

        std::cout << "Start optimization\n";
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        gaussianMixture->train();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Done.\n";
        std::cout << "Optimization took = " << double(chrono::duration_cast<chrono::microseconds>(end - begin).count()) / 1000000.0 << "s" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
