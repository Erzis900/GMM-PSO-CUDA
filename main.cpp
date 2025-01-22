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
#include <fstream>

using namespace std;

int main()
{
    try
    {
        /// set all required fields in config manualy or read them from file
        //GaussianMixture::Config config("configGlobal.xml");
        srand((unsigned int)time(NULL)); // initialize random number generator
        int runs = 100;
        float totalMemcpyTime = 0;
        std::vector<float> memcpyTimes;
        for (int i = 0; i < runs + 1; i++)
        {
            GaussianMixture::Config config("configGlobal.xml");

            // Approximation *gaussianMixture = createGaussianApproximation(config);
            std::unique_ptr<Approximation> gaussianMixture = approximator::createGaussianApproximation(config);

            std::cout << "Start optimization\n";

            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            // gaussianMixture->train();
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            if(i != 0)
            {
                memcpyTimes.push_back(gaussianMixture->getMemcpyTime());
            }
            
            std::cout << "Done.\n";
            std::cout << "Optimization took = " << double(chrono::duration_cast<chrono::microseconds>(end - begin).count()) / 1000000.0 << "s" << std::endl;
        }

        float totalTime = 0;
        for (auto &v : memcpyTimes)
        {
            std::cout << v << ", ";
            totalTime += v;
        }

        float avgMemcpyTime = totalTime / runs;
        std::cout << "AVG memcpy: " << avgMemcpyTime << std::endl; 

        float stdVar = calc_std_var(memcpyTimes, avgMemcpyTime);
        std::cout << "std dev memcpy: " << stdVar << std::endl;

        std::ofstream memcpyTimeFile("../../memcpy_times.csv", std::ios::app);
        memcpyTimeFile << 5 << "," << avgMemcpyTime << "," << stdVar << "\n";
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
