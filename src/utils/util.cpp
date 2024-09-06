#include "../include/util.hpp"

void PrintVec(std::vector <float> vec)
{
    int nelements = std::min(10, (int)vec.size());
    for (int i=0; i<nelements-1; i++)
    {
        std::cout << vec[i] << "\t";
    }
    std::cout << vec[nelements-1] << "\n";
}