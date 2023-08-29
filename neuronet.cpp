#include "neuronet.h"

namespace neuronet {

    double fRand(double fMin, double fMax)
    {
        double f = (double)rand() / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }



    double SigmaFunction(double d) {
        return 1 / (1 + std::exp(-d));

    }



}