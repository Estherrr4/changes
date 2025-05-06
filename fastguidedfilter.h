#pragma once
#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include "custom_types.h"

// Forward declaration
class FastGuidedFilterImpl;

class FastGuidedFilter
{
public:
    FastGuidedFilter(const ImageF64& I, int r, double eps, int s);
    ~FastGuidedFilter();

    ImageF64 filter(const ImageF64& p, int depth = -1) const;

private:
    FastGuidedFilterImpl* impl_;
};

// Function declaration only
ImageF64 fastGuidedFilter(const ImageF64& I, const ImageF64& p, int r, double eps, int s = 1, int depth = -1);

#endif // GUIDED_FILTER_H