#include "fastguidedfilter.h"
#include <cmath>

// Utility functions
static ImageF64 boxfilter(const ImageF64& I, int r)
{
    ImageF64 result(I.width, I.height, I.channels);

    for (int c = 0; c < I.channels; c++) {
        for (int i = 0; i < I.height; i++) {
            for (int j = 0; j < I.width; j++) {
                double sum = 0.0;
                int count = 0;

                // Box filter using a simple average
                for (int y = std::max(0, i - r); y <= std::min(I.height - 1, i + r); y++) {
                    for (int x = std::max(0, j - r); x <= std::min(I.width - 1, j + r); x++) {
                        sum += I.at(y, x, c);
                        count++;
                    }
                }

                result.at(i, j, c) = sum / count;
            }
        }
    }

    return result;
}

static ImageF64 convertTo(const ImageF64& mat, int depth)
{
    // This is a no-op since we're using double type
    return mat;
}

// Implementation classes
class FastGuidedFilterImpl
{
public:
    FastGuidedFilterImpl(int r, double eps, int s) : r(r), eps(eps), s(s) {}
    virtual ~FastGuidedFilterImpl() {}

    ImageF64 filter(const ImageF64& p, int depth);

protected:
    int Idepth, r, s;
    double eps;

private:
    virtual ImageF64 filterSingleChannel(const ImageF64& p) const = 0;
};

ImageF64 FastGuidedFilterImpl::filter(const ImageF64& p, int depth)
{
    ImageF64 p2 = convertTo(p, Idepth);

    // Downsample for fast calculation
    ImageF64 p2_small = p2.resize(p2.width / s, p2.height / s);

    ImageF64 result;
    if (p.channels == 1)
    {
        result = filterSingleChannel(p2_small);
    }
    else
    {
        // Split p2_small into channels
        ImageF64 pc[3];
        split(p2_small, pc, p2_small.channels);

        // Filter each channel separately
        for (int i = 0; i < p2_small.channels; ++i)
            pc[i] = filterSingleChannel(pc[i]);

        // Merge channels back
        merge(pc, p2_small.channels, result);
    }

    return result;
}

class FastGuidedFilterMono : public FastGuidedFilterImpl
{
public:
    FastGuidedFilterMono(const ImageF64& I, int r, double eps, int s);

private:
    virtual ImageF64 filterSingleChannel(const ImageF64& p) const;

private:
    ImageF64 I, origI, mean_I, var_I;
};

FastGuidedFilterMono::FastGuidedFilterMono(const ImageF64& origI, int r, double eps, int s)
    : FastGuidedFilterImpl(r, eps, s)
{
    this->origI = origI.clone();

    // Downsample for faster processing
    this->I = this->origI.resize(this->origI.width / s, this->origI.height / s);
    Idepth = ImageTypeTraits<double>::depth;

    mean_I = boxfilter(I, r);
    ImageF64 mean_II = boxfilter(I.mul(I), r);
    var_I = ImageF64(mean_II.width, mean_II.height, mean_II.channels);

    // Calculate variance: var_I = mean_II - mean_I^2
    for (int i = 0; i < var_I.height; i++) {
        for (int j = 0; j < var_I.width; j++) {
            for (int c = 0; c < var_I.channels; c++) {
                var_I.at(i, j, c) = mean_II.at(i, j, c) - mean_I.at(i, j, c) * mean_I.at(i, j, c);
            }
        }
    }
}

ImageF64 FastGuidedFilterMono::filterSingleChannel(const ImageF64& p) const
{
    ImageF64 mean_p = boxfilter(p, r);

    // Calculate mean_Ip = boxfilter(I * p)
    ImageF64 Ip(p.width, p.height, p.channels);
    for (int i = 0; i < Ip.height; i++) {
        for (int j = 0; j < Ip.width; j++) {
            Ip.at(i, j) = I.at(i, j) * p.at(i, j);
        }
    }
    ImageF64 mean_Ip = boxfilter(Ip, r);

    // Calculate covariance: cov_Ip = mean_Ip - mean_I * mean_p
    ImageF64 cov_Ip(mean_Ip.width, mean_Ip.height, mean_Ip.channels);
    for (int i = 0; i < cov_Ip.height; i++) {
        for (int j = 0; j < cov_Ip.width; j++) {
            cov_Ip.at(i, j) = mean_Ip.at(i, j) - mean_I.at(i, j) * mean_p.at(i, j);
        }
    }

    // Calculate a = cov_Ip / (var_I + eps)
    ImageF64 a(cov_Ip.width, cov_Ip.height, cov_Ip.channels);
    for (int i = 0; i < a.height; i++) {
        for (int j = 0; j < a.width; j++) {
            a.at(i, j) = cov_Ip.at(i, j) / (var_I.at(i, j) + eps);
        }
    }

    // Calculate b = mean_p - a * mean_I
    ImageF64 b(a.width, a.height, a.channels);
    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            b.at(i, j) = mean_p.at(i, j) - a.at(i, j) * mean_I.at(i, j);
        }
    }

    ImageF64 mean_a = boxfilter(a, r);
    ImageF64 mean_b = boxfilter(b, r);

    // Upsample to original size
    ImageF64 mean_a_upsampled = mean_a.resize(origI.width, origI.height);
    ImageF64 mean_b_upsampled = mean_b.resize(origI.width, origI.height);

    // Final filtering: q = mean_a * I + mean_b
    ImageF64 result(origI.width, origI.height, origI.channels);
    for (int i = 0; i < result.height; i++) {
        for (int j = 0; j < result.width; j++) {
            for (int c = 0; c < result.channels; c++) {
                result.at(i, j, c) = mean_a_upsampled.at(i, j, c) * origI.at(i, j, c) + mean_b_upsampled.at(i, j, c);
            }
        }
    }

    return result;
}

class FastGuidedFilterColor : public FastGuidedFilterImpl
{
public:
    FastGuidedFilterColor(const ImageF64& I, int r, double eps, int s);

private:
    virtual ImageF64 filterSingleChannel(const ImageF64& p) const;

private:
    ImageF64 origIchannels[3], Ichannels[3];
    ImageF64 mean_I_r, mean_I_g, mean_I_b;
    ImageF64 invrr, invrg, invrb, invgg, invgb, invbb;
};

FastGuidedFilterColor::FastGuidedFilterColor(const ImageF64& origI, int r, double eps, int s)
    : FastGuidedFilterImpl(r, eps, s)
{
    ImageF64 I = origI.clone();
    Idepth = ImageTypeTraits<double>::depth;

    // Split the original image into channels
    split(I, origIchannels, 3);

    // Downsample for faster processing
    I = I.resize(I.width / s, I.height / s);

    // Split the downsampled image into channels
    split(I, Ichannels, 3);

    // Box filter each channel
    mean_I_r = boxfilter(Ichannels[0], r);
    mean_I_g = boxfilter(Ichannels[1], r);
    mean_I_b = boxfilter(Ichannels[2], r);

    // Variance of I in each local patch
    ImageF64 var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r);
    for (int i = 0; i < var_I_rr.height; i++) {
        for (int j = 0; j < var_I_rr.width; j++) {
            var_I_rr.at(i, j) = var_I_rr.at(i, j) - mean_I_r.at(i, j) * mean_I_r.at(i, j) + eps;
        }
    }

    ImageF64 var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r);
    for (int i = 0; i < var_I_rg.height; i++) {
        for (int j = 0; j < var_I_rg.width; j++) {
            var_I_rg.at(i, j) = var_I_rg.at(i, j) - mean_I_r.at(i, j) * mean_I_g.at(i, j);
        }
    }

    ImageF64 var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r);
    for (int i = 0; i < var_I_rb.height; i++) {
        for (int j = 0; j < var_I_rb.width; j++) {
            var_I_rb.at(i, j) = var_I_rb.at(i, j) - mean_I_r.at(i, j) * mean_I_b.at(i, j);
        }
    }

    ImageF64 var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r);
    for (int i = 0; i < var_I_gg.height; i++) {
        for (int j = 0; j < var_I_gg.width; j++) {
            var_I_gg.at(i, j) = var_I_gg.at(i, j) - mean_I_g.at(i, j) * mean_I_g.at(i, j) + eps;
        }
    }

    ImageF64 var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r);
    for (int i = 0; i < var_I_gb.height; i++) {
        for (int j = 0; j < var_I_gb.width; j++) {
            var_I_gb.at(i, j) = var_I_gb.at(i, j) - mean_I_g.at(i, j) * mean_I_b.at(i, j);
        }
    }

    ImageF64 var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r);
    for (int i = 0; i < var_I_bb.height; i++) {
        for (int j = 0; j < var_I_bb.width; j++) {
            var_I_bb.at(i, j) = var_I_bb.at(i, j) - mean_I_b.at(i, j) * mean_I_b.at(i, j) + eps;
        }
    }

    // Inverse of (Sigma + eps * I)
    invrr = ImageF64(var_I_rr.width, var_I_rr.height, 1);
    invrg = ImageF64(var_I_rr.width, var_I_rr.height, 1);
    invrb = ImageF64(var_I_rr.width, var_I_rr.height, 1);
    invgg = ImageF64(var_I_rr.width, var_I_rr.height, 1);
    invgb = ImageF64(var_I_rr.width, var_I_rr.height, 1);
    invbb = ImageF64(var_I_rr.width, var_I_rr.height, 1);

    for (int i = 0; i < invrr.height; i++) {
        for (int j = 0; j < invrr.width; j++) {
            invrr.at(i, j) = var_I_gg.at(i, j) * var_I_bb.at(i, j) - var_I_gb.at(i, j) * var_I_gb.at(i, j);
            invrg.at(i, j) = var_I_gb.at(i, j) * var_I_rb.at(i, j) - var_I_rg.at(i, j) * var_I_bb.at(i, j);
            invrb.at(i, j) = var_I_rg.at(i, j) * var_I_gb.at(i, j) - var_I_gg.at(i, j) * var_I_rb.at(i, j);
            invgg.at(i, j) = var_I_rr.at(i, j) * var_I_bb.at(i, j) - var_I_rb.at(i, j) * var_I_rb.at(i, j);
            invgb.at(i, j) = var_I_rb.at(i, j) * var_I_rg.at(i, j) - var_I_rr.at(i, j) * var_I_gb.at(i, j);
            invbb.at(i, j) = var_I_rr.at(i, j) * var_I_gg.at(i, j) - var_I_rg.at(i, j) * var_I_rg.at(i, j);
        }
    }

    // Calculate determinant
    ImageF64 covDet(invrr.width, invrr.height, 1);
    for (int i = 0; i < covDet.height; i++) {
        for (int j = 0; j < covDet.width; j++) {
            covDet.at(i, j) = invrr.at(i, j) * var_I_rr.at(i, j) +
                invrg.at(i, j) * var_I_rg.at(i, j) +
                invrb.at(i, j) * var_I_rb.at(i, j);
        }
    }

    // Normalize by determinant
    for (int i = 0; i < invrr.height; i++) {
        for (int j = 0; j < invrr.width; j++) {
            double det = covDet.at(i, j);
            invrr.at(i, j) /= det;
            invrg.at(i, j) /= det;
            invrb.at(i, j) /= det;
            invgg.at(i, j) /= det;
            invgb.at(i, j) /= det;
            invbb.at(i, j) /= det;
        }
    }
}

ImageF64 FastGuidedFilterColor::filterSingleChannel(const ImageF64& p) const
{
    ImageF64 mean_p = boxfilter(p, r);

    // Calculate covariance between I and p
    ImageF64 mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    ImageF64 mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    ImageF64 mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    // Calculate covariance
    ImageF64 cov_Ip_r(mean_Ip_r.width, mean_Ip_r.height, 1);
    ImageF64 cov_Ip_g(mean_Ip_g.width, mean_Ip_g.height, 1);
    ImageF64 cov_Ip_b(mean_Ip_b.width, mean_Ip_b.height, 1);

    for (int i = 0; i < cov_Ip_r.height; i++) {
        for (int j = 0; j < cov_Ip_r.width; j++) {
            cov_Ip_r.at(i, j) = mean_Ip_r.at(i, j) - mean_I_r.at(i, j) * mean_p.at(i, j);
            cov_Ip_g.at(i, j) = mean_Ip_g.at(i, j) - mean_I_g.at(i, j) * mean_p.at(i, j);
            cov_Ip_b.at(i, j) = mean_Ip_b.at(i, j) - mean_I_b.at(i, j) * mean_p.at(i, j);
        }
    }

    // Calculate a_r, a_g, a_b
    ImageF64 a_r(invrr.width, invrr.height, 1);
    ImageF64 a_g(invrr.width, invrr.height, 1);
    ImageF64 a_b(invrr.width, invrr.height, 1);

    for (int i = 0; i < a_r.height; i++) {
        for (int j = 0; j < a_r.width; j++) {
            a_r.at(i, j) = invrr.at(i, j) * cov_Ip_r.at(i, j) +
                invrg.at(i, j) * cov_Ip_g.at(i, j) +
                invrb.at(i, j) * cov_Ip_b.at(i, j);

            a_g.at(i, j) = invrg.at(i, j) * cov_Ip_r.at(i, j) +
                invgg.at(i, j) * cov_Ip_g.at(i, j) +
                invgb.at(i, j) * cov_Ip_b.at(i, j);

            a_b.at(i, j) = invrb.at(i, j) * cov_Ip_r.at(i, j) +
                invgb.at(i, j) * cov_Ip_g.at(i, j) +
                invbb.at(i, j) * cov_Ip_b.at(i, j);
        }
    }

    // Calculate b
    ImageF64 b(a_r.width, a_r.height, 1);
    for (int i = 0; i < b.height; i++) {
        for (int j = 0; j < b.width; j++) {
            b.at(i, j) = mean_p.at(i, j) -
                a_r.at(i, j) * mean_I_r.at(i, j) -
                a_g.at(i, j) * mean_I_g.at(i, j) -
                a_b.at(i, j) * mean_I_b.at(i, j);
        }
    }

    // Apply box filter to a_r, a_g, a_b, b
    ImageF64 mean_a_r = boxfilter(a_r, r);
    ImageF64 mean_a_g = boxfilter(a_g, r);
    ImageF64 mean_a_b = boxfilter(a_b, r);
    ImageF64 mean_b = boxfilter(b, r);

    // Upsample to original resolution
    ImageF64 mean_a_r_upsampled = mean_a_r.resize(origIchannels[0].width, origIchannels[0].height);
    ImageF64 mean_a_g_upsampled = mean_a_g.resize(origIchannels[1].width, origIchannels[1].height);
    ImageF64 mean_a_b_upsampled = mean_a_b.resize(origIchannels[2].width, origIchannels[2].height);
    ImageF64 mean_b_upsampled = mean_b.resize(origIchannels[2].width, origIchannels[2].height);

    // Final filtering: q = mean_a_r * I_r + mean_a_g * I_g + mean_a_b * I_b + mean_b
    ImageF64 result(origIchannels[0].width, origIchannels[0].height, 1);
    for (int i = 0; i < result.height; i++) {
        for (int j = 0; j < result.width; j++) {
            result.at(i, j) = mean_a_r_upsampled.at(i, j) * origIchannels[0].at(i, j) +
                mean_a_g_upsampled.at(i, j) * origIchannels[1].at(i, j) +
                mean_a_b_upsampled.at(i, j) * origIchannels[2].at(i, j) +
                mean_b_upsampled.at(i, j);
        }
    }

    return result;
}

// FastGuidedFilter implementation
FastGuidedFilter::FastGuidedFilter(const ImageF64& I, int r, double eps, int s)
{
    if (I.channels != 1 && I.channels != 3) {
        throw std::invalid_argument("Image must have 1 or 3 channels");
    }

    if (I.channels == 1)
        impl_ = new FastGuidedFilterMono(I, 2 * (r / s) + 1, eps, s);
    else
        impl_ = new FastGuidedFilterColor(I, 2 * (r / s) + 1, eps, s);
}

FastGuidedFilter::~FastGuidedFilter()
{
    delete impl_;
}

ImageF64 FastGuidedFilter::filter(const ImageF64& p, int depth) const
{
    return impl_->filter(p, depth);
}

ImageF64 fastGuidedFilter(const ImageF64& I, const ImageF64& p, int r, double eps, int s, int depth)
{
    return FastGuidedFilter(I, r, eps, s).filter(p, depth);
}