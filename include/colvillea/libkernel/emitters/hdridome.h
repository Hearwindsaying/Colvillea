#pragma once

#include <owl/common/math/vec.h>

#include <libkernel/base/math.h>
#include <libkernel/base/warp.h>
#include <libkernel/base/emitter.h>
#include <libkernel/base/texture.h>
#include <libkernel/base/ray.h>
#include <libkernel/base/samplingrecord.h>

namespace colvillea
{
namespace kernel
{
// Helper macro for flattening 2d array index to
// 1d.
#define IndexFrom2DIndex(u, v, width) \
    (v) * (width) + (u)

struct Distribution1D
{
    CL_CPU_GPU CL_INLINE
    Distribution1D(uint32_t Nstep, float* func, float* cdfFunc, float funcIntegral) :
        N{Nstep}, f{func}, cdf{cdfFunc}, c{funcIntegral}
    {
        assert(this->f != nullptr && this->cdf != nullptr);
    }

    CL_CPU_GPU
    float sample1D(const float u, uint32_t cdfOffset, float* pdf, uint32_t* sampledOffset)
    {
        if (c == 0.0f)
            return 0.0f;

        *sampledOffset = upper_bound(this->cdf + cdfOffset,
                                     this->cdf + cdfOffset + this->N + 1,
                                     u) -
            (this->cdf);
        *sampledOffset = clamp(*sampledOffset - 1,
                               cdfOffset,
                               cdfOffset + this->N - 1);

        // p(x) = f(x) / c
        *pdf = this->f[*sampledOffset] / this->c;

        // Sample between searched two cdfs values.
        float du = u - this->cdf[*sampledOffset];

        // Zero division check.
        if (this->cdf[*sampledOffset + 1] - this->cdf[*sampledOffset] != 0.0f)
        {
            du /= this->cdf[*sampledOffset + 1] - this->cdf[*sampledOffset];
        }

        // Our sampledOffset should be relative to cdfOffset.
        *sampledOffset -= cdfOffset;

        return (*sampledOffset + du) / this->N;
    }

    /// This is a N step piecewise constant function.
    uint32_t N{0};

    /// 1D (piecewise constant) function to sample.
    float* f{nullptr};

    /// CDF of f.
    float* cdf{nullptr};

    /// c = \int f(x)dx.
    float c{0.0f};
};

struct Distribution2D
{
    CL_CPU_GPU CL_INLINE
    Distribution2D(float*   _pCondV_c,
                   float*   _cdfpCondV,
                   float*   _pV_c,
                   float*   _cdfpV,
                   uint32_t _nu,
                   uint32_t _nv) :
        pCondV_c{_pCondV_c},
        cdfpCondV{_cdfpCondV},
        pV_c{_pV_c},
        cdfpV{_cdfpV},
        nu{_nu},
        nv{_nv}
    {
        assert(this->pCondV_c && this->cdfpCondV && this->pV_c && this->cdfpV);
        assert(nu > 0 && nv > 0);
    }

    CL_CPU_GPU
    vec2f sample2D(const vec2f& random,
                   float*       pdf)
    {
        vec2f    tmpPdf{0.0f};
        uint32_t sampledOffset{0};

        Distribution1D dist1d_marginal{this->nv,
                                       this->pV_c, this->cdfpV, this->pV_c[nv]};

        // Sample marginal pdf.
        vec2f sampledPt{0.0f};
        sampledPt.y = dist1d_marginal.sample1D(random.y, 0, &tmpPdf.y, &sampledOffset);

        // Sample conditional v.
        Distribution1D dist1d_cond{this->nu,
                                   this->pCondV_c,
                                   this->cdfpCondV,
                                   this->pCondV_c[IndexFrom2DIndex(this->nu, sampledOffset, this->nu + 1)]};

        sampledPt.x = dist1d_cond.sample1D(random.x,
                                           sampledOffset * (nu + 1),
                                           &tmpPdf.x,
                                           &sampledOffset);

        *pdf = tmpPdf.x * tmpPdf.y;

        return sampledPt;
    }

    CL_CPU_GPU CL_INLINE float pdf2D(const vec2f& sample)
    {
        if (sample.x < 0.f || sample.y < 0.f)
            return 0.0f;

        uint32_t tildeU = clamp(static_cast<uint32_t>(this->nu * sample.x), 0u, this->nu - 1);
        uint32_t tildeV = clamp(static_cast<uint32_t>(this->nv * sample.y), 0u, this->nv - 1);

        // The \c for the whole step 2d function is stored in pV_c's last term.
        return this->pCondV_c[IndexFrom2DIndex(tildeU, tildeV, this->nu + 1)] / this->pV_c[this->nv];
    }


    uint32_t nu{0}, nv{0};

    float* pCondV_c{nullptr};
    float* cdfpCondV{nullptr};
    float* pV_c{nullptr};
    float* cdfpV{nullptr};
};

class HDRIDome
{
public:
    HDRIDome() = default;

    /**
     * \brief.
     *    Constructor for kernel HDRIDome. All pointer parameters are device
     * pointers.   
     * 
     * \param hdri
     * \param pUcondV_c
     * \param CDFpUcondV
     * \param pV_c
     * \param CDFpV
     * \return 
     */
    CL_CPU_GPU CL_INLINE HDRIDome(const Texture& hdri,
                                  const vec2ui&  textureResolution,
                                  float*         pUcondV_c,
                                  float*         CDFpUcondV,
                                  float*         pV_c,
                                  float*         CDFpV) :
        m_hdriTex{hdri},
        m_textureResolution{textureResolution},
        m_pUcondV_c{pUcondV_c},
        m_CDFpUcondV{CDFpUcondV},
        m_pV_c{pV_c},
        m_CDFpV{CDFpV}
    {}

#ifdef __CUDACC__
    /// <summary>
    /// Sample EDF of the dome emitter.
    /// </summary>
    /// <param name="pDirectRec"></param>
    /// <param name="sample">uniform 2D samples.</param>
    /// <returns></returns>
    CL_GPU vec3f sampleDirect(DirectSamplingRecord* pDirectRec, const vec2f& sample) const
    {
#    define IMPORTANCE_SAMPLING
#    ifdef IMPORTANCE_SAMPLING
        // Construct dist2D.
        Distribution2D dist2D{this->m_pUcondV_c,
                              this->m_CDFpUcondV,
                              this->m_pV_c,
                              this->m_CDFpV,
                              this->m_textureResolution.x,
                              this->m_textureResolution.y};


        //printf("m_pUcondV_c:%f %f\n", this->m_pV_c[this->m_textureResolution.y], this->m_pUcondV_c[this->m_textureResolution.x]);


        // Sample according to dist2D and get uv, pdf.
        vec2f uv = dist2D.sample2D(sample, &pDirectRec->pdf);

        // Convert to spherical coords.
        // This mapping follows our skybox query convention.
        // See also: \directionToUVCoords().
        float theta = M_PIf * uv.y;
        float phi   = -2 * M_PIf * (uv.x - 0.5f);

        float sinTheta = sin(theta), cosTheta = cos(theta);
        float sinPhi = sin(phi), cosPhi = cos(phi);

        /*printf("%f %f\n", sinTheta, sinPhi);*/

        if (sinTheta == 0.0f)
        {
            pDirectRec->pdf = 0.0f;
            return vec3f{0.f};
        }

        // Convert to Cartesian direction.
        pDirectRec->direction = HDRIDome::sphericalCoordsToCartesian(sinTheta, cosTheta, sinPhi, cosPhi);

        // Compute pdf.
        pDirectRec->measure = SamplingMeasure::SolidAngle;
        pDirectRec->pdf /= 2.0f * M_PIf * M_PIf * sinTheta;

        return vec3f{this->m_hdriTex.eval2D(uv)};
#    endif // IMPORTANCE_SAMPLING

        /*pDirectRec->direction = warp::squareToCosineHemisphere(sample);
        pDirectRec->pdf       = warp::squareToCosineHemispherePdf(pDirectRec->direction);

        return vec3f{this->m_hdriTex.eval2D(HDRIDome::directionToUVCoords(pDirectRec->direction))};*/
    }
#endif

    /// <summary>
    /// Return pdf of EDF sampling.
    /// </summary>
    /// <param name="dRec"></param>
    /// <returns></returns>
    CL_CPU_GPU CL_INLINE float pdfDirect(const DirectSamplingRecord& dRec) const
    {
#define IMPORTANCE_SAMPLING
#ifdef IMPORTANCE_SAMPLING
        // Construct dist2D.
        Distribution2D dist2D{this->m_pUcondV_c,
                              this->m_CDFpUcondV,
                              this->m_pV_c,
                              this->m_CDFpV,
                              this->m_textureResolution.x,
                              this->m_textureResolution.y};

        // Convert ray direction to spherical coords.
        vec2f  phi_theta = HDRIDome::directionToSphericalCoords(dRec.direction);
        float& phi       = phi_theta.x;
        float& theta     = phi_theta.y;

        float sinTheta = sin(theta);
        // Check degenerate case.
        if (sinTheta == 0.0f)
            return 0.0f;

        // directionToUVCoords().
        vec2f uv{phi * -M_1_PIf * 0.5f + 0.5f, theta * M_1_PIf};

        assert(dRec.measure == SamplingMeasure::SolidAngle);

        return dist2D.pdf2D(uv) / (2.0f * M_PIf * M_PIf * sinTheta);
#endif // IMPORTANCE_SAMPLING

        //return warp::squareToCosineHemispherePdf(dRec.direction);
    }

    /**
     * \brief.
     *    Evaluate environment EDF with ray.   
     * 
     * \param ray
     *    Ray in the world space.
     * 
     * \return 
     */
#ifdef __CUDACC__
    CL_GPU CL_INLINE
        vec3f
        evalEnvironment(const Ray& ray) const
    {
        vec2f uv = directionToUVCoords(vec3f{ray.d.x, ray.d.y, ray.d.z});
        return vec3f{this->m_hdriTex.eval2D(uv)};
    }
#endif


#ifdef __CUDACC__
    /**
     * \brief.
     *    Launch prefiltering process. This is the first HDRIDome preprocessing 
     * pass. It looks up the skybox texture and multiply by sinTheta and fills 
     * m_pUcondV_c (without the c part).
     * 
     * \param jobID
     * \param domeTextureDim
     */
    CL_GPU CL_INLINE void prefiltering(const vec2ui& jobID, const vec2ui& domeTextureDim)
    {
        if (jobID.x >= domeTextureDim.x ||
            jobID.y >= domeTextureDim.y)
        {
            return;
        }

        const vec2f uv = (vec2f{jobID} + 0.5f) / vec2f{domeTextureDim};

        const float sinTheta = sin(M_PIf * (jobID.y + 0.5f) / domeTextureDim.y);

        this->m_pUcondV_c[IndexFrom2DIndex(jobID.x, jobID.y, domeTextureDim.x + 1)] = sinTheta * linearToLuminance(vec3f{this->m_hdriTex.eval2D(uv)});
    }
#endif

    /**
     * \brief.
     *    Preprocessing pCondV.
     * 
     * \return 
     */
    CL_CPU_GPU CL_INLINE void preprocessPCondV(const vec2ui& jobID, const vec2ui& domeTextureDim)
    {
        if (jobID.y >= domeTextureDim.y)
        {
            return;
        }

        // We have already fill in values for m_pUcondV_c the pUcondV part.
        // We could now compute the CDF of the pUcondV.

        // We first do the accumulation.
        this->m_CDFpUcondV[IndexFrom2DIndex(0, jobID.y, domeTextureDim.x + 1)] = 0.0f;
        for (int i = 1; i < domeTextureDim.x + 1; ++i)
        {
            this->m_CDFpUcondV[IndexFrom2DIndex(i, jobID.y, domeTextureDim.x + 1)] =
                this->m_CDFpUcondV[IndexFrom2DIndex(i - 1, jobID.y, domeTextureDim.x + 1)] +
                this->m_pUcondV_c[IndexFrom2DIndex(i - 1, jobID.y, domeTextureDim.x + 1)] / domeTextureDim.x;
        }

        // The accumulation part which by the way gives us the function integral \c.
        float c = this->m_CDFpUcondV[IndexFrom2DIndex(domeTextureDim.x, jobID.y, domeTextureDim.x + 1)];
        // So that we can fill in the c part in m_pUcondV.
        this->m_pUcondV_c[IndexFrom2DIndex(domeTextureDim.x, jobID.y, domeTextureDim.x + 1)] = c;

        // Finally divide cdf by c gives us the correct cdf.

        // Check for integrity.
        if (c <= 0.0f)
        {
            // Our pUcondV gives us the all black values which is valid in reality. We should deal with this.
            for (int i = 0; i < domeTextureDim.x; ++i)
            {
                // Force p(u|v) to be zero.
                this->m_pUcondV_c[IndexFrom2DIndex(i, jobID.y, domeTextureDim.x + 1)] = 0.0f;

                this->m_CDFpUcondV[IndexFrom2DIndex(i, jobID.y, domeTextureDim.x + 1)] = static_cast<float>(i) / domeTextureDim.x;
            }

            // The last cdf should be 1.0
            this->m_CDFpUcondV[IndexFrom2DIndex(domeTextureDim.x, jobID.y, domeTextureDim.x + 1)] = 1.0f;
            this->m_pUcondV_c[IndexFrom2DIndex(domeTextureDim.x, jobID.y, domeTextureDim.x + 1)]  = 0.0f;
        }
        else
        {
            for (int i = 0; i < domeTextureDim.x + 1; ++i)
            {
                this->m_CDFpUcondV[IndexFrom2DIndex(i, jobID.y, domeTextureDim.x + 1)] /= c;
            }
        }

        // Note that c is exactly the marginal function p(v). (after leaving out the normalization factor).
        this->m_pV_c[jobID.y] = c;
    }

    CL_CPU_GPU CL_INLINE void preprocessPV(const vec2ui& jobID, const vec2ui& domeTextureDim)
    {
        if (jobID.x > domeTextureDim.y)
        {
            return;
        }

        // First compute CDF w/o divided by c. This is indeed a
        // prefix sum but we do not need to optimize since it
        // is efficient already.
        // Compute c BTW.
        float sum = 0.0f, c = 0.0f;
        for (int i = 1; i <= domeTextureDim.y; ++i)
        {
            if (i <= jobID.x)
            {
                sum += this->m_pV_c[i - 1];
            }
            c += this->m_pV_c[i - 1];
        }

        // We need to divided by N to get the CDF and c!
        this->m_CDFpV[jobID.x] = sum / domeTextureDim.y;
        c /= domeTextureDim.y;

        // If current thread is the last term, we should write c.
        if (jobID.x == domeTextureDim.y)
        {
            this->m_pV_c[domeTextureDim.y] = c;
            printf("this->m_pV_c[domeTextureDim.y] = %f\n", this->m_pV_c[domeTextureDim.y]);
        }

        //printf("jobId:%d\n", jobID.x);

        // Integrity check.
        if (c <= 0.0f)
        {
            this->m_pV_c[jobID.x]  = 0.0f;
            this->m_CDFpV[jobID.x] = static_cast<float>(jobID.x) / domeTextureDim.y;
        }
        else
        {
            // Divide cdf by c to get the correct cdf.
            this->m_CDFpV[jobID.x] /= c;
        }
    }

#undef IndexFrom2DIndex

private:
    /**
     * \brief.
     *    Map ray direction in Cartesian coordinates to
     * spherical coordinates. We use RHS-Y up coordinate
     * frame where phi: [-PI,PI] and theta: [0, PI].
     *    Phi is the angle between +Z axis and projected
     * ray direction in +XZ plane.
     * 
     * \param dir
     * \return 
     *     Spherical coords.
     */
    CL_CPU_GPU CL_INLINE static vec2f
    directionToSphericalCoords(const vec3f& dir)
    {
        return vec2f{atan2(dir.x, dir.z), acos(dir.y)};
    }

    /**
      * \brief
      *    Map unit ray direction to uv coordinates for skybox sampling.
      * We assume ray direction is in the world space (RHS-Y up), where
      * phi is the angle starting from +Z (a bit different from \Frame's
      * coordinate convention). 
      * 
      * \remarks
      *     Dir: -Z    +X    +Z    -X    -Z 
      *      u:   0   0.25   0.5   0.75   1
      * 
      * \param dir
      *    Unit ray direction in the world space.
      * 
      * \return 
      *    UV coordinates from [0, 1].
      */
    CL_CPU_GPU CL_INLINE static vec2f directionToUVCoords(const vec3f& dir)
    {
        return vec2f{atan2(dir.x, dir.z) * -M_1_PIf * 0.5f + 0.5f,
                     acos(dir.y) * M_1_PIf};
    }

    CL_CPU_GPU CL_INLINE static vec3f sphericalCoordsToCartesian(const float sintheta,
                                                                 const float costheta,
                                                                 const float sinphi,
                                                                 const float cosphi)
    {
        return vec3f{sintheta * sinphi, costheta, sintheta * cosphi};
    }


private:
    Texture m_hdriTex;

    /// TODO: Maybe we want to keep a size?

    vec2ui m_textureResolution{0};

    // Sampling Related:
    /// p(u|v) is a 1D distribution given v. c is function integral of p(u|v) given v.
    /// This is a flatten 2D array.
    /// size: (nu+1)*nv
    float* m_pUcondV_c{nullptr};

    /// CDF of 1D distribution function p(u|v) given v.
    /// size: (nu+1)*nv (the last term for each row is CDF==1.0)
    float* m_CDFpUcondV{nullptr};

    /// Marginal p(v).
    /// size: nv+1
    float* m_pV_c{nullptr};

    /// CDF of marginal p(v).
    /// size: nv+1 (the last term is CDF==1.0)
    float* m_CDFpV{nullptr};
};
} // namespace kernel
} // namespace colvillea
