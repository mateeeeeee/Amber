#pragma once
#include "Math.cuh"
#include "ONB.cuh"
#include "PRNG.cuh"
#include "Color.cuh"

//https://github.com/Twinklebear/ChameleonRT/blob/master/backends/optix/disney_bsdf.h

__device__ Bool SameHemisphere(Float3 const& w_o, Float3 const& w_i, Float3 const& n)
{
	return dot(w_o, n) * dot(w_i, n) > 0.0f;
}

__device__ Float3 CosSampleHemisphere(Float2 u)
{
	Float phi = 2.0f * M_PI * u.x;
	Float cos_theta = sqrt(u.y);
	Float sin_theta = sqrt(1.0f - u.y);
	Float x = cos(phi) * sin_theta;
	Float y = sin(phi) * sin_theta;
	Float z = cos_theta;
	return MakeFloat3(x, y, z);
}

__device__ Float Pow2(Float x) 
{
	return x * x;
}

__device__ Float3 SphericalDir(Float sin_theta, Float cos_theta, Float phi)
{
    return MakeFloat3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

__device__ Float PowerHeuristic(Float n_f, Float pdf_f, Float n_g, Float pdf_g)
{
    Float f = n_f * pdf_f;
    Float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

__device__ Float SchlickWeight(Float cos_theta)
{
    return pow(clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
}

__device__ Float FresnelDielectric(Float cos_theta_i, Float eta_i, Float eta_t)
{
    Float g = Pow2(eta_t) / Pow2(eta_i) - 1.0f + Pow2(cos_theta_i);
    if (g < 0.0f) 
    {
        return 1.0f;
    }
    return 0.5f * Pow2(g - cos_theta_i) / Pow2(g + cos_theta_i) *
        (1.0f + Pow2(cos_theta_i * (g + cos_theta_i) - 1.0f) / Pow2(cos_theta_i * (g - cos_theta_i) + 1.0f));
}

// D_GTR1: Generalized Trowbridge-Reitz with gamma=1
__device__ Float Gtr1(Float cos_theta_h, Float alpha)
{
    Float alpha_sqr = alpha * alpha;
    Float result = M_INV_PI * (alpha_sqr - 1.0f) /
        (log(alpha_sqr) * (1.0f + (alpha_sqr - 1.0f) * cos_theta_h * cos_theta_h));

    result = alpha >= 1.0f ? M_INV_PI : result;
    return result;
}

// D_GTR2: Generalized Trowbridge-Reitz with gamma=2
__device__ Float Gtr2(Float cos_theta_h, Float alpha)
{
    Float alpha_sqr = alpha * alpha;
    return M_INV_PI * alpha_sqr / Pow2(1.0f + (alpha_sqr - 1.0f) * cos_theta_h * cos_theta_h);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
__device__ Float Gtr2Aniso(Float h_dot_n, Float h_dot_x, Float h_dot_y, Float2 alpha)
{
    return M_INV_PI /
        (alpha.x * alpha.y *
            Pow2(Pow2(h_dot_x / alpha.x) + Pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n));
}

__device__ Float SmithShadowingGGX(Float n_dot_o, Float alpha_g)
{
    Float a = alpha_g * alpha_g;
    Float b = n_dot_o * n_dot_o;
    return 1.0f / (n_dot_o + sqrt(a + b - a * b));
}

__device__ Float SmithShadowingGGXAniso(Float n_dot_o,
    Float o_dot_x,
    Float o_dot_y,
    Float2 alpha)
{
    return 1.0f / (n_dot_o + sqrt(Pow2(o_dot_x * alpha.x) + Pow2(o_dot_y * alpha.y) + Pow2(n_dot_o)));
}

__device__ Float3 SampleLambertianDir(Float3 const& n,
    Float3 const& v_x,
    Float3 const& v_y,
    Float2 const& s)
{
    const Float3 hemi_dir = CosSampleHemisphere(s);
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

// Sample the microfacet normal vectors for the various microfacet distributions
__device__ Float3 SampleGtr1H(Float3 const& n, Float3 const& v_x, Float3 const& v_y, Float alpha, Float2 const& s)
{
    Float phi_h = 2.0f * M_PI * s.x;
    Float alpha_sqr = alpha * alpha;
    Float cos_theta_h_sqr = (1.0f - pow(alpha_sqr, 1.0f - s.y)) / (1.0f - alpha_sqr);
    Float cos_theta_h = sqrt(cos_theta_h_sqr);
    Float sin_theta_h = 1.0f - cos_theta_h_sqr;
    Float3 hemi_dir = normalize(SphericalDir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ Float3 SampleGtr2H(Float3 const& n, Float3 const& v_x, Float3 const& v_y, Float alpha, Float2 const& s)
{
    Float phi_h = 2.0f * M_PI * s.x;
    Float cos_theta_h_sqr = (1.0f - s.y) / (1.0f + (alpha * alpha - 1.0f) * s.y);
    Float cos_theta_h = sqrt(cos_theta_h_sqr);
    Float sin_theta_h = 1.0f - cos_theta_h_sqr;
    Float3 hemi_dir = normalize(SphericalDir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ Float3 SampleGtr2AnisoH(Float3 const& n,
    Float3 const& v_x,
    Float3 const& v_y,
	Float2 const& alpha,
	Float2 const& s)
{
    Float x = 2.0f * M_PI * s.x;
    Float3 w_h = sqrt(s.y / (1.0f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
    return normalize(w_h);
}

__device__ Float LamberitanPdf(Float3 const& w_i, Float3 const& n)
{
    Float d = dot(w_i, n);
    if (d > 0.0f) 
    {
        return d * M_INV_PI;
    }
    return 0.0f;
}

__device__ Float Gtr1Pdf(Float3 const& w_o, Float3 const& w_i, Float3 const& n, Float alpha)
{
    Float result_scale = SameHemisphere(w_o, w_i, n) ? 1.0f : 0.0f;
    Float3 w_h = normalize(w_i + w_o);
    Float cos_theta_h = dot(n, w_h);
    Float d = Gtr1(cos_theta_h, alpha);
    return result_scale * d * cos_theta_h / (4.0f * dot(w_o, w_h));
}

__device__ Float Gtr2Pdf(Float3 const& w_o, Float3 const& w_i, Float3 const& n, Float alpha)
{
    Float result_scale = SameHemisphere(w_o, w_i, n) ? 1.0f : 0.0f;
    Float3 w_h = normalize(w_i + w_o);
    Float cos_theta_h = dot(n, w_h);
    Float d = Gtr2(cos_theta_h, alpha);
    return result_scale * d * cos_theta_h / (4.0f * dot(w_o, w_h));
}

__device__ Float Gtr2TransmissionPdf(
    Float3 const& w_o, Float3 const& w_i, Float3 const& n, Float alpha, Float ior)
{
    if (SameHemisphere(w_o, w_i, n)) 
    {
        return 0.0f;
    }
    Bool entering = dot(w_o, n) > 0.0f;
    Float eta_o = entering ? 1.0f : ior;
    Float eta_i = entering ? ior : 1.0f;
    Float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
    Float cos_theta_h = fabs(dot(n, w_h));
    Float i_dot_h = dot(w_i, w_h);
    Float o_dot_h = dot(w_o, w_h);
    Float d = Gtr2(cos_theta_h, alpha);
    Float dwh_dwi = o_dot_h * Pow2(eta_o) / Pow2(eta_o * o_dot_h + eta_i * i_dot_h);
    return d * cos_theta_h * fabs(dwh_dwi);
}

__device__ Float Gtr2AnisoPdf(Float3 const& w_o,
    Float3 const& w_i,
    Float3 const& n,
    Float3 const& v_x,
    Float3 const& v_y,
    Float2 alpha)
{
    if (!SameHemisphere(w_o, w_i, n))
    {
        return 0.0f;
    }
    Float3 w_h = normalize(w_i + w_o);
    Float cos_theta_h = dot(n, w_h);
    Float d = Gtr2Aniso(cos_theta_h, fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
    return d * cos_theta_h / (4.0f * dot(w_o, w_h));
}

struct EvaluatedMaterial
{
	Float3 base_color;
	Float  metallic;
	Float3 emissive;
	Float  specular;
	Float3 normal;
	Float  roughness;
	Float  ao;
	Float  specular_tint;
	Float  anisotropy;

	Float  sheen;
	Float  sheen_tint;
	Float  clearcoat;
	Float  clearcoat_gloss;

	Float  ior;
	Float  specular_transmission;
};
