#pragma once
#include "Math.cuh"
#include "Sampling.cuh"
#include "Random.cuh"
#include "Color.cuh"

//https://github.com/Twinklebear/ChameleonRT/blob/master/backends/optix/disney_bsdf.h

struct DisneyMaterial
{
	float3 base_color;
	float  metallic;
	float3 emissive;
	float  specular;
	float3 normal;
	float  roughness;
	float  ao;
	float  specular_tint;
	float  anisotropy;

	float  sheen;
	float  sheen_tint;
	float  clearcoat;
	float  clearcoat_gloss;

	float  ior;
	float  specular_transmission;
};

__device__ bool SameHemisphere(const float3& w_o, const float3& w_i, const float3& n)
{
	return dot(w_o, n) * dot(w_i, n) > 0.f;
}

__device__ float3 CosSampleHemisphere(float2 u)
{
	float phi = 2.0f * M_PI * u.x;
	float cosTheta = sqrt(u.y);
	float sinTheta = sqrt(1.0f - u.y);
	float x = cos(phi) * sinTheta;
	float y = sin(phi) * sinTheta;
	float z = cosTheta;
	return make_float3(x, y, z);
}

__device__ float pow2(float x) {
	return x * x;
}

__device__ float3 SphericalDir(float sin_theta, float cos_theta, float phi)
{
    return make_float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

__device__ float PowerHeuristic(float n_f, float pdf_f, float n_g, float pdf_g)
{
    float f = n_f * pdf_f;
    float g = n_g * pdf_g;
    return (f * f) / (f * f + g * g);
}

__device__ float SchlickWeight(float cos_theta)
{
    return pow(clamp(1.f - cos_theta, 0.0f, 1.0f), 5.f);
}

// Complete Fresnel Dielectric computation, for transmission at ior near 1
// they mention having issues with the Schlick approximation.
// eta_i: material on incident side's ior
// eta_t: material on transmitted side's ior
__device__ float FresnelDielectric(float cos_theta_i, float eta_i, float eta_t)
{
    float g = pow2(eta_t) / pow2(eta_i) - 1.f + pow2(cos_theta_i);
    if (g < 0.f) {
        return 1.f;
    }
    return 0.5f * pow2(g - cos_theta_i) / pow2(g + cos_theta_i) *
        (1.f + pow2(cos_theta_i * (g + cos_theta_i) - 1.f) /
            pow2(cos_theta_i * (g - cos_theta_i) + 1.f));
}

// D_GTR1: Generalized Trowbridge-Reitz with gamma=1
// Burley notes eq. 4
__device__ float Gtr1(float cos_theta_h, float alpha)
{
    float alpha_sqr = alpha * alpha;
    float result = M_INV_PI * (alpha_sqr - 1.f) /
        (log(alpha_sqr) * (1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h));

    result = alpha >= 1.f ? M_INV_PI : result;

    return result;
}

// D_GTR2: Generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 8
__device__ float Gtr2(float cos_theta_h, float alpha)
{
    float alpha_sqr = alpha * alpha;
    return M_INV_PI * alpha_sqr / pow2(1.f + (alpha_sqr - 1.f) * cos_theta_h * cos_theta_h);
}

// D_GTR2 Anisotropic: Anisotropic generalized Trowbridge-Reitz with gamma=2
// Burley notes eq. 13
__device__ float Gtr2Aniso(float h_dot_n, float h_dot_x, float h_dot_y, float2 alpha)
{
    return M_INV_PI /
        (alpha.x * alpha.y *
            pow2(pow2(h_dot_x / alpha.x) + pow2(h_dot_y / alpha.y) + h_dot_n * h_dot_n));
}

__device__ float SmithShadowingGGX(float n_dot_o, float alpha_g)
{
    float a = alpha_g * alpha_g;
    float b = n_dot_o * n_dot_o;
    return 1.f / (n_dot_o + sqrt(a + b - a * b));
}

__device__ float SmithShadowingGGXAniso(float n_dot_o,
    float o_dot_x,
    float o_dot_y,
    float2 alpha)
{
    return 1.f /
        (n_dot_o + sqrt(pow2(o_dot_x * alpha.x) + pow2(o_dot_y * alpha.y) + pow2(n_dot_o)));
}

// Sample a reflection direction the hemisphere oriented along n and spanned by v_x, v_y using
// the random samples in s
__device__ float3 SampleLambertianDir(const float3& n,
    const float3& v_x,
    const float3& v_y,
    const float2& s)
{
    const float3 hemi_dir = CosSampleHemisphere(s);
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

// Sample the microfacet normal vectors for the various microfacet distributions
__device__ float3 SampleGtr1H(
    const float3& n, const float3& v_x, const float3& v_y, float alpha, const float2& s)
{
    float phi_h = 2.f * M_PI * s.x;
    float alpha_sqr = alpha * alpha;
    float cos_theta_h_sqr = (1.f - pow(alpha_sqr, 1.f - s.y)) / (1.f - alpha_sqr);
    float cos_theta_h = sqrt(cos_theta_h_sqr);
    float sin_theta_h = 1.f - cos_theta_h_sqr;
    float3 hemi_dir = normalize(SphericalDir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ float3 SampleGtr2H(
    const float3& n, const float3& v_x, const float3& v_y, float alpha, const float2& s)
{
    float phi_h = 2.f * M_PI * s.x;
    float cos_theta_h_sqr = (1.f - s.y) / (1.f + (alpha * alpha - 1.f) * s.y);
    float cos_theta_h = sqrt(cos_theta_h_sqr);
    float sin_theta_h = 1.f - cos_theta_h_sqr;
    float3 hemi_dir = normalize(SphericalDir(sin_theta_h, cos_theta_h, phi_h));
    return hemi_dir.x * v_x + hemi_dir.y * v_y + hemi_dir.z * n;
}

__device__ float3 SampleGtr2AnisoH(const float3& n,
    const float3& v_x,
    const float3& v_y,
    const float2& alpha,
    const float2& s)
{
    float x = 2.f * M_PI * s.x;
    float3 w_h =
        sqrt(s.y / (1.f - s.y)) * (alpha.x * cos(x) * v_x + alpha.y * sin(x) * v_y) + n;
    return normalize(w_h);
}

__device__ float LamberitanPdf(const float3& w_i, const float3& n)
{
    float d = dot(w_i, n);
    if (d > 0.f) {
        return d * M_INV_PI;
    }
    return 0.f;
}

__device__ float Gtr1Pdf(const float3& w_o, const float3& w_i, const float3& n, float alpha)
{
    float result_scale = SameHemisphere(w_o, w_i, n) ? 1.f : 0.f;
    float3 w_h = normalize(w_i + w_o);
    float cos_theta_h = dot(n, w_h);
    float d = Gtr1(cos_theta_h, alpha);
    return result_scale * d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ float Gtr2Pdf(const float3& w_o, const float3& w_i, const float3& n, float alpha)
{
    float result_scale = SameHemisphere(w_o, w_i, n) ? 1.f : 0.f;
    float3 w_h = normalize(w_i + w_o);
    float cos_theta_h = dot(n, w_h);
    float d = Gtr2(cos_theta_h, alpha);
    return result_scale * d * cos_theta_h / (4.f * dot(w_o, w_h));
}
__device__ float Gtr2TransmissionPdf(
    const float3& w_o, const float3& w_i, const float3& n, float alpha, float ior)
{
    if (SameHemisphere(w_o, w_i, n)) 
    {
        return 0.f;
    }
    bool entering = dot(w_o, n) > 0.f;
    float eta_o = entering ? 1.f : ior;
    float eta_i = entering ? ior : 1.f;
    float3 w_h = normalize(w_o + w_i * eta_i / eta_o);
    float cos_theta_h = fabs(dot(n, w_h));
    float i_dot_h = dot(w_i, w_h);
    float o_dot_h = dot(w_o, w_h);
    float d = Gtr2(cos_theta_h, alpha);
    float dwh_dwi = o_dot_h * pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);
    return d * cos_theta_h * fabs(dwh_dwi);
}

__device__ float Gtr2AnisoPdf(const float3& w_o,
    const float3& w_i,
    const float3& n,
    const float3& v_x,
    const float3& v_y,
    const float2 alpha)
{
    if (!SameHemisphere(w_o, w_i, n)) {
        return 0.f;
    }
    float3 w_h = normalize(w_i + w_o);
    float cos_theta_h = dot(n, w_h);
    float d = Gtr2Aniso(cos_theta_h, fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
    return d * cos_theta_h / (4.f * dot(w_o, w_h));
}

__device__ float3 DisneyDiffuse(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i)
{
    float3 w_h = normalize(w_i + w_o);
    float n_dot_o = fabs(dot(w_o, n));
    float n_dot_i = fabs(dot(w_i, n));
    float i_dot_h = dot(w_i, w_h);
    float fd90 = 0.5f + 2.f * mat.roughness * i_dot_h * i_dot_h;
    float fi = SchlickWeight(n_dot_i);
    float fo = SchlickWeight(n_dot_o);
    return mat.base_color * M_INV_PI * lerp(1.f, fd90, fi) * lerp(1.f, fd90, fo);
}

__device__ float3 DisneyMicrofacetIsotropic(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i)
{
    float3 w_h = normalize(w_i + w_o);
    float lum = Luminance(mat.base_color);
    float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
    float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint),
        mat.base_color,
        mat.metallic);

    float alpha = max(0.001f, mat.roughness * mat.roughness);
    float d = Gtr2(dot(n, w_h), alpha);
    float3 f = lerp(spec, make_float3(1.f), SchlickWeight(dot(w_i, w_h)));
    float g =
        SmithShadowingGGX(dot(n, w_i), alpha) * SmithShadowingGGX(dot(n, w_o), alpha);
    return d * f * g;
}

__device__ float3 DisneyMicrofacetTransmissionIsotropic(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i)
{
    float o_dot_n = dot(w_o, n);
    float i_dot_n = dot(w_i, n);
    if (o_dot_n == 0.f || i_dot_n == 0.f) {
        return make_float3(0.f);
    }
    bool entering = o_dot_n > 0.f;
    float eta_o = entering ? 1.f : mat.ior;
    float eta_i = entering ? mat.ior : 1.f;
    float3 w_h = normalize(w_o + w_i * eta_i / eta_o);

    float alpha = max(0.001f, mat.roughness * mat.roughness);
    float d = Gtr2(fabs(dot(n, w_h)), alpha);

    float f = FresnelDielectric(fabs(dot(w_i, n)), eta_o, eta_i);
    float g = SmithShadowingGGX(fabs(dot(n, w_i)), alpha) *
        SmithShadowingGGX(fabs(dot(n, w_o)), alpha);

    float i_dot_h = dot(w_i, w_h);
    float o_dot_h = dot(w_o, w_h);

    float c = fabs(o_dot_h) / fabs(dot(w_o, n)) * fabs(i_dot_h) / fabs(dot(w_i, n)) *
        pow2(eta_o) / pow2(eta_o * o_dot_h + eta_i * i_dot_h);

    return mat.base_color * c * (1.f - f) * g * d;
}

__device__ float3 DisneyMicrofacetAnisotropic(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i,
    const float3& v_x,
    const float3& v_y)
{
    float3 w_h = normalize(w_i + w_o);
    float lum = Luminance(mat.base_color);
    float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
    float3 spec = lerp(mat.specular * 0.08f * lerp(make_float3(1.f), tint, mat.specular_tint),
        mat.base_color,
        mat.metallic);

    float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
    float a = mat.roughness * mat.roughness;
    float2 alpha = make_float2(max(0.001f, a / aspect), max(0.001f, a * aspect));
    float d = Gtr2Aniso(dot(n, w_h), fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
    float3 f = lerp(spec, make_float3(1.f), SchlickWeight(dot(w_i, w_h)));
    float g = SmithShadowingGGXAniso(
        dot(n, w_i), fabs(dot(w_i, v_x)), fabs(dot(w_i, v_y)), alpha) *
        SmithShadowingGGXAniso(
            dot(n, w_o), fabs(dot(w_o, v_x)), fabs(dot(w_o, v_y)), alpha);
    return d * f * g;
}

__device__ float DisneyClearCoat(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i)
{
    float3 w_h = normalize(w_i + w_o);
    float alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
    float d = Gtr1(dot(n, w_h), alpha);
    float f = lerp(0.04f, 1.f, SchlickWeight(dot(w_i, n)));
    float g =
        SmithShadowingGGX(dot(n, w_i), 0.25f) * SmithShadowingGGX(dot(n, w_o), 0.25f);
    return 0.25f * mat.clearcoat * d * f * g;
}

__device__ float3 DisneySheen(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i)
{
    float3 w_h = normalize(w_i + w_o);
    float lum = Luminance(mat.base_color);
    float3 tint = lum > 0.f ? mat.base_color / lum : make_float3(1.f);
    float3 sheen_color = lerp(make_float3(1.f), tint, mat.sheen_tint);
    float f = SchlickWeight(dot(w_i, n));
    return f * mat.sheen * sheen_color;
}

__device__ float3 DisneyBrdf(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i,
    const float3& v_x,
    const float3& v_y)
{
    if (!SameHemisphere(w_o, w_i, n)) 
    {
        if (mat.specular_transmission > 0.f) 
        {
            float3 spec_trans = DisneyMicrofacetTransmissionIsotropic(mat, n, w_o, w_i);
            return spec_trans * (1.f - mat.metallic) * mat.specular_transmission;
        }
        return make_float3(0.f);
    }

    float coat = DisneyClearCoat(mat, n, w_o, w_i);
    float3 sheen = DisneySheen(mat, n, w_o, w_i);
    float3 diffuse = DisneyDiffuse(mat, n, w_o, w_i);
    float3 gloss;
    if (mat.anisotropy == 0.f)
    {
        gloss = DisneyMicrofacetIsotropic(mat, n, w_o, w_i);
    }
    else 
    {
        gloss = DisneyMicrofacetAnisotropic(mat, n, w_o, w_i, v_x, v_y);
    }
    return (diffuse + sheen) * (1.f - mat.metallic) * (1.f - mat.specular_transmission) +
        gloss + coat;
}

__device__ float DisneyPdf(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& w_i,
    const float3& v_x,
    const float3& v_y)
{
    float alpha = max(0.001f, mat.roughness * mat.roughness);
    float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
    float2 alpha_aniso = make_float2(max(0.001f, alpha / aspect), max(0.001f, alpha * aspect));

    float clearcoat_alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);

    float diffuse = LamberitanPdf(w_i, n);
    float clear_coat = Gtr1Pdf(w_o, w_i, n, clearcoat_alpha);

    float n_comp = 3.f;
    float microfacet = 0.f;
    float microfacet_transmission = 0.f;
    if (mat.anisotropy == 0.f) 
    {
        microfacet = Gtr2Pdf(w_o, w_i, n, alpha);
    }
    else
    {
        microfacet = Gtr2AnisoPdf(w_o, w_i, n, v_x, v_y, alpha_aniso);
    }
    if (mat.specular_transmission > 0.f && !SameHemisphere(w_o, w_i, n)) 
    {
        n_comp = 4.f;
        microfacet_transmission = Gtr2TransmissionPdf(w_o, w_i, n, alpha, mat.ior);
    }
    return (diffuse + microfacet + microfacet_transmission + clear_coat) / n_comp;
}

/* Sample a component of the Disney BRDF, returns the sampled BRDF color,
 * ray reflection direction (w_i) and sample PDF.
 */
__device__ float3 SampleDisneyBrdf(const DisneyMaterial& mat,
    const float3& n,
    const float3& w_o,
    const float3& v_x,
    const float3& v_y,
    unsigned int& seed,
    float3& w_i,
    float& pdf)
{
    int component = 0;
    if (mat.specular_transmission == 0.f) 
    {
        component = rnd(seed) * 3.f;
        component = clamp(component, 0, 2);
    }
    else 
    {
		component = rnd(seed) * 4.f;
		component = clamp(component, 0, 3);
    }

    float2 samples = make_float2(rnd(seed), rnd(seed));
    if (component == 0) 
    {
        // Sample diffuse component
        w_i = SampleLambertianDir(n, v_x, v_y, samples);
    }
    else if (component == 1) 
    {
        float3 w_h;
        float alpha = max(0.001f, mat.roughness * mat.roughness);
        if (mat.anisotropy == 0.f) 
        {
            w_h = SampleGtr2H(n, v_x, v_y, alpha, samples);
        }
        else
        {
            float aspect = sqrt(1.f - mat.anisotropy * 0.9f);
            float2 alpha_aniso =
                make_float2(max(0.001f, alpha / aspect), max(0.001f, alpha * aspect));
            w_h = SampleGtr2AnisoH(n, v_x, v_y, alpha_aniso, samples);
        }
        w_i = reflect(-w_o, w_h);

        // Invalid reflection, terminate ray
        if (!SameHemisphere(w_o, w_i, n))
        {
            pdf = 0.f;
            w_i = make_float3(0.f);
            return make_float3(0.f);
        }
    }
    else if (component == 2) 
    {
        // Sample clear coat component
        float alpha = lerp(0.1f, 0.001f, mat.clearcoat_gloss);
        float3 w_h = SampleGtr1H(n, v_x, v_y, alpha, samples);
        w_i = reflect(-w_o, w_h);

        // Invalid reflection, terminate ray
        if (!SameHemisphere(w_o, w_i, n)) 
        {
            pdf = 0.f;
            w_i = make_float3(0.f);
            return make_float3(0.f);
        }
    }
    else 
    {
        // Sample microfacet transmission component
        float alpha = max(0.001f, mat.roughness * mat.roughness);
        float3 w_h = SampleGtr2H(n, v_x, v_y, alpha, samples);
        if (dot(w_o, w_h) < 0.f) 
        {
            w_h = -w_h;
        }
        bool entering = dot(w_o, n) > 0.f;
        w_i = refract_ray(-w_o, w_h, entering ? 1.f / mat.ior : mat.ior);

        // Invalid refraction, terminate ray
        if (length(w_i) < M_EPSILON) 
        {
			pdf = 0.f;
			w_i = make_float3(0.f);
			return make_float3(0.f);
        }
    }
    pdf = DisneyPdf(mat, n, w_o, w_i, v_x, v_y);
    return DisneyBrdf(mat, n, w_o, w_i, v_x, v_y);
}
