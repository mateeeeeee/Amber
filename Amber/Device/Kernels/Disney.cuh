#include "Material.cuh"


__device__ Float3 DisneyDiffuse(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i)
{
	Float3 w_h = normalize(w_i + w_o);
	Float n_dot_o = fabs(dot(w_o, n));
	Float n_dot_i = fabs(dot(w_i, n));
	Float i_dot_h = dot(w_i, w_h);
	Float fd90 = 0.5f + 2.0f * material.roughness * i_dot_h * i_dot_h;
	Float fi = SchlickWeight(n_dot_i);
	Float fo = SchlickWeight(n_dot_o);
	return material.base_color * M_INV_PI * lerp(1.0f, fd90, fi) * lerp(1.0f, fd90, fo);
}

__device__ Float3 DisneyMicrofacetIsotropic(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i)
{
	Float3 w_h = normalize(w_i + w_o);
	Float lum = Luminance(material.base_color);
	Float3 tint = lum > 0.0f ? material.base_color / lum : MakeFloat3(1.0f);
	Float3 spec = lerp(material.specular * 0.08f * lerp(MakeFloat3(1.0f), tint, material.specular_tint),
		material.base_color,
		material.metallic);

	Float alpha = max(0.001f, material.roughness * material.roughness);
	Float d = Gtr2(dot(n, w_h), alpha);
	Float3 f = lerp(spec, MakeFloat3(1.0f), SchlickWeight(dot(w_i, w_h)));
	Float g = SmithShadowingGGX(dot(n, w_i), alpha) * SmithShadowingGGX(dot(n, w_o), alpha);
	return d * f * g;
}

__device__ Float3 DisneyMicrofacetTransmissionIsotropic(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i)
{
	Float o_dot_n = dot(w_o, n);
	Float i_dot_n = dot(w_i, n);
	if (o_dot_n == 0.0f || i_dot_n == 0.0f)
	{
		return MakeFloat3(0.0f);
	}
	Bool entering = o_dot_n > 0.0f;
	Float eta_o = entering ? 1.0f : material.ior;
	Float eta_i = entering ? material.ior : 1.0f;
	Float3 w_h = normalize(w_o + w_i * eta_i / eta_o);

	Float alpha = max(0.001f, material.roughness * material.roughness);
	Float d = Gtr2(fabs(dot(n, w_h)), alpha);

	Float f = FresnelDielectric(fabs(dot(w_i, n)), eta_o, eta_i);
	Float g = SmithShadowingGGX(fabs(dot(n, w_i)), alpha) *
		SmithShadowingGGX(fabs(dot(n, w_o)), alpha);

	Float i_dot_h = dot(w_i, w_h);
	Float o_dot_h = dot(w_o, w_h);

	Float c = fabs(o_dot_h) / fabs(dot(w_o, n)) * fabs(i_dot_h) / fabs(dot(w_i, n)) *
		Pow2(eta_o) / Pow2(eta_o * o_dot_h + eta_i * i_dot_h);

	return material.base_color * c * (1.0f - f) * g * d;
}

__device__ Float3 DisneyMicrofacetAnisotropic(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i,
	Float3 const& v_x,
	Float3 const& v_y)
{
	Float3 w_h = normalize(w_i + w_o);
	Float lum = Luminance(material.base_color);
	Float3 tint = lum > 0.0f ? material.base_color / lum : MakeFloat3(1.0f);
	Float3 spec = lerp(material.specular * 0.08f * lerp(MakeFloat3(1.0f), tint, material.specular_tint),
		material.base_color,
		material.metallic);

	Float aspect = sqrt(1.0f - material.anisotropy * 0.9f);
	Float a = material.roughness * material.roughness;
	Float2 alpha = make_float2(max(0.001f, a / aspect), max(0.001f, a * aspect));
	Float d = Gtr2Aniso(dot(n, w_h), fabs(dot(w_h, v_x)), fabs(dot(w_h, v_y)), alpha);
	Float3 f = lerp(spec, MakeFloat3(1.0f), SchlickWeight(dot(w_i, w_h)));
	Float g = SmithShadowingGGXAniso(
		dot(n, w_i), fabs(dot(w_i, v_x)), fabs(dot(w_i, v_y)), alpha) *
		SmithShadowingGGXAniso(dot(n, w_o), fabs(dot(w_o, v_x)), fabs(dot(w_o, v_y)), alpha);
	return d * f * g;
}

__device__ Float DisneyClearCoat(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i)
{
	Float3 w_h = normalize(w_i + w_o);
	Float alpha = lerp(0.1f, 0.001f, material.clearcoat_gloss);
	Float d = Gtr1(dot(n, w_h), alpha);
	Float f = lerp(0.04f, 1.0f, SchlickWeight(dot(w_i, n)));
	Float g = SmithShadowingGGX(dot(n, w_i), 0.25f) * SmithShadowingGGX(dot(n, w_o), 0.25f);
	return 0.25f * material.clearcoat * d * f * g;
}

__device__ Float3 DisneySheen(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i)
{
	Float3 w_h = normalize(w_i + w_o);
	Float lum = Luminance(material.base_color);
	Float3 tint = lum > 0.0f ? material.base_color / lum : MakeFloat3(1.0f);
	Float3 sheen_color = lerp(MakeFloat3(1.0f), tint, material.sheen_tint);
	Float f = SchlickWeight(dot(w_i, n));
	return f * material.sheen * sheen_color;
}

__device__ Float3 DisneyBrdf(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i,
	Float3 const& v_x,
	Float3 const& v_y)
{
	if (!SameHemisphere(w_o, w_i, n))
	{
		if (material.specular_transmission > 0.0f)
		{
			Float3 spec_trans = DisneyMicrofacetTransmissionIsotropic(material, n, w_o, w_i);
			return spec_trans * (1.0f - material.metallic) * material.specular_transmission;
		}
		return MakeFloat3(0.0f);
	}

	Float coat = DisneyClearCoat(material, n, w_o, w_i);
	Float3 sheen = DisneySheen(material, n, w_o, w_i);
	Float3 diffuse = DisneyDiffuse(material, n, w_o, w_i);
	Float3 gloss;
	if (material.anisotropy == 0.0f)
	{
		gloss = DisneyMicrofacetIsotropic(material, n, w_o, w_i);
	}
	else
	{
		gloss = DisneyMicrofacetAnisotropic(material, n, w_o, w_i, v_x, v_y);
	}
	return (diffuse + sheen) * (1.0f - material.metallic) * (1.0f - material.specular_transmission) + gloss + coat;
}

__device__ Float DisneyPdf(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& w_i,
	Float3 const& v_x,
	Float3 const& v_y)
{
	Float alpha = max(0.001f, material.roughness * material.roughness);
	Float aspect = sqrt(1.0f - material.anisotropy * 0.9f);
	Float2 alpha_aniso = make_float2(max(0.001f, alpha / aspect), max(0.001f, alpha * aspect));
	Float clearcoat_alpha = lerp(0.1f, 0.001f, material.clearcoat_gloss);

	Float diffuse = LamberitanPdf(w_i, n);
	Float clear_coat = Gtr1Pdf(w_o, w_i, n, clearcoat_alpha);

	Int component_count = 3;
	Float microfacet = 0.0f;
	Float microfacet_transmission = 0.0f;
	if (material.anisotropy == 0.0f)
	{
		microfacet = Gtr2Pdf(w_o, w_i, n, alpha);
	}
	else
	{
		microfacet = Gtr2AnisoPdf(w_o, w_i, n, v_x, v_y, alpha_aniso);
	}
	if (material.specular_transmission > 0.0f && !SameHemisphere(w_o, w_i, n))
	{
		component_count = 4;
		microfacet_transmission = Gtr2TransmissionPdf(w_o, w_i, n, alpha, material.ior);
	}
	return (diffuse + microfacet + microfacet_transmission + clear_coat) / component_count;
}

__device__ Float3 SampleDisneyBrdf(EvaluatedMaterial const& material,
	Float3 const& n,
	Float3 const& w_o,
	Float3 const& v_x,
	Float3 const& v_y,
	PRNG& prng,
	Float3& w_i,
	Float& pdf)
{
	Int component = 0;
	if (material.specular_transmission == 0.0f)
	{
		component = prng.RandomInt() % 3;
	}
	else
	{
		component = prng.RandomInt() % 4;
	}

	Float2 samples = make_float2(prng.RandomFloat(), prng.RandomFloat());
	if (component == 0)
	{
		w_i = SampleLambertianDir(n, v_x, v_y, samples);
	}
	else if (component == 1)
	{
		Float3 w_h;
		Float alpha = max(0.001f, material.roughness * material.roughness);
		if (material.anisotropy == 0.0f)
		{
			w_h = SampleGtr2H(n, v_x, v_y, alpha, samples);
		}
		else
		{
			Float aspect = sqrt(1.0f - material.anisotropy * 0.9f);
			Float2 alpha_aniso =
				make_float2(max(0.001f, alpha / aspect), max(0.001f, alpha * aspect));
			w_h = SampleGtr2AnisoH(n, v_x, v_y, alpha_aniso, samples);
		}
		w_i = reflect(-w_o, w_h);

		if (!SameHemisphere(w_o, w_i, n))
		{
			pdf = 0.0f;
			w_i = MakeFloat3(0.0f);
			return MakeFloat3(0.0f);
		}
	}
	else if (component == 2)
	{
		// Sample clear coat component
		Float alpha = lerp(0.1f, 0.001f, material.clearcoat_gloss);
		Float3 w_h = SampleGtr1H(n, v_x, v_y, alpha, samples);
		w_i = reflect(-w_o, w_h);

		if (!SameHemisphere(w_o, w_i, n))
		{
			pdf = 0.0f;
			w_i = MakeFloat3(0.0f);
			return MakeFloat3(0.0f);
		}
	}
	else
	{
		// Sample microfacet transmission component
		Float alpha = max(0.001f, material.roughness * material.roughness);
		Float3 w_h = SampleGtr2H(n, v_x, v_y, alpha, samples);
		if (dot(w_o, w_h) < 0.0f)
		{
			w_h = -w_h;
		}
		Bool entering = dot(w_o, n) > 0.0f;
		w_i = refract_ray(-w_o, w_h, entering ? 1.0f / material.ior : material.ior);

		if (length(w_i) < M_EPSILON)
		{
			pdf = 0.0f;
			w_i = MakeFloat3(0.0f);
			return MakeFloat3(0.0f);
		}
	}
	pdf = DisneyPdf(material, n, w_o, w_i, v_x, v_y);
	return DisneyBrdf(material, n, w_o, w_i, v_x, v_y);
}