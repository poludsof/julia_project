module CUDAExt

using Subset_minimal_search, CUDA
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.StatsBase
using Subset_minimal_search.Flux
using Subset_minimal_search: ConditionedUniformDistribution, UniformDistribution, condition, ConditionedBernoulliMixture

@inline function Subset_minimal_search.condition(r::UniformDistribution, xₛ::CuArray, known_ii::SBitSet)
    ConditionedUniformDistribution(cu(condition(r, Array(xₛ), known_ii).p))
end

@inline function Subset_minimal_search.sample_all(r::ConditionedUniformDistribution{<:CuVector}, n::Integer)
    x = CUDA.rand(Float32, length(r.p), n)
    _f(xᵢ, pᵢ) = 2*(xᵢ < pᵢ) - 1
    _f.(x, r.p)
end




##############
#   This implements sampling from conditioned bernoulli
##############
function Subset_minimal_search.condition(r::BernoulliMixture{<:Any,<:CuArray,<:CuMatrix}, xₛ::CuVector, mask::Vector{Bool})
    mask = cu(mask)
    Subset_minimal_search.condition(r, xₛ, mask)
end

function Subset_minimal_search.condition(r::BernoulliMixture{<:Any,<:CuArray,<:CuMatrix}, xₛ::CuVector, mask::CuVector{Bool})
    _xₛ = vcat((xₛ .≤ 0)', (xₛ .> 0)')
    pzx = softmax(vec(sum(mask' .* _xₛ .* r.log_p, dims = (1,2))))
    w = StatsBase.Weights(Vector(pzx))
    ConditionedBernoulliMixture(r, xₛ, mask, w)
end

function Subset_minimal_search.sample_all(r::ConditionedBernoulliMixture{<:Any, <:CuArray, <:CuArray}, n::Integer)
    mask = r.mask
    p = r.r.p
    xₛ = r.xₛ
    _f(mᵢ, xᵢ, xₛᵢ, pᵢ) = mᵢ ? Float32(xₛᵢ) : Float32((2*(xᵢ < pᵢ) - 1))
    
    x = CUDA.rand(Float32, size(p,1), n)
    cids = sample(1:length(r.w), r.w, n)
    _f.(mask, x, xₛ, p[:,cids])
end

end