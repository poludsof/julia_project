module CUDAExt

using Subset_minimal_search, CUDA
using Subset_minimal_search.StaticBitSets


@inline function Subset_minimal_search.sample_input(img::CuVector, ii::SBitSet, num_samples::Integer)
    ii = CuArray(collect(ii))
    x = CUDA.rand(Float32, length(img), num_samples)
    map!(x -> 2*round(x) - 1, x, x)
    x[ii,:] .= img[ii]
    x
end


##############
#   This implements a fast operation of sum(==(y), ŷ)
##############
# include("sum_equal.jl")
end