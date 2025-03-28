module Subset_minimal_search

using Flux
using JuMP
using HiGHS
using LinearAlgebra
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using StaticBitSets
using TimerOutputs
using Optimisers
using MLUtils: DataLoader
using CairoMakie
using Makie
using Makie.Colors
using Serialization
using DataStructures
using Distributions

const to = TimerOutput()

struct Subset_minimal{NN, I, O}
    nn::NN
    input::I
    output::O
end


include("mnist_training.jl")
include("plots.jl")
include("milp.jl")
include("full_search.jl")
include("implicative_subsets.jl")

include("criterium.jl")
include("forward_search.jl")
include("backward_search.jl")
include("beam_search.jl")
include("dataset_prep.jl")
include("heuristic.jl")
include("mixture_sampler.jl")

include("tmp_bcw.jl")
include("tmp_one_search_for_all.jl")

export one_subset_backward_search, one_subset_forward_search, one_subset_beam_search
export preprocess_binary, preprocess_bin_neg 
export forward_search, backward_search_error_priority, beam_search, backward_search_length_priority
export criterium_sdp, criterium_ep, sdp_partial, ep_partial
export plot_mnist_image

end
