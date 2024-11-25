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

# import CairoMakie
# import Makie

const to = TimerOutput()

struct Subset_minimal{NN, I, O}
    nn::NN
    input::I
    output::O
end


include("mnist_training.jl")
include("plots.jl")
include("backward_search.jl")
include("forward_search.jl")
include("milp.jl")
include("random_sampling.jl")

# calculate_sdp(Subset_minimal(nn, img, label_img), best_set, num_samples)


end
