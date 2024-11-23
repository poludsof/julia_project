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
using Plots
using Optimisers
using MLUtils: DataLoader

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

# ii_set = best_set
# best_set = ii_set
# plot_set = Set([i for i in ii_set])
# plot_mnist_with_active_pixels(img, Set(plot_set))
end
