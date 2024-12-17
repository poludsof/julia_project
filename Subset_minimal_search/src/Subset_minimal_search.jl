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
include("beam_search.jl")
include("full_search.jl")
include("full_fwd_search.jl")
# include("neural_network_visualization.jl")
include("visualization.jl")
include("subset_search_exmpl.jl")



end
