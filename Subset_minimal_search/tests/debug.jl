# srun -p gpufast --gres=gpu:1  --mem=16000  --pty bash -i
# cd julia/Pkg/Subset_minimal_search/tests/
#  ~/.juliaup/bin/julia --project=..
using CUDA
using Subset_minimal_search
import Subset_minimal_search as SMS
using Subset_minimal_search.Flux
using Subset_minimal_search.LinearAlgebra
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.TimerOutputs
using Serialization
using Subset_minimal_search.Makie
using ProfileCanvas, TimerOutputs
const to = Subset_minimal_search.to
# using Subset_minimal_search.Makie.Colors
# using Subset_minimal_search.Serialization
# using Subset_minimal_search.DataStructures
# using Subset_minimal_search.Distributions
# using Subset_minimal_search.Serialization

to_gpu = gpu
# to_gpu = cpu

""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path) |> to_gpu;

""" nn for MILP search """
# nn = Chain(Dense(28^2, 28, relu), Dense(28,28, relu), Dense(28,10)) 
# nn = train_nn(nn, train_X_bin_neg, train_y, test_X_bin_neg, test_y)


""" Prepare data """
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = SMS.onehot_labels(train_y)
test_y = SMS.onehot_labels(test_y)

""" Prepare image and label """
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(train_y[:, 1]) - 1
sm = SMS.Subset_minimal(model, xₛ, yₛ)


function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

r = SMS.BernoulliMixture(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls")))

#test
# ii = init_sbitset(784)
# data_matrix = SMS.data_distribution(xₛ, ii, r, 100)

#first image
# println(data_matrix[:, 1])
#change 0 to -1 and 1 to 1
# println(2*data_matrix[:, 1] .- 1)
# plot_mnist_image(2*data_matrix[:, 20] .- 1)


""" Test conditioning """

""" Test MILP search """


# """ Test one subset(layer) backward/forward/beam search """ -- too slow
# threshold denotes the required precision of the subset
# solution_subset = one_subset_forward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=10, terminate_on_first_solution=false)
# solution_subset = one_subset_backward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=100, time_limit=60)
# solution_beam_subsets = one_subset_beam_search(sm, criterium_ep; data_model=r, threshold=0.5, beam_size=5, num_samples=100, time_limit=60)


""" Test search functions that fit one and all subsets"""
# Threshold denotes allowed error of the subset
# Valid_criterium is needed to distinguish with respect to what the validity of a subset is being checked, because it can be different, e.g. as with MILP 

reset_timer!(to)

#1. Initialize starting subsets
I3, I2, I1 = (init_sbitset(784), nothing, nothing)

#2. Search
""" Prepare image and label """
img_i = 1
xₛ = train_X_bin_neg[:, img_i] |> to_gpu
yₛ = argmax(model(xₛ))
sm = SMS.Subset_minimal(model, xₛ, yₛ)
I3, I2, I1 = (init_sbitset(784), nothing, nothing)

t = @elapsed solution_subsets = forward_search(sm, (I3, I2, I1), criterium_sdp; calc_func=criterium_sdp, calc_func_partial=sdp_partial, data_model=nothing, threshold_total_err=0.1, num_samples=10000, terminate_on_first_solution=true)

