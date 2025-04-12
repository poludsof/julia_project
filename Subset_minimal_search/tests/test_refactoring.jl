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


to_gpu = gpu
""" Prepare image and label """
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(train_y[:, 1]) - 1
sm = SMS.Subset_minimal(to_gpu(model), xₛ, yₛ)


function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

sampler = SMS.BernoulliMixture(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls")))

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
# data_model = r for data distribution
# data_model = nothing for uniform
for img_i in 1:100
    """ Prepare image and label """
    xₛ = train_X_bin_neg[:, img_i] |> to_gpu
    yₛ = argmax(model(xₛ))
    sm = SMS.Subset_minimal(to_gpu(model), xₛ, yₛ)
    I3, I2, I1 = (init_sbitset(784), nothing, nothing)
end

#new API
ϵ = 0.8
II = init_sbitset(length(xₛ))
t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 100),  ShapleyHeuristic(sm, sampler, 100))

ϵ = 0.9
II = SBitSet{13, UInt64}(collect(1:784))
t = @elapsed solution_subsets = backward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10),  ShapleyHeuristic(sm, sampler, 10), time_limit=60)

II = init_sbitset(length(xₛ))
t = @elapsed solution_subsets = beam_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10),  ShapleyHeuristic(sm, sampler, 10), beam_size=5, time_limit=60)

show(to)
println(solution_subsets[1])



II = SBitSet{13, UInt64}(collect(1:70))
pop(II, 1)
II

push(II, 8)
II

new_subsets(II, 7)

new_subsets(ii::SBitSet, idim) = [pop(ii, i) for i in 1:idim if i in ii]




""" not important for now """
# # Greedy approach
# best_set = greedy_subsets_search(sm, threshold_total_err=0.5, num_samples=100)

# # Search for subsets implicating indices of 2nd and 3rd layers
# I3, I2, I1 = reduced_solution
# subsubset_I2 = implicative_subsets(sm.nn[1], sm.input, I3, I2, threshold=0.5, num_samples=100)
# subsubset_I1 = implicative_subsets(sm.nn[2], sm.nn[1](sm.input), I2, I1, threshold=0.5, num_samples=100)


# TODO
#? forward and backward greedy search + dfs/bfs
#! milp choice (heuristic + criterium)
#! valid criterium (for milp)
#// sdp/ep choice
#// Attempt to write one search for all
#! ep_partial doesn't work
#// beam search for all
#// timeouts
#! terminate on the first valid subset
#// threshold of the error
#! implicative_subsets
#// forward/backward/beam add isvalid
#! distribution choice
#! check isvalid (threshold error and 0 <=)

#? если bacward имеет много решений в начале, хранить ли все? terminate on the first solution? 


# solution = forward_search_for_all(sm, (I3, I2, I1), ep, ep_partial, threshold_total_err=0.5, num_samples=100)
# reduced_solution = backward_reduction_for_all(sm, solution, sdp, sdp_partial, threshold=0.5, max_steps=100, num_samples=100)


# fix_inputs = collect(1:7)
# adversarial(model, sm.input, sm.output, fix_inputs)