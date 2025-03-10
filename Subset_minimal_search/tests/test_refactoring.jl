using Subset_minimal_search
import Subset_minimal_search as SMS
using Subset_minimal_search.Flux
using Subset_minimal_search.LinearAlgebra
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.TimerOutputs
using Serialization
# using Subset_minimal_search.Makie
# using Subset_minimal_search.Makie.Colors
# using Subset_minimal_search.Serialization
# using Subset_minimal_search.DataStructures
# using Subset_minimal_search.Distributions
# using Subset_minimal_search.Serialization
``` Test refactoring of the code ```


""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)

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
xₛ = train_X_bin_neg[:, 1]
yₛ = argmax(train_y[:, 1]) - 1
sm = SMS.Subset_minimal(model, xₛ, yₛ)


function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end


r = BernoulliMixture(deserialize("../models/milan_centers.jls"))

""" Test one subset(layer) backward/forward/beam search """
# threshold denotes the required precision of the subset
solution_subset = one_subset_forward_search(sm, criterium_sdp; max_steps=50, threshold=0.5, num_samples=100, time_limit=60, terminate_on_first_solution=false)
solution_subset = one_subset_backward_search(sm, criterium_sdp; max_steps=50, threshold=0.5, num_samples=100, time_limit=60)
solution_beam_subsets = one_subset_beam_search(sm, criterium_ep; threshold=0.5, beam_size=5, num_samples=100, time_limit=60)

solution_subsets = forward_search(sm, criterium_ep; calc_func=criterium_sdp, calc_func_partial=sdp_partial, threshold_total_err=0.5, num_samples=100, time_limit=100, terminate_on_first_solution=false)



""" Test search functions that fit one and all subsets"""
# Threshold denotes allowed error of the subset
# Valid_criterium is needed to distinguish with respect to what the validity of a subset is being checked, because it can be different, e.g. as with MILP 

#1. Initialize starting subsets
I3, I2, I1 = (init_sbitset(784), nothing, nothing)

#2. Search
solution_subsets = forward_search(sm, (I3, I2, I1), criterium_sdp; calc_func=criterium_sdp, calc_func_partial=sdp_partial, threshold_total_err=0.5, num_samples=100, time_limit=100, terminate_on_first_solution=false)
#3. Backward search or refining subsets
reduced_solution = backward_search_length_priority(sm, solution_subsets, criterium_sdp, sdp_partial; threshold=0.5, max_steps=500, num_samples=100, time_limit=50)

#4. Or use beam search
beam_solution = beam_search(sm, (I3, I2, I1), criterium_sdp, sdp_partial; error_threshold=0.4, beam_size=5, num_samples=100, time_limit=600)

show(to)
println(reduced_solution)
println(beam_solution[1])




""" Test full search"""
# Greedy approach
best_set = greedy_subsets_search(sm, threshold_total_err=0.5, num_samples=100)

# Search for subsets implicating indices of 2nd and 3rd layers
I3, I2, I1 = reduced_solution
subsubset_I2 = implicative_subsets(sm.nn[1], sm.input, I3, I2, threshold=0.5, num_samples=100)
subsubset_I1 = implicative_subsets(sm.nn[2], sm.nn[1](sm.input), I2, I1, threshold=0.5, num_samples=100)


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

#// BEAM SEARCH сортировка по максимальной ошибке, сначала рассмотреть все расширения в одной греппе подмножеств, сортировка по всем группам
#// 3 марта - добавить timeout(forward есть, добавить в остальные функции)
#* 4 марта - add milp choice

#? если bacward имеет много решений в начале, хранить ли все? terminate on the first solution? 


solution = forward_search_for_all(sm, (I3, I2, I1), ep, ep_partial, threshold_total_err=0.5, num_samples=100)
reduced_solution = backward_reduction_for_all(sm, solution, sdp, sdp_partial, threshold=0.5, max_steps=100, num_samples=100)


fix_inputs = collect(1:7)
adversarial(model, sm.input, sm.output, fix_inputs)