using Subset_minimal_search
import Subset_minimal_search as SMS
using Subset_minimal_search.Flux
using Subset_minimal_search.LinearAlgebra
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.TimerOutputs
using Serialization
using Subset_minimal_search.Makie
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

r = SMS.BernoulliMixture(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls")))
ii = init_sbitset(784)
data_matrix = SMS.data_distribution(xₛ, ii, r, 100)

#first image
println(data_matrix[:, 1])
#change 0 to -1 and 1 to 1
println(2*data_matrix[:, 1] .- 1)
# plot_mnist_image(2*data_matrix[:, 20] .- 1)


""" Test conditioning """

""" Test MILP search """


""" Test one subset(layer) backward/forward/beam search """
# threshold denotes the required precision of the subset
# solution_subset = one_subset_forward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=10, terminate_on_first_solution=false)
# solution_subset = one_subset_backward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=100, time_limit=60)
# solution_beam_subsets = one_subset_beam_search(sm, criterium_ep; data_model=r, threshold=0.5, beam_size=5, num_samples=100, time_limit=60)


""" Test search functions that fit one and all subsets"""
# Threshold denotes allowed error of the subset
# Valid_criterium is needed to distinguish with respect to what the validity of a subset is being checked, because it can be different, e.g. as with MILP 

const to = TimerOutput()
reset_timer!(to)

#1. Initialize starting subsets
I3, I2, I1 = (init_sbitset(784), nothing, nothing)

#2. Search
solution_subsets = forward_search(sm, (I3, I2, I1), criterium_sdp; calc_func=criterium_sdp, calc_func_partial=sdp_partial, data_model=r, threshold_total_err=0.1, num_samples=1000, terminate_on_first_solution=true)
#3. Backward search or refining subsets
reduced_solution = backward_search_length_priority(sm, solution_subsets, criterium_sdp, sdp_partial; data_model=r, threshold=0.5, max_steps=100, num_samples=10, time_limit=50)

#4. Or use beam search
beam_solution = beam_search(sm, (I3, I2, I1), criterium_sdp, sdp_partial; data_model=r, error_threshold=0.5, beam_size=5, num_samples=10)

show(to)
println(solution_subsets[1])
println(beam_solution[1][1][1])

#call 10 times forward search
solutions = []
for i in 1:20
    println("Iteration: ", i)
    reset_timer!(to)
    I3, I2, I1 = (init_sbitset(784), nothing, nothing)
    subset = forward_search(sm, (I3, I2, I1), criterium_sdp; calc_func=criterium_sdp, calc_func_partial=sdp_partial, data_model=r, threshold_total_err=0.1, num_samples=1000, terminate_on_first_solution=true)
    show(to)
    push!(solutions, subset[1])
end

println(solutions)
# forward_search(sm, (I3, I2, I1), criterium_sdp; calc_func=criterium_sdp, calc_func_partial=sdp_partial, data_model=r, threshold_total_err=0.1, num_samples=1000, terminate_on_first_solution=true)
[1485, 3066, 1551, 1283, 2274, 1379, 1466, 1841, 1371, 1554]
[1284, 1465, 1369, 1735, 1375, 1559, 1378, 1476, 1298, 1300]

163,181,191,205,215,272,349,350,434,485,579,599,605,606,626,775
15,162,163,191,198,205,215,234,242,265,330,349,381,444,544,547,569,571,577,596,597,599,623,631,658,660,671
191,205,233,244,263,274,293,302,320,327,435,493,497,579,599,605,769
145,163,205,206,233,320,322,357,405,432,579,595,631,633
160,179,191,215,270,296,299,318,321,324,327,329,352,358,419,429,456,543,548,573,580,623,626,661,740
163,191,205,207,215,299,321,350,406,456,579,605,606,607,782
163,191,205,206,216,274,291,321,377,431,465,516,579,623,631,783
121,160,163,191,205,234,263,272,302,332,378,406,434,455,458,579,606,623,626,685
163,205,206,263,274,320,377,406,432,435,599,605,631,657,675
163,181,205,206,263,272,293,320,341,377,404,429,462,567,579,606,634

191,205,206,215,244,263,272,292,322,330,435,497,579,633
163,191,205,207,245,291,297,321,435,514,553,566,579,606,608,649
163,181,191,205,233,271,302,320,321,406,440,525,579,631,718
17,110,163,205,233,246,263,320,327,381,435,465,599,605,608,626,632,683
181,191,205,299,301,303,321,381,406,490,518,580,599,605,606
163,205,233,273,296,320,321,327,410,435,445,460,516,579,606,608,673
163,205,233,262,263,293,302,321,405,435,579,592,599,606,607
163,180,205,263,320,321,330,403,405,567,571,579,593,599,606,626
163,182,205,233,245,320,321,381,435,525,579,599,607,771
163,191,205,233,274,321,406,458,461,553,580,605,633,724


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