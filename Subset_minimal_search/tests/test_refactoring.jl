
``` Test refactoring of the code ```


""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)

""" nn for MILP search """
# nn = Chain(Dense(28^2, 28, relu), Dense(28,28,relu), Dense(28,10)) 
# model = train_nn(nn, train_X_binary, train_y, test_X_binary, test_y)


""" Prepare data """
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)


""" Prepare image and label """
xₛ = train_X_bin_neg[:, 1]
yₛ = argmax(train_y[:, 1]) - 1
sm = Subset_minimal(model, xₛ, yₛ)


function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end




""" Test one subset(layer) backward/forward/beam search """
# threshold denotes the required precision of the subset
solution_subset = one_subset_forward_search(sm, sdp; max_steps=50, threshold=0.5, num_samples=100)
solution_subset = one_subset_backward_search(sm, sdp; max_steps=50, threshold=0.5, num_samples=100)
solution_beam_subsets = beam_search(sm, ep; threshold=0.5, beam_size=5, num_samples=100)




""" Test search functions that fit one and all subsets"""
# threshold denotes allowed error of the subset

#1. Initialize starting subsets
I3, I2, I1 = (init_sbitset(784), init_sbitset(256), init_sbitset(256))
#2. Search
solution_subsets = forward_search(sm, (I3, I2, I1), sdp, sdp_partial; threshold_total_err=0.5, num_samples=100)
#3. Backward search or refining subsets
reduced_solution = backward_search_length_priority(sm, solution_subsets, sdp, sdp_partial; threshold=0.5, max_steps=500, num_samples=100)

println(reduced_solution)




""" Test full search"""
# Greedy approach
best_set = greedy_subsets_search(sm, threshold_total_err=0.5, num_samples=100)


# Search for subsets implicating indices of 2nd and 3rd layers
I3, I2, I1 = reduced_solution
subsubset_I2 = implicative_subsets(sm.nn[1], sm.input, I3, I2, threshold=0.5, num_samples=100)
subsubset_I1 = implicative_subsets(sm.nn[2], sm.nn[1](sm.input), I2, I1, threshold=0.5, num_samples=100)


#! todo
#? forward and backward greedy search + dfs/bfs
#! milp choice (heuristic + criterium)
#// sdp/ep choice
#// Attempt to write one search for all
#! ep_partial doesn't work
#! beam search for all
#! timeouts
#! terminate on the first valid subset
#// threshold of the error
#! implicative_subsets

#* BEAM SEARCH сортировка по максимальной ошибке, сначала рассмотреть все расширения в одной греппе подмножеств, сортировка по всем группам
#* 3 марта - добавить timeout
#* 4 марта - добавить milp choice


solution = forward_search_for_all(sm, (I3, I2, I1), ep, ep_partial, threshold_total_err=0.5, num_samples=100)
reduced_solution = backward_reduction_for_all(sm, solution, sdp, sdp_partial, threshold=0.5, max_steps=100, num_samples=100)