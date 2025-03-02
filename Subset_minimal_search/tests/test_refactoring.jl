
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


""" Test backward search """
solution = backward_search(sm, sdp; max_steps=10, threshold=0.5, num_samples=100)

""" Test forward search """
solution = forward_search(sm, sdp; max_steps=50, threshold=0.5, num_samples=100)

""" Test beam search, returns the beam_size number of subsets"""
solutions = beam_search(sm, ep; threshold=0.5, beam_size=5, num_samples=100)

println(solution)



""" Test full search"""
# Greedy approach
best_set = greedy_subsets_search(sm, threshold_total_err=0.5, num_samples=100)
# Or forward priority search
best_set = forward_priority_search(sm, threshold_total_err=0.5, num_samples=100)

# Backward search search to reduce subsets
reduced_sets = backward_priority_reduction(sm, best_set, threshold=0.5, num_samples=100)

# Search for subsets implicating indices of 2nd and 3rd layers
I3, I2, I1 = reduced_sets
subsubset_I2 = implicative_subsets(sm.nn[1], sm.input, I3, I2, threshold=0.5, num_samples=100)
subsubset_I1 = implicative_subsets(sm.nn[2], sm.nn[1](sm.input), I2, I1, threshold=0.5, num_samples=100)


#! todo
#? forward and backward greedy search + dfs/bfs

#// sdp/ep choice

#! Attempt to write one search for all

#! ep_partial doesn't work

#! beam/forward/backward search for all

#! timeouts

#! terminate on the first valid subset


function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

I3 = init_sbitset(784)
I2 = nothing
I1 = nothing

solution = forward_search_for_all(sm, (I3, I2, I1), ep, ep_partial, threshold_total_err=0.5, num_samples=100)
reduced_solution = backward_reduction_for_all(sm, solution, sdp, sdp_partial, threshold=0.5, num_samples=100)