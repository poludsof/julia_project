
``` Test refactoring of the code ```
# using Subset_minimal_search: make_forward_search_sdp, compute_sdp_fwd, Subset_minimal, forward_search

# using Subset_minimal_search
# using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy

""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)

""" nn for MILP search """
nn = Chain(Dense(28^2, 28, relu), Dense(28,28,relu), Dense(28,10)) 
model = train_nn(nn, train_X_binary, train_y, test_X_binary, test_y)


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
backward_search_sdp = make_backward_search_sdp(sm)
solution = backward_search_sdp(max_steps=1000, sdp_threshold=0.9, num_samples=100)

""" Test forward search """
forward_search_sdp = make_forward_search_sdp(sm)
solution = forward_search_sdp(max_steps=5000, sdp_threshold=0.5, num_samples=100)

