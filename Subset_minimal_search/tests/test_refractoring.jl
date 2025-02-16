
``` Test refactoring of the code ```
using Subset_minimal_search: make_forward_search_sdp, compute_sdp_fwd, Subset_minimal, forward_search

""" Prepare model and data """
# using Subset_minimal_search
# using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy

model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)

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


""" Test forward search """
# solutions, reason = forward_search(model, xₛ, yₛ, max_steps=30, sdp_threshold=0.5, num_samples=100)
# println("Solutions: ", solutions)
# println("Reason: ", reason)

# using closure:
forward_search_sdp = make_forward_search_sdp(sm)
solution = forward_search_sdp(max_steps=50, sdp_threshold=0.5, num_samples=1000)
