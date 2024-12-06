
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)
# model = deserialize("/home/sofia/julia_project/Subset_minimal_search/models/binary_model.jls")

train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

img = train_X_bin_neg[:, 1]
label_img = argmax(test_y[:, 1]) - 1

using Subset_minimal_search: compute_sdp, forward_search


xₛ = train_X_bin_neg[:, 1]
yₛ = argmax(test_y[:, 1]) - 1

# xₛ = collect(1:10)
solutions, reason = forward_search(model, xₛ, yₛ, max_steps=30, sdp_threshold=0.5, num_samples=1000)
println("Solutions: ", solutions)
println("Reason: ", reason)

best_solution, best_sdp_value = choose_best_solution(solutions, model, xₛ, 1000)
println("Best solution: ", best_solution, " \nwith SDP value: ", best_sdp_value)

# ii_set = best_solution
# plot_set = Set([i for i in ii_set])
# img = train_X_binary[:, 1]
# plot_mnist_with_active_pixels(img, Set(plot_set))