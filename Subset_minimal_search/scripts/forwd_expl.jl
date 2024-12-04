
model = deserialize("/home/sofia/julia_project/Subset_minimal_search/models/binary_model.jls")

train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_bin_neg = preprocess_bin_neg(preprocess_binary(train_X))
test_X_bin_neg = preprocess_bin_neg(preprocess_binary(test_X))

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

img = train_X_bin_neg[:, 1]
label_img = argmax(test_y[:, 1]) - 1

using Subset_minimal_search: compute_sdp, forward_search


xₛ = train_X_bin_neg[:, 1]
yₛ = argmax(test_y[:, 1]) - 1

# xₛ = collect(1:10)
solutions, reason = forward_search(model, xₛ, yₛ, max_steps=30, sdp_threshold=0.5)
println("Solutions: ", solutions)
println("Reason: ", reason)

best_solution = []
best_sdp_value = -Inf
for solution in solutions
    sdp_value = compute_sdp(model, img, solution)  # Recompute SDP for each solution
    if sdp_value > best_sdp_value
        best_sdp_value = sdp_value
        best_solution = solution
    end
end
println("Best solution: ", best_solution, " with SDP value: ", best_sdp_value)
