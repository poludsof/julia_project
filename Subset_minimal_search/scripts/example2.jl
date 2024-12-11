
"""
Beam search with b=1 with partial and full probability criterion
"""

using Subset_minimal_search

using Subset_minimal_search.Flux
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.Makie
using Subset_minimal_search.CairoMakie
using Subset_minimal_search.Serialization

using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy, Subset_minimal, full_beam_search

model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)
# println(model)


# Prepare data and model
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

# println("Digit: ", argmax(train_y[:, 1]) - 1, "\nOtput of the model: ", model(train_X_bin_neg[:, 1]), "\nModel's digit: ", argmax(model(train_X_bin_neg[:, 1])) - 1)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)
 
# test accuracy of the model
# println("Train Accuracy: ", accuracy(model, train_X_bin_neg, train_y) * 100, "%")
# println("Test Accuracy: ", accuracy(model, test_X_bin_neg, test_y) * 100, "%")

# Test random sampling
img = train_X_bin_neg[:, 1]
label_img = argmax(train_y[:, 1]) - 1
threshold=0.1
num_best=1
num_samples=100

# calculate_sdp or calculate_ep
best_set = full_beam_search2(Subset_minimal(model, img, label_img), threshold, num_samples)
reduced_sets = backward_dfs_search(Subset_minimal(model, img, label_img), best_set, 0.8, num_samples)

I3_set, I2_set, I1_set = best_set
println("Subset I3: ", I3_set)
println("Subset I2: ", I2_set)
println("Subset I1: ", I1_set)


for i in collect(I3_test)
    println("Subset: ", i)
end

I3_set_tmp = deepcopy(I3_set)
I3_tmp, I2_tmp, I1_tmp = deepcopy(best_set)
popped = pop(I3_tmp, 4)
I3_test = push(I3_test, 4)
println("Subset I3: ", I3_test)


#test backward
I3_test = SBitSet{32, UInt32}(collect(1:2))
I2_test = SBitSet{32, UInt32}(collect(3:4))
I1_test = SBitSet{32, UInt32}(collect(5:6))
red_test_sets = backward_dfs_search(Subset_minimal(model, img, label_img), (I3_test, I2_test, I1_test), num_samples)

img_bin = train_X_binary[:, 1]
plot_set = Set([i for i in best_set[1]])
plot_mnist_with_active_pixels(img_bin, plot_set)

best_set = forward_beam_search2(Subset_minimal(model, img, label_img), 0.7, num_samples, 3)

best_set = full_beam_search_with_sorted_candidates(Subset_minimal(model, img, label_img), 0.8, num_samples)