using Subset_minimal_search

using Subset_minimal_search.Flux
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.Makie
using Subset_minimal_search.CairoMakie
using Subset_minimal_search.Serialization

using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy, Subset_minimal, full_beam_search

model = deserialize("/home/sofia/julia_project/Subset_minimal_search/models/binary_model.jls")
# model = deserialize("C:/Users/spolu/Desktop/julia_project/Subset_minimal_search/models/binary_model.jls")
println(model)


# Prepare data and model
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_bin_neg = preprocess_bin_neg(preprocess_binary(train_X))
test_X_bin_neg = preprocess_bin_neg(preprocess_binary(test_X))

println("Digit: ", argmax(train_y[:, 1]) - 1, "\nOtput of the model: ", model(train_X_bin_neg[:, 1]), "\nModel's digit: ", argmax(model(train_X_bin_neg[:, 1])) - 1)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)
 
# test accuracy of the model
println("Train Accuracy: ", accuracy(model, train_X_bin_neg, train_y) * 100, "%")
println("Test Accuracy: ", accuracy(model, test_X_bin_neg, test_y) * 100, "%")

# Test random sampling
img = train_X_bin_neg[:, 1]
label_img = argmax(test_y[:, 1]) - 1
threshold=0.1
num_best=1
num_samples=100

# calculate_sdp or calculate_ep
best_set = full_beam_search(Subset_minimal(model, img, label_img), threshold, num_best, num_samples)
