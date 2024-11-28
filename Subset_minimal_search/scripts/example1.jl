using Subset_minimal_search

using Subset_minimal_search.Flux
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.Makie
using Subset_minimal_search.CairoMakie

using Subset_minimal_search.Serialization
using Subset_minimal_search.BSON

using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy, get_minimal_set_generic, Subset_minimal

model_data = BSON.load("/home/sofia/julia_project/Subset_minimal_search/models/binary_model.jls")
filesize = stat("/home/sofia/julia_project/Subset_minimal_search/models/binary_model.jls").size
model = deserialize("/home/sofia/julia_project/Subset_minimal_search/models/binary_model.jls")
println(model)

# === Create a neural network ===
nn = Chain(Dense(28^2, 28, relu), Dense(28,28,relu), Dense(28,10)) 

# Prepare data and model
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_bin_neg = preprocess_bin_neg(train_X)
test_X_bin_neg = preprocess_bin_neg(test_X)

println("Digit: ", argmax(train_y[:, 1]) - 1, "\nOtput of the model: ", model(train_X_bin_neg[:, 1]), "\nModel's digit: ", argmax(model(train_X_bin_neg[:, 1])) - 1)

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

 
# === Training ===
nn = Subset_minimal_search.train_nn(nn, train_X_binary, train_y, test_X_binary, test_y)

# argmax(nn(train_X_binary[:, 1])) - 1
println("Train Accuracy: ", accuracy(nn, train_X_binary, train_y) * 100, "%")  # 97-99%
println("Test Accuracy: ", accuracy(nn, test_X_binary, test_y) * 100, "%")     # 95-96%


# test accuracy of the model
println("Train Accuracy: ", accuracy(model, train_X_bin_neg, train_y) * 100, "%")
println("Test Accuracy: ", accuracy(model, test_X_bin_neg, test_y) * 100, "%")

# === Try to find adversarial img ===
# fix_inputs = collect(4:780)
# adversarial(nn, train_X_binary[:, 1], argmax(train_y[:, 1]) - 1, fix_inputs)


# === Ploting ===
using Subset_minimal_search: plot_images
image_original = train_X[:, :, 1]
image_binary = reshape(train_X_binary[:, 1], 28, 28)
plot_images(image_original, image_binary)


# Test random sampling
img = test_X_binary[:, 2]
label_img = argmax(test_y[:, 2]) - 1
ii_set = SBitSet{32, UInt32}(collect(1:7))
threshold=0.1
num_best=3
num_samples=70

using Subset_minimal_search: plot_mnist_with_active_pixels
# calculate_sdp or calculate_ep
best_set = get_minimal_set_generic(Subset_minimal(nn, img, label_img), calculate_ep, threshold, num_best, num_samples)

ii_set = best_set
best_set = ii_set
plot_set = Set([i for i in ii_set])
plot_mnist_with_active_pixels(img, Set(plot_set))