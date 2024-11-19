module Subset_minimal_search

using Flux
using JuMP
using HiGHS
using LinearAlgebra
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using StaticBitSets
using TimerOutputs
using Plots
using Optimisers
using MLUtils: DataLoader

const to = TimerOutput()

struct Subset_minimal{NN, I, O}
    nn::NN
    input::I
    output::O
end


include("mnist_training.jl")
include("plots.jl")
include("backward_search.jl")
include("forward_search.jl")
include("milp.jl")
include("random_sampling.jl")

# === Create a neural network ===
nn = Chain(Dense(28^2, 28, relu), Dense(28,28,relu), Dense(28,10)) 

# Prepare data and model
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

 
# === Training ===
nn = train_nn(nn, train_X_binary, train_y, test_X_binary, test_y)

# argmax(nn(train_X_binary[:, 1])) - 1
println("Train Accuracy: ", accuracy(nn, train_X_binary, train_y) * 100, "%")  # 97-99%
println("Test Accuracy: ", accuracy(nn, test_X_binary, test_y) * 100, "%")     # 95-96%


# === Try to find adversarial img ===
# fix_inputs = collect(4:780)
# adversarial(nn, train_X_binary[:, 1], argmax(train_y[:, 1]) - 1, fix_inputs)


# === Ploting ===
image_original = train_X[:, :, 2]
image_binary = reshape(train_X_binary[:, 2], 28, 28)
plot_images(image_original, image_binary)





# subset_min = minimal_set_search(Subset_minimal(nn, img, label_img))

# Test random sampling
img = test_X_binary[:, 9]
label_img = argmax(test_y[:, 9]) - 1
ii_set = SBitSet{32, UInt32}()
threshold=0.99
num_best=3
num_samples=100
best_set = get_best_best_sdp(Subset_minimal(nn, img, label_img), threshold, num_best, num_samples)
println(best_set)

calculate_sdp(Subset_minimal(nn, img, label_img), best_set, num_samples)

ii_set = best_set
best_set = ii_set
plot_set = Set([i for i in ii_set])
plot_mnist_with_active_pixels(img, Set(plot_set))
end
