include("mnist_training.jl")
include(raw"plots.jl")
include("backward_search.jl")
include("milp.jl")
using Flux
using JuMP
using HiGHS
using LinearAlgebra
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using Flux.Data: DataLoader
using Flux.Losses

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
println("Train Accuracy: ", accuracy(nn, train_X_binary, train_y) * 100, "%")  # 98-99%
println("Test Accuracy: ", accuracy(nn, test_X_binary, test_y) * 100, "%")     # 96%


# === Try to find adversarial img ===
fix_inputs = collect(4:784)
adversarial(nn, train_X_binary[:, 1], argmax(train_y[:,1]) - 1, fix_inputs)


# === Backward DFS ===
img = train_X_binary[:, 1]
label_img = argmax(train_y[:,1]) - 1
minimal_set_dfs(nn, img, label_img)


# === Ploting ===
image_original = train_X[:, :, 20]
image_binary = reshape(train_X_binary[:, 1], 28, 28)
plot_images(image_original, image_binary)
