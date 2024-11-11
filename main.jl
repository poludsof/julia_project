using Flux
using JuMP
using HiGHS
using LinearAlgebra
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using Flux.Data: DataLoader
using Flux.Losses
using StaticBitSets
include("mnist_training.jl")
include(raw"plots.jl")
include("backward_search.jl")
include("milp.jl")

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
println("Test Accuracy: ", accuracy(nn, test_X_binary, test_y) * 100, "%")     # 95-96%


# === Try to find adversarial img ===
fix_inputs = collect(4:780)
adversarial(nn, train_X_binary[:, 1], argmax(train_y[:, 1]) - 1, fix_inputs)


# === Backward DFS ===
img = train_X_binary[:, 1]
label_img = argmax(train_y[:,1]) - 1
subset_min = minimal_set_dfs(nn, img, label_img)
#check
adversarial(nn, img, label_img, subset_min)
#plot
plot_set = Set([i for i in subset_min])
plot_mnist_with_active_pixels(train_X_binary[:, 1], Set(plot_set))

# given_input_set = SBitSet{2,UInt8}(1,2,8)
println(visited_local)



# === Ploting ===
image_original = train_X[:, :, 20]
image_binary = reshape(train_X_binary[:, 1], 28, 28)
plot_images(image_original, image_binary)


N = 8
current_set = SBitSet{N, UInt8}(1, 3, 5)
next_set = current_set ~ SBitSet{N, UInt8}([3])
for i in current_set
    next_set = current_set ~ SBitSet{N, UInt8}(i)
    println(next_set)
end