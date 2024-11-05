include(raw"plots.jl")
using Flux
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using Flux.Data: DataLoader

# Load data
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

# Preprocess
function preprocess_binary(X, threshold=0.5)
    binary_X = Float32.(X) .>= threshold
    return reshape(binary_X, 28 * 28, size(binary_X, 3))  # Return binary images in 784x60000 format
end

function preprocess_binary(X, threshold=0.5)
    binary_X = Float32.(X) .>= threshold  # Create binary dataset (0 or 1)
    return reshape(binary_X, 28 * 28, size(binary_X, 3))  # Return binary images in 784x60000 format
end

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)


function onehot_labels(y)
    return Flux.onehotbatch(y, 0:9)
end

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

# Define nn
nn = Chain(Dense(28^2, 128, relu), Dense(128, 64, relu), Dense(64, 10))

# Loss and optimizer
using Flux.Losses
loss(x, y) = mean(Flux.Losses.logitcrossentropy(nn(x), y))
batch_size = 64  # You can adjust this based on your memory constraints
train_data = DataLoader((train_X_binary, train_y), batchsize=batch_size, shuffle=true)
opt = ADAM(0.001)


# Training
epochs = 10
for epoch in 1:epochs
    for (x, y) in train_data
        Flux.train!(loss, Flux.params(nn), [(x, y)], opt)
    end
    println("Epoch $epoch complete")
end

function accuracy(x, y)
    y_pred = nn(x)
    y_pred = Flux.onecold(y_pred, 0:9)
    y_true = Flux.onecold(y, 0:9)
    return sum(y_pred .== y_true) / length(y_true)
end

println("Test Accuracy: ", accuracy(test_X_binary, test_y))


# Plotting
image_original = train_X[:, :, 1]
image_binary = reshape(train_X_binary[:, 1], 28, 28)

plot_images(image_original, image_binary)