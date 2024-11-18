
function preprocess_binary(X, threshold=0.5)
    binary_X = Float32.(X) .>= threshold
    return reshape(binary_X, 28 * 28, size(binary_X, 3))
end

function preprocess_binary(X, threshold=0.5)
    binary_X = Float32.(X) .>= threshold
    return reshape(binary_X, 28 * 28, size(binary_X, 3))
end

function onehot_labels(y)
    return Flux.onehotbatch(y, 0:9)
end

function smoothed_crossentropy(pred, target, epsilon=0.1)
    num_classes = size(target, 1)
    smooth_target = (1 - epsilon) * target .+ epsilon / num_classes
    return mean(Flux.Losses.logitcrossentropy(pred, smooth_target))
end

function input_drouput(x)
    x = round.(x)
    mod.(x .+ (rand(size(x)) .< 0.2), 2)
end

# Training
function train_nn(nn, train_X, train_y, test_X, test_y)

    # Loss and optimizer
    loss(x, y) = mean(Flux.Losses.logitcrossentropy(nn(x), y))
    # loss(x, y) = smoothed_crossentropy(nn(x), y)
    batch_size = 64
    train_data = DataLoader((train_X, train_y), batchsize=batch_size, shuffle=true)
    # opt = ADAM(0.001)
    rule = Optimisers.Adam()  # use the Adam optimiser with its default settings
    state_tree = Optimisers.setup(rule, nn);  # initialise this optimiser's momentum etc.


    epochs = 8
    for epoch in 1:epochs
        for (x, y) in train_data
            # Flux.train!(loss, Flux.params(nn), [(x, y)], opt)
            x = input_drouput(x)
            ∇nn = Flux.Zygote.gradient(model -> Flux.Losses.logitcrossentropy(model(x), y), nn)[1]
            state_tree, nn = Optimisers.update(state_tree, nn, ∇nn);
        end
        println("Epoch $epoch complete")
    end
    println("Test Accuracy: ", accuracy(nn, test_X, test_y) * 100, "%")
    nn
end

function accuracy(model, x, y)
    y_pred = model(x)
    y_pred = Flux.onecold(y_pred, 0:9)
    y_true = Flux.onecold(y, 0:9)
    return sum(y_pred .== y_true) / length(y_true) 
end
