
using CSV
using MLDatasets
using DataFrames, Serialization, StaticBitSets, Statistics, CairoMakie, Makie
using Subset_minimal_search

# data = CSV.File("/home/sofia/julia_project/Subset_minimal_search/tests/beam_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/Subset_minimal_search/tests/bcwd_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/Subset_minimal_search/tests/forward_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/tuned_forward_search_uniform_all") |> DataFrame
data = CSV.File("/home/sofia/julia_project/ep_tuned_forward_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/forward_search_bernl_all") |> DataFrame

train_X, train_y = MNIST(split=:train)[:]
train_X_binary = preprocess_binary(train_X)
train_X_bin_neg = preprocess_bin_neg(train_X_binary)

println(size(train_X_bin_neg))

function visualize_image_with_subset(image_idx, subset, train_X_bin_neg)
    fig = Figure(resolution = (600, 600))
    
    image = train_X_bin_neg[:, image_idx]  # Предполагаем, что изображения 28x28
    
    subset_pixels = parse.(Int, split(subset, ","))
    for i in 1:784
        if i in subset_pixels
            image[i] = 9
        elseif image[i] == 1
            image[i] = 15
        elseif image[i] == -1
            image[i] = 0
        end
    end
    
    reshaped_image = reshape(image, 28, 28)

    fig = Figure(resolution = (600, 600))
    ax = Axis(fig[1, 1], aspect = 1, yreversed = true, xgridvisible = false, 
                                                       ygridvisible = false)
                                                       ax.xticklabelsvisible = false   # прячем цифры
    ax.xticksvisible      = false   # прячем сами риски
    ax.yticksvisible      = false
    ax.xticklabelsvisible = false   # прячем цифры
    ax.yticklabelsvisible = false
    heatmap!(fig[1, 1], reshaped_image, colormap=:inferno)
    # Plots.heatmap(reshaped_image, color=:inferno, aspect_ratio=1, colorbar=false)
    fig
end

function visualize_binary_mnist(image_idx, dataset)
    fig = Figure(resolution = (600, 600))
    
    image = dataset[:, image_idx]  # Предполагаем, что изображения 28x28
    reshaped_image = reshape(image, 28, 28)

    fig = Figure(resolution = (600, 600))
    ax = Axis(fig[1, 1], aspect = 1, yreversed = true, xgridvisible = false, 
                                                       ygridvisible = false)
                                                       ax.xticklabelsvisible = false   # прячем цифры
    ax.xticksvisible      = false   # прячем сами риски
    ax.yticksvisible      = false
    ax.xticklabelsvisible = false   # прячем цифры
    ax.yticklabelsvisible = false
    heatmap!(fig[1, 1], reshaped_image, colormap=:inferno)
    fig
end
# image_idx = 1

for image_idx in 1:nrow(data)
    # println("image_idx: ", image_idx)
    image = data[image_idx, :image]
    subset = data[image_idx, :solution]
    err = data[image_idx, :ϵ]
    num_samples = data[image_idx, :num_samples]
    # println("image: ", subset)
    if image == 1 && err != 0.8
        fig = visualize_image_with_subset(2, subset, train_X_bin_neg)
        save("tuned_frwd_image_2_$(err)_$(num_samples).png", fig)
    end
end