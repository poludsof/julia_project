using Random, Distributions

function bernoulli_mixture_em(data, K, max_iters=70)
    N, D = size(data) 
    Random.seed!(42)

    π = fill(1/K, K)
    θ = rand(K, D)
    
    for iter in 1:max_iters
        println("Iteration: $iter")
        log_prob = [log(π[k]) + sum(log.(θ[k, :] .* data[n, :] .+ (1 .- θ[k, :]) .* (1 .- data[n, :]))) for n in 1:N, k in 1:K]
        log_prob .-= maximum(log_prob, dims=2)
        norm_prob = exp.(log_prob)  
        norm_prob ./= sum(norm_prob, dims=2)

        Nk = sum(norm_prob, dims=1)
        θ = (norm_prob' * data) ./ Nk'
        π = Nk[:] / N
    end

    return π, θ
end

function bernoulli_mixture_em_supervised(data, labels, K)
    N, D = size(data)
    
    π = fill(1/K, K)
    θ = rand(K, D)
    
    norm_prob = zeros(N, K)
    for n in 1:N
        k = labels[n] + 1
        norm_prob[n, k] = 1.0
    end

    Nk = sum(norm_prob, dims=1)
    θ = (norm_prob' * data) ./ Nk'

    return π, θ
end

function preprocess_binary(X)
    return X .> 0.5
end



train_X, train_y = MNIST(split=:train)[:]
X = preprocess_binary(train_X)
X = reshape(X, 28*28, :)' 

K = 10
π, θ = bernoulli_mixture_em(X, K)
# π, θ = bernoulli_mixture_em_supervised(X, train_y, K)



function generate_bin_sample(θ, k)
    println("Generating binary sample from component $k")
    return rand(Float32, size(θ, 2)) .< θ[k, :]
end

function plot_mnist_image(img)
    img_reshaped = reshape(img, 28, 28)'
    fig = Figure(size = (300, 300))
    axis = Axis(fig[1, 1], aspect = 1)

    hidedecorations!(axis)
    axis.xreversed[] = false
    axis.yreversed[] = true

    heatmap!(axis, img_reshaped', colormap = :grays)

    display(fig)
end



generated_image = generate_bin_sample(θ, 6)
plot_mnist_image(generated_image)

# Plotting the learned components
for k in 1:K
    plot_mnist_image(θ[k, :])
    # generated_image = generate_sample(θ, k)
    # plot_mnist_image(generated_image)
end
