import ProbAbEx as PAE
using Flux
using Random

train_X, train_y = PAE.MNIST(split=:train)[:]
test_X, test_y = PAE.MNIST(split=:test)[:]

train_X_binary = PAE.preprocess_binary(train_X)
test_X_binary = PAE.preprocess_binary(test_X)

train_X_bin_neg = PAE.preprocess_bin_neg(train_X_binary)
test_X_bin_neg = PAE.preprocess_bin_neg(test_X_binary)

train_y = PAE.onehot_labels(train_y)
test_y = PAE.onehot_labels(test_y)


input_dim = 784
latent_dim = 20

# encoder qϕ(z | x, b)
proposal_net_mu = Chain(Dense(2input_dim, 400, relu), Dense(400, latent_dim))
proposal_net_logσ = Chain(Dense(2input_dim, 400, relu), Dense(400, latent_dim))

# prior pψ(z | x₁₋b, b)
prior_net_mu = Chain(Dense(2input_dim, 400, relu), Dense(400, latent_dim))
prior_net_logσ = Chain(Dense(2input_dim, 400, relu), Dense(400, latent_dim))

# decoder pθ(x_b | z, x₁₋b, b)
decoder_net = Chain(Dense(latent_dim + 2input_dim, 400, relu), Dense(400, input_dim))


function sample_z(μ, logσ)
    ϵ = randn(Float32, size(μ))
    return μ .+ exp.(0.5f0 .* logσ) .* ϵ
end

function kl_div(μq, logσq, μp, logσp)
    σq2 = exp.(logσq)
    σp2 = exp.(logσp)
    kl = 0.5f0 * sum((logσp .- logσq) .+ (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1) / size(μq, 2)
    return kl
end

function binary_cross_entropy(x̂, x, b)
    x̂ = sigmoid.(x̂)
    mask = b .== 1
    return -sum(x[mask] .* log.(x̂[mask] .+ 1e-7f0) .+ (1 .- x[mask]) .* log.(1 .- x̂[mask] .+ 1e-7f0)) / size(x, 2)
end

