using Flux
using Flux: Dense, Chain, logitcrossentropy, σ
using MLDatasets
using Random
using Statistics: mean
using Distributions
using MLUtils: eachbatch
using Flux.Optimise: update!, Adam
using CairoMakie

# ========== Hyperparameters ==========
input_dim = 28 * 28
latent_dim = 20
hidden_dim = 400
batch_size = 100
epochs = 10
lr = 1e-3

# ========== Load & binarize MNIST ==========
function load_binary_mnist()
    train_x, _ = MNIST.traindata()
    train_x = Float32.(reshape(train_x, :, size(train_x, 3)) .> 0.5)  # binarize
    return train_x
end

# ========== Mask generation ==========
function generate_mask(size_tuple)
    # Create random binary mask with 0.5 probability known
    return Float32.(rand(Bool, size_tuple))
end

# ========== KL divergence for diagonal Gaussians ==========
function kl_div_diag_gaussians(μq, logσq, μp, logσp)
    σq2 = exp.(2f0 .* logσq)
    σp2 = exp.(2f0 .* logσp)
    kl = 0.5f0 * sum(
        (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1f0 .+ 2f0 .* (logσp .- logσq)
    )
    return kl
end

# ========== Model Definition ==========
function create_model()
    proposal_net = Chain(
        Dense(2 * input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 2 * latent_dim)
    )

    prior_net = Chain(
        Dense(input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, 2 * latent_dim)
    )

    decoder_net = Chain(
        Dense(latent_dim + input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, input_dim)
    )

    return proposal_net, prior_net, decoder_net
end

# ========== Forward pass ==========
function forward(x, mask, proposal_net, prior_net, decoder_net)
    x_masked = x .* mask
    xb = vcat(x_masked, mask)

    proposal_out = proposal_net(xb)
    μ_q = proposal_out[1:latent_dim, :]
    logσ_q = proposal_out[latent_dim+1:end, :]

    ε = randn(Float32, latent_dim, size(x, 2))
    z = μ_q .+ exp.(logσ_q) .* ε

    prior_out = prior_net(mask)
    μ_p = prior_out[1:latent_dim, :]
    logσ_p = prior_out[latent_dim+1:end, :]

    decoder_input = vcat(z, mask)
    logits = decoder_net(decoder_input)

    return logits, μ_q, logσ_q, μ_p, logσ_p
end

# ========== Loss function ==========
function loss_fn(x, mask, proposal_net, prior_net, decoder_net)
    logits, μ_q, logσ_q, μ_p, logσ_p = forward(x, mask, proposal_net, prior_net, decoder_net)
    recon_loss = sum(Flux.binarycrossentropy.(σ.(logits), x) .* (1f0 .- mask)) / size(x, 2)

    kl = 0f0
    for i in 1:size(x, 2)
        kl += kl_div_diag_gaussians(μ_q[:, i], logσ_q[:, i], μ_p[:, i], logσ_p[:, i])
    end
    kl /= size(x, 2)
    return recon_loss + kl
end

# ========== Training ==========
function train()
    proposal_net, prior_net, decoder_net = create_model()
    ps = Flux.params(proposal_net, prior_net, decoder_net)
    opt = Adam(lr)
    data = load_binary_mnist()

    for epoch in 1:epochs
        total_loss = 0f0
        batches = 0

        for x_batch in eachbatch(data; size=batch_size, shuffle=true)
            mask = generate_mask(size(x_batch))

            grads = gradient(ps) do
                loss_fn(x_batch, mask, proposal_net, prior_net, decoder_net)
            end

            update!(opt, ps, grads)

            total_loss += loss_fn(x_batch, mask, proposal_net, prior_net, decoder_net)
            batches += 1
        end

        @info "Epoch $epoch, Avg loss: $(total_loss / batches)"
    end
    return proposal_net, prior_net, decoder_net
end

# ========== Run training ==========
proposal_net, prior_net, decoder_net = train()

function sample_and_save(proposal_net, prior_net, decoder_net)
    x = rand(Float32, 784, 1)
    mask = falses(784, 1)  # Hide all pixels
    logits, _, _, _, _ = forward(x, mask, proposal_net, prior_net, decoder_net)
    x_hat = σ.(logits)

    # img = Gray.(reshape(x_hat[:, 1], 28, 28))
    # save("reconstructed.png", img)
    fig = Figure(size = (400, 400))
    img = reshape(x_hat[:, 1], 28, 28)

    ax1 = Axis(fig[1, 1], title = "Original MNIST Digit", yreversed = true, aspect = DataAspect())
    image!(ax1, img, colormap = :grays, interpolate = false)
    hidespines!(ax1)
    hidedecorations!(ax1)

    fig
end

fig = sample_and_save(proposal_net, prior_net, decoder_net)

function sample_and_save2(proposal_net, prior_net, decoder_net)
    x = rand(Float32, 784, 1)
    mask = falses(784, 1)
    mask[100:101] .= true  # для примера: скрываем часть пикселей
    mask[600:601] .= true  # для примера: скрываем часть пикселей

    logits, _, _, _, _ = forward(x, mask, proposal_net, prior_net, decoder_net)
    x_hat = σ.(logits)

    grayscale_image = reshape(x_hat[:, 1], 28, 28)
    mask_image = reshape(mask[:, 1], 28, 28)

    color_image = Array{RGBf}(undef, 28, 28)

    for i in 1:28, j in 1:28
        if mask_image[i, j]
            color_image[i, j] = RGBf(1, 0, 0)  # красный
        else
            gray_val = clamp(grayscale_image[i, j], 0f0, 1f0)
            color_image[i, j] = RGBf(gray_val, gray_val, gray_val)
        end
    end

    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], title = "Masked MNIST Reconstruction", yreversed = true, aspect = DataAspect())
    image!(ax, color_image, interpolate = false)
    hidespines!(ax)
    hidedecorations!(ax)

    fig
end

xₛ = train_X_bin_neg[:, 1]
fig = sample_and_save2(proposal_net, prior_net, decoder_net)