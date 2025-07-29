import ProbAbEx as PAE
using Flux
using Random
using Revise
using Zygote
using Optimisers

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

function get_data()
    train_x, _ = PAE.MNIST.traindata(Float32)
    train_x = reshape(train_x, :, size(train_x, 3))  # 784 × 60000
    train_x = train_x .> 0.5f0                      # бинаризация
    return train_x
end

data = get_data()

function sample_z(μ, logσ)
    ϵ = randn(Float32, size(μ))
    return μ .+ exp.(0.5f0 .* logσ) .* ϵ
end

function kl_div(μq, logσq, μp, logσp)
    σq2 = exp.(logσq)
    σp2 = exp.(logσp)
    kl_element = (logσp .- logσq) .+ (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1
    kl = 0.5 * sum(kl_element) / size(μq, 2)
    return kl
end

function binary_cross_entropy(x̂, x, b)
    x̂ = sigmoid.(x̂)
    mask = b .== 1
    return -sum(x[mask] .* log.(x̂[mask] .+ 1e-7) .+ (1 .- x[mask]) .* log.(1 .- x̂[mask] .+ 1e-7)) / size(x, 2)
end


epochs = 10
batchsize = 128
model = (proposal_net_mu, proposal_net_logσ,
         prior_net_mu, prior_net_logσ, decoder_net)
opt = Optimisers.Adam()
opt_state = Optimisers.setup(opt, Flux.params(model...))

function train!(data, model, opt_state)
    for epoch in 1:epochs
        elbo_total = 0f0

        for i in 1:batchsize:size(data, 2)
            inds = i:min(i + batchsize - 1, size(data, 2))
            x = data[:, inds]                      # x ∈ 784×B
            B = size(x, 2)
            b = rand(Bool, size(x))                # маска b ∈ 784×B
            x_obs = x .* .!b                       # наблюдаемые признаки

            # Входы в proposal и prior: [x; b] ∈ 1568×B
            xb_input = vcat(x, Float32.(b))
            xobs_input = vcat(x_obs, Float32.(b))

            # Вычисляем параметры латентных распределений
            μq = proposal_net_mu(xb_input)
            logσq = proposal_net_logσ(xb_input)
            z = sample_z(μq, logσq)                # z ∈ latent_dim×B

            μp = prior_net_mu(xobs_input)
            logσp = prior_net_logσ(xobs_input)

            # Вход в decoder: [z; x_obs; b] ∈ (latent_dim + 2×784)×B = 1588×B
            dec_input = vcat(z, x_obs, Float32.(b))
            x̂ = decoder_net(dec_input)            # логиты x̂ ∈ 784×B

            # Потери
            loss_recon = binary_cross_entropy(x̂, x, b)
            loss_kl = kl_div(μq, logσq, μp, logσp)
            loss = loss_recon + loss_kl

            # Градиенты и обновление
            # ps = Flux.params(model...)
            # grads = Zygote.gradient(() -> loss, ps)
            # opt_state, new_params = Optimisers.update(opt_state, ps, grads)
            # Flux.loadparams!(model..., new_params)

            elbo_total -= loss
        end

        println("Epoch $epoch | ELBO = $(round(elbo_total, digits=2))")
    end
end

train!(data, model, opt_state)
