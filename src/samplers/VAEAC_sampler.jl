using Flux
using Flux: logitbinarycrossentropy
using MLDatasets: MNIST
using Optimisers
using Statistics
using Random
using LinearAlgebra
using Zygote

# Set random seed for reproducibility
Random.seed!(123)

# VAEAC Model Structure
struct VAEAC
    encoder
    prior
    decoder
    latent_dim::Int
end

# Constructor for VAEAC
function VAEAC(input_dim::Int, hidden_dim::Int, latent_dim::Int)
    # Encoder: Input = [image, mask], Output = [mu, logvar] for latent space
    encoder = Chain(
        Dense(input_dim * 2, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, latent_dim * 2)  # mu and logvar
    )
    
    # Prior: Input = [image, mask], Output = [mu, logvar] for prior
    prior = Chain(
        Dense(input_dim * 2, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, latent_dim * 2)  # mu and logvar
    )
    
    # Decoder: Input = [latent, mask], Output = probabilities for binary pixels
    decoder = Chain(
        Dense(latent_dim + input_dim, hidden_dim, relu),
        Dense(hidden_dim, hidden_dim, relu),
        Dense(hidden_dim, input_dim, sigmoid)  # Output probabilities in [0,1]
    )
    
    VAEAC(encoder, prior, decoder, latent_dim)
end

# Sampling from latent space (reparameterization trick)
function sample_latent(mu, logvar)
    epsilon = randn(Float32, size(mu))
    return mu + exp.(logvar / 2) .* epsilon
end

# VAEAC forward pass (normal function)
function vaeac_forward(encoder, prior, decoder, latent_dim, x::AbstractMatrix, mask::AbstractMatrix)
    # Concatenate input image and mask
    input = vcat(x, mask)
    
    # Encoder: Get mu and logvar for proposal distribution
    enc_output = encoder(input)
    mu_q, logvar_q = enc_output[1:latent_dim, :], enc_output[latent_dim+1:end, :]
    
    # Prior: Get mu and logvar for prior distribution
    prior_output = prior(input)
    mu_p, logvar_p = prior_output[1:latent_dim, :], prior_output[latent_dim+1:end, :]
    
    # Sample latent variable
    z = sample_latent(mu_q, logvar_q)
    
    # Decoder: Reconstruct image (probabilities) using latent variable and mask
    recon = decoder(vcat(z, mask))
    
    return recon, mu_q, logvar_q, mu_p, logvar_p
end

# KL-divergence between two Gaussians
function kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
    kl = 0.5 * sum(@.(logvar_p - logvar_q + (exp(logvar_q) + (mu_q - mu_p)^2) / exp(logvar_p) - 1))
    mean(kl)
end

# Loss function: Binary cross-entropy (on observed pixels) + KL-divergence
function vaeac_loss(encoder, prior, decoder, latent_dim, x, mask)
    recon, mu_q, logvar_q, mu_p, logvar_p = vaeac_forward(encoder, prior, decoder, latent_dim, x, mask)
    
    # Binary cross-entropy loss (only for observed pixels, where mask = 1)
    recon_loss = mean(sum(mask .* logitbinarycrossentropy.(recon, x, agg=identity), dims=1))
    
    # KL-divergence
    kl_loss = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
    
    return recon_loss + kl_loss
end

# Generate random mask for inpainting (e.g., 50% of pixels missing)
function generate_mask(input_dim, batch_size, missing_ratio=0.5)
    mask = rand(Float32, input_dim, batch_size) .> missing_ratio
    Float32.(mask)
end

# Binarize MNIST images
function binarize_images(images)
    return Float32.(images .>= 0.5)
end

# Inpainting function
function inpaint(model, x, mask)
    # Set model to evaluation mode
    Flux.testmode!(model.encoder)
    Flux.testmode!(model.prior)
    Flux.testmode!(model.decoder)
    
    # Forward pass with partial image and mask
    input = x .* mask  # Apply mask to input (zero out unobserved pixels)
    recon, _, _, _, _ = vaeac_forward(model.encoder, model.prior, model.decoder, model.latent_dim, input, mask)
    
    # Binarize reconstructed pixels (threshold at 0.5)
    recon_binary = recon .>= 0.5
    
    # Combine observed and reconstructed pixels
    output = x .* mask + recon_binary .* (1 .- mask)
    
    return output
end

# Load and preprocess MNIST (binarized)
function load_mnist(split=:train, batch_size=64)
    data = MNIST(split, Tx=Float32, dir="MNIST")
    images = reshape(data.features, 28*28, :)
    images = binarize_images(images)  # Binarize pixel values
    labels = data.targets
    loader = Flux.DataLoader((images, labels), batchsize=batch_size, shuffle=true)
    return loader, size(images, 2)
end

# Training function
function train_vaeac(; epochs=20, batch_size=64, learning_rate=1e-3)
    # Model parameters
    input_dim = 28 * 28  # MNIST image size
    hidden_dim = 512
    latent_dim = 32
    
    # Initialize model and optimizer
    model = VAEAC(input_dim, hidden_dim, latent_dim)
    opt = Optimisers.ADAM(learning_rate)
    
    # Set up Optimisers.jl state
    state = Optimisers.setup(opt, (encoder=model.encoder, prior=model.prior, decoder=model.decoder))

    # Load data
    train_loader, _ = load_mnist(:train, batch_size)
    test_loader, _ = load_mnist(:test, batch_size)
    
    # Training loop
    for epoch in 1:epochs
        total_loss = 0.0
        num_batches = 0
        
        for (x, _) in train_loader
            mask = generate_mask(input_dim, size(x, 2), 0.5)
            # Define loss function to take named tuple of parameters
            loss_fn(params) = vaeac_loss(params.encoder, params.prior, params.decoder, model.latent_dim, x, mask)
            # Compute loss and gradients
            println("Compute loss and gradients")
            grads = Zygote.gradient(loss_fn, (encoder=model.encoder, prior=model.prior, decoder=model.decoder))[1]
            loss = loss_fn((encoder=model.encoder, prior=model.prior, decoder=model.decoder))
            # Update optimizer state and parameters
            println("Update optimizer state and parameters")
            state, (encoder, prior, decoder) = Optimisers.update(state, 
                                                               (encoder=model.encoder, prior=model.prior, decoder=model.decoder), 
                                                               grads)
            # Update model with new parameters
            println("Update model with new parameters")
            model = VAEAC(encoder, prior, decoder, model.latent_dim)
            total_loss += loss
            num_batches += 1
        end
        
        avg_loss = total_loss / num_batches
        println("Epoch $epoch, Average Loss: $avg_loss")
        
        # Validation (optional, using test set)
        if epoch % 5 == 0
            val_loss = 0.0
            val_batches = 0
            for (x, _) in test_loader
                mask = generate_mask(input_dim, size(x, 2), 0.5)
                val_loss += vaeac_loss(model.encoder, model.prior, model.decoder, model.latent_dim, x, mask)
                val_batches += 1
            end
            println("Validation Loss: $(val_loss / val_batches)")
        end
    end
    
    return model
end

# Main execution
function main()
    # Train the model
    model = train_vaeac(epochs=20, batch_size=64)
    
    # Test inpainting on a few MNIST images
    test_loader, _ = load_mnist(:test, 10)
    x, _ = first(test_loader)
    mask = generate_mask(28*28, 10, 0.5)
    
    # Perform inpainting
    inpainted = inpaint(model, x, mask)
    
    # Save results (optional, for visualization)
    println("Inpainting completed. Results shape: ", size(inpainted))
end

# Run the model
main()
