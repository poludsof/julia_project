
function plot_images(original_image::Matrix{Float32}, binary_image::BitMatrix)
    fig = Figure(size = (900, 400))
    
    ax1 = Axis(fig[1, 1], title = "Original MNIST Digit", yreversed = true, aspect = DataAspect())
    image!(ax1, original_image, colormap = :grays, interpolate = false)
    hidespines!(ax1)
    hidedecorations!(ax1)

    ax2 = Axis(fig[1, 2], title = "Binary MNIST Digit", yreversed = true, aspect = DataAspect())
    image!(ax2, binary_image, colormap = [:black, :white], interpolate = false)
    hidespines!(ax2)
    hidedecorations!(ax2)

    fig
end

function plot_mnist_with_active_pixels(mnist_vector::BitVector, active_indices::Set{Int})
    fig = Figure(size = (400, 400))
    img = reshape(mnist_vector, 28, 28)
    highlighted_img = [RGBf(x, x, x) for x in img]

    for idx in active_indices
        row, col = Tuple(CartesianIndices(img)[idx])
        red_intensity = 0.6 + 0.4 * img[row, col]
        highlighted_img[row, col] = RGBf(red_intensity, 0, 0)
    end

    ax = Axis(fig[1, 1], title = "MNIST Digit", yreversed = true, aspect = DataAspect())
    image!(ax, highlighted_img, interpolate = false)
    hidespines!(ax)
    hidedecorations!(ax)

    fig
end
