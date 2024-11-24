
# function rotated_image(image::AbstractArray{T, 2}) where T
#     rotated_image = zeros(T, 28, 28)
#     for i in 1:28
#         for j in 1:28
#             rotated_image[28 - i + 1, j] = image[j, i] 
#         end
#     end
#     rotated_image
# end

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
    img = reshape(mnist_vector, 28, 28)'

    background_img = [RGB(x, x, x) for x in img]

    for idx in active_indices
        row = div(idx - 1, 28) + 1
        col = (idx - 1) % 28 + 1

        red_intensity = 0.6 + 0.4 * img[row, col]
        background_img[row, col] = RGB(red_intensity, 0, 0)
    end

    fig = Figure(resolution = (400, 400))
    ax = Axis(fig[1, 1], aspect = 1)
    image!(ax, background_img)
    hidespines!(ax)
    hidedecorations!(ax)

    display(fig)
end


function test_makie(mnist_vector::BitVector, active_indices::Set{Int})
    img = reshape(mnist_vector, 28, 28)

    background_img = copy(img)

    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], yreversed = true, aspect = DataAspect())

    image!(ax, background_img, colormap = :grays, interpolate = false)

    # for idx in active_indices
    #     row = div(idx - 1, 28) + 1
    #     col = (idx - 1) % 28 + 1

    #     red_intensity = 0.6 + 0.4 * img[row, col]
    #     background_img[row, col] = red_intensity
    # end

    # # Re-apply the image with red highlighting after the update
    # image!(ax, background_img, colormap = :reds, interpolate = false)

    hidespines!(ax)
    hidedecorations!(ax)

    fig
end