using Plots

function rotated_image(image::AbstractArray{T, 2}) where T
    rotated_image = zeros(T, 28, 28)
    for i in 1:28
        for j in 1:28
            rotated_image[28 - i + 1, j] = image[j, i] 
        end
    end
    rotated_image
end

function plot_images(original_image::Matrix{Float32}, binary_image::BitMatrix)
    # Plot the original image
    p1 = heatmap(rotated_image(original_image), color=:grays, axis=false, title="Original MNIST Digit")
    # Plot the binary image
    p2 = heatmap(rotated_image(binary_image), color=[RGB(0, 0, 0), RGB(1, 1, 1)], axis=false, title="Binary MNIST Digit")
    
    # Combine the two plots
    plot(p1, p2, layout=(1, 2), size=(900, 400))
end