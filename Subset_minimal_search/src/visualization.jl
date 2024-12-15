using CairoMakie
println("Subsubset I2: ", subsubset_I2)

I3 = [4,55,71,72,79,100,101,102,103,109,133,145,163,169,191,200,201,229,232,242,257,264,268,269,296,300,301,302,310,312,319,320,330,332,333,359,360,361,391,412,431,447,456,588,602,604,606,626,648,669,670,704,705,735,736,778]
I2 = [2,27,42,52,55,67,69,82,83,84,85,98,104,156,183,186,191,197,209,213,224,234]
I1 = [8,9,12,15,23,28,31,56,63,67,80,89,92,110,124,126,151,154,176,189,196,203,208,235,240,246,248,252,255]


fig = Figure(resolution = (2000, 1800))
# Number of neurons per layer
neurons_per_column = [1, length(I2), length(I1), 1]
# neurons_per_column = [1, 4, 3, 1]

# A grid for each layer (arranged in a column)
col_grids = [GridLayout(fig[1, i]; valign = :center) for i in eachindex(neurons_per_column)]
# An axis per neuron
col_axes = [
    [Axis(col_grids[i][j, 1]) for j in 1:neurons_per_column[i]]
    for i in eachindex(neurons_per_column)
]

# Plot into the axes
for col_of_axes in col_axes
    for axis in col_of_axes
        hidedecorations!(axis)
        hidespines!(axis)
        axis.aspect[] = DataAspect()
        heatmap!(axis, rand(28, 28); colormap = :plasma)
    end
end
# Size the columns that are not the input/output columns appropriately
l = 1
for (nlayers, grid) in collect(zip(neurons_per_column, col_grids))[begin+1:end-1]
    grid.tellheight[] = false
    grid.height[] = Relative(nlayers / maximum(neurons_per_column))
    # if l == 1
    #     grid.tellheight[] = false
    #     grid.height[] = Relative(nlayers / maximum(neurons_per_column) + 0.05)
    # else
    #     grid.height[] = Relative(nlayers / maximum(neurons_per_column))
    # end
    # l += 1
end

# Connect random neurons
# First, figure out what neurons to connect
important_neurons = [rand(1:n, 2) for n in neurons_per_column]
important_neurons= [[[1]], 
                    [[1], [1], [1], [1]],
                    [[1], [2, 3], [3]], 
                    [[1]]]
# println(important_neurons)
# I2_subsubsets = subsubset_I2
important_neurons[2] = [[1] for i in collect(1:length(I2))]
important_neurons[3] = [[] for i in collect(1:length(I1))]
important_neurons[4] = [[1:length(I1)...]]

for i in 1:length(I1)
    for j in subsubset_I1[i]
        for iii in collect(1:length(I2))
            if j == I2[iii]
                println("Found for $j index is $iii in I2")
                push!(important_neurons[3][i], iii)
                break
            end
        end
    end
end
println(subsubset_I1[1])
println(important_neurons[4])

# This utility function computes the "east" and "west" anchors (in latex parlance)
function _compute_left_right_anchors(bbox_obs)
    left_anchor = @lift(Point2($(bbox_obs).origin[1], $(bbox_obs).origin[2] + $(bbox_obs).widths[2] / 2))
    right_anchor = @lift(Point2($(bbox_obs).origin[1] + $(bbox_obs).widths[1], $(bbox_obs).origin[2] + $(bbox_obs).widths[2] / 2))
    # return NamedTuple of observables
    return (; left = left_anchor, right = right_anchor)
end

# Compute anchors-as-observables for each axis
axis_anchors = [
    [
        _compute_left_right_anchors(axis.scene.viewport)
        for axis in col_of_axes
    ]
    for col_of_axes in col_axes
]
# Connect them up!
# neurons_per_column = [1, 4, 3, 1]
for layer_idx in collect(eachindex(neurons_per_column))[2:end] # 1:3
    neurons_for_this_layer = important_neurons[layer_idx]
    println("layer: ", layer_idx)
    for dest_idx in collect(1:neurons_per_column[layer_idx])
        println("neuron: $dest_idx, ", "its subset for prev layer: ", neurons_for_this_layer[dest_idx])
        for source_idx in neurons_for_this_layer[dest_idx]
            println("source: ", source_idx)
            lines!(
                fig.scene, 
                lift(axis_anchors[layer_idx-1][source_idx].right, axis_anchors[layer_idx][dest_idx].left) do start, stop
                    rel = stop - start
                    c1 = Point2(start[1] + rel[1]/4, start[2])
                    c2 = Point2(stop[1] - rel[1]/4, stop[2])
                    return BezierPath([MoveTo(start), CurveTo(c1, c2, stop)])
                end;
                color = :black,
            )
        end
    end
end

fig

