using CairoMakie

fig = Figure()
# Number of neurons per layer
neurons_per_column = [1, 10, 7, 5, 8, 1]
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
        heatmap!(axis, rand(10, 10); colormap = :plasma)
    end
end
# Size the columns that are not the input/output columns appropriately
for (nlayers, grid) in collect(zip(neurons_per_column, col_grids))[begin+1:end-1]
    grid.tellheight[] = false
    grid.height[] = Relative(nlayers / maximum(neurons_per_column))
end

# Connect random neurons
# First, figure out what neurons to connect
important_neurons = [rand(1:n, 2) for n in neurons_per_column]
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
for layer_idx in collect(eachindex(neurons_per_column))[1:end-1]
    neurons_for_this_layer = important_neurons[layer_idx]
    neurons_for_next_layer = important_neurons[layer_idx+1]
    for source_idx in neurons_for_this_layer
        for dest_idx in neurons_for_next_layer
            lines!(
                fig.scene, 
                lift(axis_anchors[layer_idx][source_idx].right, axis_anchors[layer_idx + 1][dest_idx].left) do start, stop
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

