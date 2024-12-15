using CairoMakie

# Length of ii: (71, 25, 31), full_error: (hsum = 0.0, hmax = 0.0)
I3 = [4,21,46,55,71,72,79,100,101,102,103,109,116,133,134,145,158,163,169,191,197,200,201,229,232,242,257,262,264,267,268,269,296,300,301,302,310,312,319,320,330,332,333,359,360,361,391,404,412,431,447,456,538,588,602,604,606,626,627,648,667,669,670,679,698,704,705,735,736,778,781]
I2 = [2,27,42,52,55,60,67,69,82,83,84,85,98,104,147,156,183,186,191,197,209,213,224,230,234]
I1 = [8,9,12,15,19,23,28,31,52,56,63,67,80,89,92,110,124,126,151,154,176,189,196,203,208,235,240,246,248,252,255]

fig = Figure()
# Number of neurons per layer
neurons_per_column = [1, 25, 31, 1]
# A grid for each layer (arranged in a column)
col_grids = [GridLayout(fig[1, i]; valign = :center) for i in eachindex(neurons_per_column)]
# An axis per neuron
col_axes = [
    [Axis(col_grids[i][j, 1]) for j in 1:neurons_per_column[i]]
    for i in eachindex(neurons_per_column)
]

for col_of_axes in col_axes
    for axis in col_of_axes
        axis.width[] = Relative(0.5)
        axis.height[] = Relative(6)
    end
end

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

