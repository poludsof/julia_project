using DataFrames, Serialization, StaticBitSets, Statistics, CairoMakie
using Flux
import ProbAbEx as PAE
# using MakieExtra
using CairoMakie
using ProbAbEx
using ProbAbEx: condition
using ProbAbEx.TimerOutputs
include("visualization_tools.jl")

"""
    add activations in the skeleton, which are union of all activations of the neurons
"""

function add_input_activatations(skeleton)
    a = Any[[merge(x, (;activations = x.rule)) for x in skeleton[1]]]
    for l in 2:length(skeleton)
        layer = map(skeleton[l]) do unit
            inputs = filter(u -> u.j ∈ unit.rule, a[l-1])
            merge(unit, (;activations = mapreduce(x -> x.activations, union, inputs)))
        end
        push!(a, layer)
    end
    a
end


# This utility function computes the "east" and "west" anchors (in latex parlance)
function _compute_left_right_anchors(bbox_obs)
    left_anchor = @lift(Point2($(bbox_obs).origin[1], $(bbox_obs).origin[2] + $(bbox_obs).widths[2] / 2))
    right_anchor = @lift(Point2($(bbox_obs).origin[1] + $(bbox_obs).widths[1], $(bbox_obs).origin[2] + $(bbox_obs).widths[2] / 2))
    # return NamedTuple of observables
    return (; left = left_anchor, right = right_anchor)
end


function plot_skeleton(rule, skeleton, xₛ, samplers;explanation_color = :red)
    input_sampler = first(samplers)
    skeleton = add_input_activatations(skeleton)

    fig = Figure(size = ((1 + length(skeleton))*800, maximum(length.(skeleton))*500));
    # Number of neurons per layer
    neurons_per_column = [1, length(skeleton[1]), length(skeleton[2]), length(skeleton[3])]

    # A grid for each layer (arranged in a column)
    col_grids = [GridLayout(fig[1, i]; valign = :center) for i in eachindex(neurons_per_column)]
    # An axis per neuron
    col_axes = [
        [Axis(col_grids[i][j, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0)) for j in 1:neurons_per_column[i]]
        for i in eachindex(neurons_per_column)
    ]

    # Plot into the axes
    for (col_idx, col_of_axes) in enumerate(col_axes)
        if col_idx == 1
            ax = col_of_axes[1]
            dotted_digit!(ax, xₛ)
            ax.aspect[] = DataAspect()
            hidedecorations!(ax)
            hidespines!(ax)
            continue
        end
        for (row_idx, ax) in enumerate(col_of_axes)
            # ax.yreversed[] = true
            ii = skeleton[col_idx-1][row_idx].activations
            println(ii)
            if isempty(ii)
                ax.aspect[] = DataAspect()
                hidedecorations!(ax)
                hidespines!(ax)
                continue
            end
            add_rule!(ax, input_sampler, xₛ, ii; explanation_color)
            ax.aspect[] = DataAspect()
            hidedecorations!(ax)
            hidespines!(ax)
        end
    end

    # Size the columns that are not the input/output columns appropriately
    for (nlayers, grid) in collect(zip(neurons_per_column, col_grids))[begin+1:end-1]
        grid.tellheight[] = false
        grid.height[] = Relative(nlayers / maximum(neurons_per_column))
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
    for layer_idx in 2:length(skeleton)
        for (dest_idx, dest_unit) in enumerate(skeleton[layer_idx])
            source_idxs = [i for (i, u) in enumerate(skeleton[layer_idx-1]) if u.j ∈ dest_unit.rule]
            for source_idx in source_idxs
                lines!(
                    fig.scene, 
                    lift(axis_anchors[layer_idx][source_idx].right, axis_anchors[layer_idx+1][dest_idx].left) do start, stop
                        rel = stop - start
                        c1 = Point2(start[1] + rel[1]/4, start[2])
                        c2 = Point2(stop[1] - rel[1]/4, stop[2])
                        return BezierPath([MoveTo(start), CurveTo(c1, c2, stop)])
                    end;
                    color = :black,
                    linewidth = 3,
                    # markersize = 20,
                )
            end
        end
    end

    for dest_idx in 1:length(skeleton[1])
        lines!(
            fig.scene, 
            lift(axis_anchors[1][1].right, axis_anchors[2][dest_idx].left) do start, stop
                rel = stop - start
                c1 = Point2(start[1] + rel[1]/4, start[2])
                c2 = Point2(stop[1] - rel[1]/4, stop[2])
                return BezierPath([MoveTo(start), CurveTo(c1, c2, stop)])
            end;
            color = :black,
            linewidth = 3,
            # markersize = 20,
        )
    end

    colsize!(fig.layout, 1, Relative(1/6.5))
    colsize!(fig.layout, length(axis_anchors), Relative(1/6.5))
    fig
end

# centers = deserialize(joinpath("mnist", "binary_model_hidden_centers.jls"))
# samplers = (BernoulliMixture(centers[:,1:784,:]), BernoulliMixture(centers[:,785:1040,:]), BernoulliMixture(centers[:,1041:end,:]))
# rule, skeleton, xₛ, _ = deserialize("mnist/skeleton_90.jls")

# plot_skeleton(rule, skeleton, xₛ, samplers)


#!
centers = deserialize("/home/sofia/julia_project/ml_in_prague/mnist/binary_model_hidden_centers.jls")
# # centers = deserialize(joinpath("mnist", "binary_model_hidden_centers.jls"))
samplers = (BernoulliMixture(centers[:,1:784,:]), BernoulliMixture(centers[:,785:1040,:]), BernoulliMixture(centers[:,1041:end,:]))
# # rule, skeleton, xₛ, _ = deserialize("mnist/skeleton_90.jls")
rule, skeleton, xₛ, _ = deserialize("/home/sofia/julia_project/ml_in_prague/mnist/skeleton_100_new.jls")
# println("rule: ", rule)
# println("skeleton: ", skeleton)
# print_skeleton(rule, skeleton)
# # println("xₛ: ", xₛ)
fig = plot_skeleton(rule, skeleton, xₛ, samplers)
# println(fig)

save("pevny_skeleton3.png", fig)

function create_brule(II, set)
    for item in set
        II = push(II, item)
    end
    return II
end
empty_sbitset(n::Int) = SBitSet{ceil(Int, n / 64), UInt64}()
empty_sbitset(x::AbstractArray) = empty_sbitset(length(x))
using CSV
# data = CSV.File("/home/sofia/julia_project/forward_search_bernl_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/tests/beam_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/tests/bcwd_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/tests/forward_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/tuned_forward_search_uniform_all") |> DataFrame
# data = CSV.File("/home/sofia/julia_project/ep_tuned_forward_search_uniform_all") |> DataFrame

# for image_idx in 1:nrow(data)
#     image_digit2 = data[image_idx, :image]
#     image_digit = 2
#     image2 = train_X_binary[:, image_digit2]
#     image = train_X_binary[:, image_digit]
#     # label = argmax(model(image)) - 1
#     rule_string = data[image_idx, :solution]
#     err = data[image_idx, :ϵ]
#     num_samples = data[image_idx, :num_samples]
#     II = parse.(Int, split(rule_string, ","))
#     # if argmax(model(image2)) - 1 != 0
#     #     println("image $image_digit2 is ", argmax(model(image2)) - 1, " but should be 0")
#     #     continue
#     # end
#     if err > 0.9 && num_samples == 10000
#         II = create_brule(empty_sbitset(784), II)
#         fig2 = plot_rule(image, II)
#         filename = "p2_frwd_img$(image_digit2)_$(err)_$(num_samples).png"
#         save(filename, fig2)
#         println("image saved as $filename")
#     end
# end