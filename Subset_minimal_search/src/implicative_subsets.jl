
function backward_for_index(model, xp, ii, index, threshold, num_samples)

    max_steps = 200
    steps = 0

    init_sdp = sdp_partial(model, xp, ii, index, num_samples)
    stack = [(ii, init_sdp)]
    subsets_for_index = [(ii, init_sdp)]
    println("Initial subset for index $index: ", ii, " initial sdp: ", init_sdp)

    closed_list = []
    closed_list = push!(closed_list, (ii, init_sdp))

    while !isempty(stack)
        steps += 1
        steps > max_steps && break        
        # println("Step: $steps")

        sort!(stack, by = x -> x[2])
        # sort!(stack, by = x -> (-length(x[1]), x[2]))

        curr_subset, curr_sdp = pop!(stack)
        # println("Current subset for index $index, length: $(length(curr_subset)), SDP: $curr_sdp")

        for i in curr_subset
            new_subset = pop(curr_subset, i)
            new_sdp = sdp_partial(model, xp, new_subset, index, num_samples)
            # if new_sdp <= threshold && new_sdp >= (threshold-0.05)  # to avoid too bad solutions
            if new_sdp >= threshold
                # println("Add subset ", length(new_subset), " sdp: ", new_sdp)
                subsets_for_index = push!(subsets_for_index, (new_subset, new_sdp))
            end
            if !(new_subset in closed_list)
                stack = push!(stack, (new_subset, new_sdp))
                closed_list = push!(closed_list, (new_subset, new_sdp))
            end
        end
    end

    subsets_for_index = sort!(subsets_for_index, by = x -> (-length(x[1]), x[2]))
    final_subset, final_sdp = pop!(subsets_for_index)
    println("Final subset for index $index: ", final_subset, " sdp: ", final_sdp)
    return final_subset
end


function implicative_subsets(model, xp, prev_layer_I, layer_I; threshold=0.9, num_samples=1000)
    subsets = []
    for i in layer_I
        subset_i = backward_for_index(model, xp, prev_layer_I, i, threshold, num_samples)
        push!(subsets, subset_i)
    end
    return subsets
end