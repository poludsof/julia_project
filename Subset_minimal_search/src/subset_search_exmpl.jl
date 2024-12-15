
function sdp_partial(model, img, ii, jj, num_samples)
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    ii = collect(ii)
    jj = collect(jj)
	x = rand([-1,1], length(img), num_samples)
	x[ii,:] .= img[ii]
	mean(model(x)[jj, :] .== model(img)[jj, :])
end

function backward_for_index(model, xp, ii, index, threshold, num_samples)
    println("Initial subset for index $index: ", ii)

    max_steps = 200
    steps = 0

    # init_sdp = sdp_partial(sm.nn[1], sm.input, ii, index, num_samples)
    init_sdp = sdp_partial(model, xp, ii, index, num_samples)
    stack = [(ii, init_sdp)]
    subsets_for_index = [(ii, init_sdp)]

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
            # new_sdp = sdp_partial(sm.nn[1], sm.input, new_subset, index, num_samples)
            new_sdp = sdp_partial(model, xp, new_subset, index, num_samples)
            if new_sdp <= threshold && new_sdp >= 0.95
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


function subset_for_I2(sm::Subset_minimal, I3, I2, threshold, num_samples)
    subset_all = []

    for iᵢ in I2
        subset_i = backward_for_index(sm.nn[1], sm.input, I3, iᵢ, threshold, num_samples)
        push!(subset_all, subset_i)
    end

    return subset_all
end

function subset_for_I1(sm::Subset_minimal, I3, I2, I1, threshold, num_samples)
    subset_all = []

    for iᵢ in I1
        subset_i = backward_for_index(sm.nn[2], sm.nn[1](sm.input), I2, iᵢ, threshold, num_samples)
        push!(subset_all, subset_i)
    end

    return subset_all
end