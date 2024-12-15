
function sdp_partial(model, img, ii, jj, num_samples)
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    ii = collect(ii)
    jj = collect(jj)
	x = rand([-1,1], length(img), num_samples)
	x[ii,:] .= img[ii]
	mean(model(x)[jj, :] .== model(img)[jj, :])
end

function backward_for_index(sm::Subset_minimal, I3, index, threshold, num_samples)
    
    subset_for_index = deepcopy(I3)

    max_steps = 100
    steps = 0

    init_sdp = sdp_partial(sm.nn[1], sm.input, I3, index, num_samples)
    stack = [(subset_for_index, init_sdp)]
    closed_list = []
    closed_list = push!(closed_list, (subset_for_index, init_sdp))

    while !isempty(stack)
        steps += 1
        steps > max_steps && break        

        sort!(stack, by = x -> -x[2])

        curr_subset, curr_sdp = pop!(stack)
        for i in curr_subset
            new_subset = pop(curr_subset, i)
            new_sdp = sdp_partial(sm.nn[1], sm.input, new_subset, index, num_samples)
            if new_sdp >= threshold
                subset_for_index = new_subset
                return subset_for_index
            end
            if new_sdp > curr_sdp
                if !(new_subset in closed_list)
                    stack = push!(stack, (new_subset, new_sdp))
                    closed_list = push!(closed_list, (new_subset, new_sdp))
                end
            end
        end
    end
    return subset_for_index
end

function subset_for_I2(sm::Subset_minimal, I3, I2, threshold, num_samples)

    subset_i = backward_for_index(sm, I3, I2[1], threshold, num_samples)
    println("Subset for index $(I2[1]): ", subset_i)

    # for iᵢ in I2
        # subset_i = backward_for_index(sm, I3, iᵢ, threshold, num_samples)
        # println("Subset for index $iᵢ: ", subset_i)
    # end
end