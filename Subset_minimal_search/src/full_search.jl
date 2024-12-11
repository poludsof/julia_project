
# function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet, n::Int) 
#     generate_random_img_with_fix_inputs(sm, collect(fix_inputs), n)
# end

# function generate_random_img_with_fix_inputs(model, input, ii::Vector{<:Integer}, n::Int) 
#     # x = rand(0:1, length(sm.input), n)
#     # x[ii, :] .= sm.input[ii]
#     # x
#     x = rand([-1,1], length(sm.input), n)
# 	x[ii,:] .= sm.input[ii]
#     x 
# end

function sdp_full(model, img, ii, num_samples)
    # x = generate_random_img_with_fix_inputs(model, input, fix_inputs, num_samples)
    ii = collect(ii)
    x = rand([-1,1], length(img), num_samples)
	x[ii,:] .= img[ii]
	mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

function sdp_partial(model, img, ii, jj, num_samples) # ii = I2 --> jj = I1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    ii = collect(ii)
    jj = collect(jj)
	x = rand([-1,1], length(img), num_samples)
	x[ii,:] .= img[ii]
	mean(model(x)[jj, :] .== model(img)[jj, :])
end


function beam_search(model, img, fix_inputs::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:length(img)
        if !(i in fix_inputs)
            new_set_I = SBitSet{N, T}()
            new_set_I = union(fix_inputs, SBitSet{32, UInt32}(i))
            threshold = sdp_full(model, img, new_set_I, num_samples) # criteruim (ep/sdp)
            if threshold >= worst_from_best_threshold
                push!(best_results, (new_set_I, threshold))
                if length(best_results) > num_best
                    sort!(best_results, by=x->x[2], rev=true)
                    pop!(best_results)
                    worst_from_best_threshold = best_results[end][2]
                end
            end
        end
    end

    return best_results
end


function full_beam_search(sm::Subset_minimal, threshold_total_err=0.1, num_best=1, num_samples=100)
    I3 = beam_search(sm.nn, sm.input, SBitSet{32, UInt32}(), num_best, num_samples)
    I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), SBitSet{32, UInt32}(), num_best, num_samples)
    I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), SBitSet{32, UInt32}(), num_best, num_samples)

    full_error = max_error(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))

    length = 1
    while full_error < threshold_total_err
        I3 = beam_search(sm.nn, sm.input, I3[1][1], num_best, num_samples)
        
        if length < 256
            I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), I2[1][1], num_best, num_samples)
            I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), I1[1][1], num_best, num_samples)
        else
            println("length is greater than 256")
        end


        full_error = max_error(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))
        println("Length of ii: $length, full_error: ", full_error)    
        length += 1
    end
    return I3[1][1], I2[1][1], I1[1][1]
end



function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

function full_beam_search2(sm::Subset_minimal, threshold_total_err=0.1, num_samples=100)
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    full_error = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)

    while full_error > threshold_total_err
        candidate = add_best(sm, (I3,I2,I1))[1]
        I3, I2, I1 = candidate.ii

        full_error = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)
        println("Length of ii: $((length(I3), length(I2), length(I1))), full_error: ", full_error, " heuristic: ", candidate.h)
    end
    return I3, I2, I1
end


function add_best(sm::Subset_minimal, (I3, I2, I1), num_samples = 100)
    # searching through I3
    ii₁ = map(setdiff(1:784, I3)) do i
        Iᵢ = push(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm.nn, sm.input, (Iᵢ, I2, I1)), num_samples)
    end
    ii₂ = map(setdiff(1:256, I2)) do i
        Iᵢ = push(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm.nn, sm.input, (I3, Iᵢ, I1)), num_samples)
    end

    ii₃ = map(setdiff(1:256, I1)) do i
        Iᵢ = push(I1, i)
        (;ii = (I3,I2, Iᵢ), h = heuristic(sm.nn, sm.input, (I3, I2, Iᵢ)), num_samples)
    end
    sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h < j.h)
end


function heuristic(model, xp, (I3, I2, I1), num_samples = 1000)
	max(0, 0.9 - sdp_full(model, xp, I3, num_samples)) +
	max(0, 0.9 - sdp_full(model[2:3], model[1](xp), I2, num_samples)) +
	max(0, 0.9 - sdp_full(model[3], model[1:2](xp), I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1], xp, I3, I2, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1:2], xp, I3, I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[2], model[1](xp), I2, I1, num_samples))
end


function max_error(model, xp, (I3, I2, I1), num_samples = 1000)
    max(max(0, 1.0 - sdp_full(model, xp, I3, num_samples)),
        max(0, 1.0 - sdp_full(model[2:3], model[1](xp), I2, num_samples)),
        max(0, 1.0 - sdp_full(model[3], model[1:2](xp), I1, num_samples)),
        max(0, 1.0 - sdp_partial(model[1], xp, I3, I2, num_samples)),
        max(0, 1.0 - sdp_partial(model[1:2], xp, I3, I1, num_samples)),
        max(0, 1.0 - sdp_partial(model[2], model[1](xp), I2, I1, num_samples))
    )
end


function backward_dfs_search(sm::Subset_minimal, subsets, _total_err=0.01, num_samples=100)
    I3, I2, I1 = subsets
    initial_total_err = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)
    println("Initial max error: ", initial_total_err)

    stack = [(heuristic(sm.nn, sm.input, (I3, I2, I1), num_samples), (I3, I2, I1))]  # Stack with heuristic value first
    closed_list = Set{Tuple{SBitSet, SBitSet, SBitSet}}()
    best_subsets = (I3, I2, I1)

    # KAPACITA STAKU
    while !isempty(stack)
        sort!(stack, by = x -> -x[1])
        current_heuristic, current_subsets = pop!(stack)
        
        if current_subsets in closed_list
            continue
        end

        push!(closed_list, current_subsets)

        current_error = max_error(sm.nn, sm.input, current_subsets, num_samples)
        println("length: $((length(current_subsets[1]), length(current_subsets[2]), length(current_subsets[3]))) Current error: ", current_error, " Current heuristic: ", current_heuristic)

        if current_error <= _total_err
            best_subsets = current_subsets
            println("Valid subset found: $((length(current_subsets[1]), length(current_subsets[2]), length(current_subsets[3]))) with error: ", current_error)
        else
            I3, I2, I1 = deepcopy(current_subsets)
            
            # Expand I3
            for i in collect(I3)
                new_subsets = (pop(I3, i), I2, I1)
                new_heuristic = heuristic(sm.nn, sm.input, new_subsets, num_samples)
                if new_heuristic < current_heuristic && new_subsets ∉ closed_list
                    push!(stack, (new_heuristic, new_subsets))  # Push with heuristic value
                end
            end

            # Expand I2
            for i in collect(I2)
                new_subsets = (I3, pop(I2, i), I1)
                new_heuristic = heuristic(sm.nn, sm.input, new_subsets, num_samples)
                if new_heuristic < current_heuristic && new_subsets ∉ closed_list
                    push!(stack, (new_heuristic, new_subsets))  # Push with heuristic value
                end
            end

            # Expand I1
            for i in collect(I1)
                new_subsets = (I3, I2, pop(I1, i))
                new_heuristic = heuristic(sm.nn, sm.input, new_subsets, num_samples)
                if new_heuristic < current_heuristic && new_subsets ∉ closed_list
                    push!(stack, (new_heuristic, new_subsets))  # Push with heuristic value
                end
            end
        end
       
    end
    
    return best_subsets
end



function full_beam_search_with_stack(sm::Subset_minimal, threshold_total_err=0.1, num_samples=100)
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    
    full_error = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)
    initial_heuristic = heuristic(sm.nn, sm.input, (I3, I2, I1), num_samples)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, (I3, I2, I1))]

    while !isempty(stack)
        sort!(stack, by = x -> -x[1]) 
        current_heuristic, current_error, (current_I3, current_I2, current_I1) = pop!(stack)
        
        if current_error <= threshold_total_err
            println("Valid subset found: $((length(current_I3), length(current_I2), length(current_I1))) with error: ", current_error)
            return current_I3, current_I2, current_I1
        end

        println("length $((length(current_I3), length(current_I2), length(current_I1))) Expanding state with error: $current_error, heuristic: $current_heuristic")

        for i in setdiff(1:784, current_I3)
            I3_new = deepcopy(current_I3)
            I3_new = push(I3_new, i)
            new_heuristic = heuristic(sm.nn, sm.input, (I3_new, current_I2, current_I1), num_samples)
            new_error = max_error(sm.nn, sm.input, (I3_new, current_I2, current_I1), num_samples)
            push!(stack, (new_heuristic, new_error, (I3_new, current_I2, current_I1)))
        end

        for i in setdiff(1:256, current_I2)
            I2_new = deepcopy(current_I2)
            I2_new = push(I2_new, i)
            new_heuristic = heuristic(sm.nn, sm.input, (current_I3, I2_new, current_I1), num_samples)
            new_error = max_error(sm.nn, sm.input, (current_I3, I2_new, current_I1), num_samples)
            push!(stack, (new_heuristic, new_error, (current_I3, I2_new, current_I1)))
        end

        for i in setdiff(1:256, current_I1)
            I1_new = deepcopy(current_I1)
            I1_new = push(I1_new, i)
            new_heuristic = heuristic(sm.nn, sm.input, (current_I3, current_I2, I1_new), num_samples)
            new_error = max_error(sm.nn, sm.input, (current_I3, current_I2, I1_new), num_samples)
            push!(stack, (new_heuristic, new_error, (current_I3, current_I2, I1_new)))
        end
    end

    println("No valid subsets found within the given threshold.")
    return I3, I2, I1
end
