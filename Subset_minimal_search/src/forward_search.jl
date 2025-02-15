
"""
Forward search with priority queue based on sdp criterion value
"""

function expand!(open_list::PriorityQueue{SBitSet{N, T}, Float64}, close_list::Set{SBitSet{N, T}}, subset::SBitSet{N, T}, model, xₛ, yₛ, num_samples) where {N, T}

    remaining_features = setdiff(1:size(xₛ, 1), subset)
    # println("Remaining features: ", length(remaining_features))

    for feature in remaining_features
        # new_subset = vcat(subset, feature)
        new_subset = union(subset, SBitSet{32, UInt32}(feature))

        if new_subset ∈ close_list
            continue
        end
        sdp_value = compute_sdp_fwd(model, xₛ, collect(new_subset), num_samples)
        # println("sdp_value: ", sdp_value, " for subset: ", new_subset)


        if !haskey(open_list, new_subset) #  checks whether the new_subset is already present in the open_list
            enqueue!(open_list, new_subset, -sdp_value)
        end
    end
end

function forward_search(model, xₛ, yₛ; max_steps::Int=10000, sdp_threshold::Float64=0.90, num_samples=100)
    open_list = PriorityQueue{SBitSet{32, UInt32}, Float64}()
    close_list = Set{SBitSet{32, UInt32}}()
    solutions = Set{SBitSet{32, UInt32}}()  
    expand!(open_list, close_list, SBitSet{32, UInt32}(), model, xₛ, yₛ, num_samples)

    steps = 0
    while !isempty(open_list)
        steps += 1
        if steps > max_steps && !isempty(solutions)
            break
        end

        current_subset, priority = peek(open_list)
        sdp_value = -priority
        current_subset = dequeue!(open_list)
        
        # println("current_subset:", current_subset)
        # println("sdp_value:", sdp_value)

        if sdp_value ≥ sdp_threshold
            println("Solution found: ", current_subset, " sdp_value: ", sdp_value)
            push!(solutions, current_subset)
        else
            println("Expanding current_subset: ", current_subset)
            expand!(open_list, close_list, current_subset, model, xₛ, yₛ, num_samples)
        end

        push!(close_list, current_subset)
    end

    reason = steps > max_steps ? :iter_limit : :exhausted
    return solutions, reason
end

function choose_best_solution(solutions, model, img, num_samples)
    best_solution = []
    best_sdp_value = -Inf
    # println("solutions: ", solutions[1])
    for solution in solutions
        # println("solution: ", solution)
        sdp_value = compute_sdp_fwd(model, img, solution, num_samples)
        if sdp_value > best_sdp_value
            best_sdp_value = sdp_value
            best_solution = solution
        end
    end
    return best_solution, best_sdp_value
end