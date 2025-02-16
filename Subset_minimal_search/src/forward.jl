"""
Forward search with priority queue based on sdp criterion value for minimal subset search for the FiRST layer of a neural network.
"""

function make_forward_search_sdp(sm::Subset_minimal)
    expand! = make_expand!(sm)
    choose_best_solution = make_choose_best_solution(sm)

    function forward_search_sdp(; max_steps::Int=1000, sdp_threshold::Float64=0.90, num_samples=1000)
        open_list = PriorityQueue{SBitSet{32, UInt32}, Float64}()
        close_list = Set{SBitSet{32, UInt32}}()
        solutions = Set{SBitSet{32, UInt32}}()

        expand!(open_list, close_list, SBitSet{32, UInt32}(), num_samples)

        steps = 0
        while !isempty(open_list)
            steps += 1
            if steps > max_steps && !isempty(solutions)
                break
            end

            current_subset, priority = peek(open_list)
            sdp_value = -priority
            current_subset = dequeue!(open_list)

            if sdp_value ≥ sdp_threshold
                println("Solution found: ", current_subset, " sdp_value: ", sdp_value)
                push!(solutions, current_subset)
            else
                println("Expanding current_subset: ", current_subset)
                expand!(open_list, close_list, current_subset, num_samples)
            end

            push!(close_list, current_subset)
        end

        # reason = steps > max_steps ? :iter_limit : :exhausted
        return choose_best_solution(solutions, num_samples)
    end
    return forward_search_sdp
end


function make_expand!(sm::Subset_minimal)
    function expand!(open_list::PriorityQueue{SBitSet{N, T}, Float64}, close_list::Set{SBitSet{N, T}}, subset::SBitSet{N, T}, num_samples) where {N, T}
        remaining_features = setdiff(1:size(sm.input, 1), subset)
        for feature in remaining_features
            new_subset = union(subset, SBitSet{32, UInt32}(feature))

            if new_subset ∈ close_list
                continue
            end
            sdp_value = compute_sdp_fwd(sm.nn, sm.input, collect(new_subset), num_samples)

            if !haskey(open_list, new_subset)
                enqueue!(open_list, new_subset, -sdp_value)
            end
        end
    end
    return expand!
end

function make_choose_best_solution(sm::Subset_minimal)
    function choose_best_solution(solutions, num_samples)
        best_solution = []
        best_sdp_value = -Inf
        for solution in solutions
            sdp_value = compute_sdp_fwd(sm.nn, sm.input, solution, num_samples)
            if sdp_value > best_sdp_value
                best_sdp_value = sdp_value
                best_solution = solution
            end
        end
        return best_solution, best_sdp_value
    end
    return choose_best_solution
end
