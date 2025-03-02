"""
Backward search with priority queue based on the criterium value for minimal subset search for the FiRST layer of a neural network.
"""

function backward_search(sm::Subset_minimal, calc_func::Function; max_steps::Int=1000, threshold::Float64=0.9, num_samples::Int=1000)
    open_list = PriorityQueue{SBitSet{32, UInt32}, Float64}()
    close_list = Set{SBitSet{32, UInt32}}()
    min_solution = nothing

    full_subset = SBitSet{32, UInt32}(collect(1:length(sm.input)))
    initial_score = calc_func(sm.nn, sm.input, full_subset, num_samples)
    enqueue!(open_list, full_subset, -initial_score)

    steps = 0
    while !isempty(open_list)
        steps += 1

        if steps > max_steps && !isempty(min_solution)
            break
        end

        current_subset, priority = peek(open_list)
        score = -priority
        current_subset = dequeue!(open_list)

        if score ≥ threshold
            # println("Solution found: score: ", score)
            if min_solution === nothing || length(current_subset) < length(min_solution)
                min_solution = current_subset
                println("length of min solution: ", length(min_solution), " score: ", score)
            end
            expand_backward(sm, calc_func, open_list, close_list, current_subset, num_samples)
        end

        push!(close_list, current_subset)

        println("Length of open_list: ", length(open_list))

    end
    return min_solution
end


function expand_backward(sm::Subset_minimal, calc_func::Function, open_list::PriorityQueue{SBitSet{N, T}, Float64}, close_list::Set{SBitSet{N, T}}, subset::SBitSet{N, T}, num_samples) where {N, T}
    for feature in collect(subset)
        new_subset = SBitSet{32, UInt32}(setdiff(subset, SBitSet{32, UInt32}(feature)))
        if new_subset ∈ close_list
            continue
        end

        score = calc_func(sm.nn, sm.input, new_subset, num_samples)
        if !haskey(open_list, new_subset)
            enqueue!(open_list, new_subset, -score)
        end

        while length(open_list) > 5000  # max size of priority queue (open_list)
            dequeue!(open_list)
        end
    end
end
