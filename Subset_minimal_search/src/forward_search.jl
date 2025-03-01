"""
Forward search with priority queue based on criterion value for minimal subset search for the FiRST layer of a neural network.
"""

function forward_search(sm::Subset_minimal, calc_func::Function; max_steps::Int=1000, threshold::Float64=0.90, num_samples=1000)
    open_list = PriorityQueue{SBitSet{32, UInt32}, Float64}()
    close_list = Set{SBitSet{32, UInt32}}()
    min_solution = nothing

    expand!(sm, calc_func, open_list, close_list, SBitSet{32, UInt32}(), num_samples)

    steps = 0
    while !isempty(open_list)
        steps += 1
        if steps > max_steps && !isempty(min_solution)
            println("Max steps reached ", steps)
            break
        end

        current_subset, priority = peek(open_list)
        score = -priority
        current_subset = dequeue!(open_list)

        if score ≥ threshold
            # println("Solution found: ", current_subset, " score: ", score)
            if min_solution === nothing || length(current_subset) < length(min_solution)
                min_solution = current_subset
                println("length of min solution: ", length(min_solution), " score: ", score)
            end
        else
            # println("Expanding current_subset: ", current_subset, " score: ", score)
            expand!(sm, calc_func, open_list, close_list, current_subset, num_samples)
        end

        push!(close_list, current_subset)
    end

    return min_solution
end


""" Expand the current subset by adding one(best) feature at a time """
function expand!(sm, calc_func::Function, open_list::PriorityQueue{SBitSet{N, T}, Float64}, close_list::Set{SBitSet{N, T}}, subset::SBitSet{N, T}, num_samples) where {N, T}
    remaining_features = setdiff(1:size(sm.input, 1), subset)
    for feature in remaining_features
        new_subset = union(subset, SBitSet{32, UInt32}(feature))

        if new_subset ∈ close_list
            continue
        end
        score = calc_func(sm, new_subset, num_samples)

        if !haskey(open_list, new_subset)
            enqueue!(open_list, new_subset, -score)
        end

    end
end