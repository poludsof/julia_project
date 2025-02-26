"""
Forward search with priority queue based on criterion value for minimal subset search for the FiRST layer of a neural network.
"""

function make_forward_search(sm::Subset_minimal, constraint, heuristic; max_steps::Int=1000, initial_solution::T=SBitSet{32, UInt32}()) where {T}
    expand! = make_expand!(sm, constraint)

    function forward_search(calc_func::Function)
        open_list = PriorityQueue{T, Float64}()
        close_list = Set{T}()
        min_solution = nothing

        expand!(calc_func, open_list, close_list, initial_solution, num_samples)

        steps = 0
        while !isempty(open_list)
            steps += 1
            if steps > max_steps && !isempty(min_solution)
                println("Max steps reached ", steps)
                break
            end

            current_subset, priority = peek(open_list)
            current_subset = dequeue!(open_list)
            isvalid = constraint(current_subset)

            if isvalid
                # println("Solution found: ", current_subset, " score: ", score)
                if min_solution === nothing || length(current_subset) < length(min_solution)
                    min_solution = current_subset
                    println("length of min solution: ", length(min_solution), " score: ", score)
                end
            else
                # println("Expanding current_subset: ", current_subset, " score: ", score)
                expand!(calc_func, open_list, close_list, current_subset, num_samples)
            end

            push!(close_list, current_subset)
        end

        return min_solution
    end
    return forward_search
end


""" Expand the current subset by adding one(best) feature at a time """
function make_expand!(sm::Subset_minimal, heuristic)
    function expand!(calc_func::Function, open_list::PriorityQueue{SBitSet{N, T}, Float64}, close_list::Set{SBitSet{N, T}}, subset::SBitSet{N, T}, num_samples) where {N, T}
        remaining_features = setdiff(1:size(sm.input, 1), subset)
        for feature in remaining_features
            new_subset = union(subset, SBitSet{32, UInt32}(feature))

            new_subset ∈ close_list && continue
            
            score = heuristic(new_subset)
            if !haskey(open_list, new_subset)
                enqueue!(open_list, new_subset, -score)
            end

        end
    end
    return expand!
end