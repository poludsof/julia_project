"""
Forward search with priority queue based on criterion value for minimal subset search for the FiRST layer of a neural network.
"""
function forward_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true) where {TT}
    initial_heuristic, full_error = heuristic_fun(ii)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, ii)]
    closed_list = Set{TT}()
    solutions = Set{TT}()

    steps = 0
    # max_steps = 100
    start_time = time()

    @timeit to "forward search" while !isempty(stack)

        # steps > max_steps && break
        steps += 1
        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return solutions
        end

        sort!(stack, by = x -> -x[1])
        current_heuristic, current_error, ii = pop!(stack)
        closed_list = push!(closed_list, ii)
        
        v = @timeit to "isvalid" isvalid(ii)

        if v
            println("Valid subset found: $(solution_length(ii)) with error: ", current_error)
            terminate_on_first_solution && return(ii)
            push!(solutions, ii)
        end

        println("step: $steps, length $(solution_length(ii)) Expanding state with error: $current_error, heuristic: $current_heuristic")

        # steps > 2 && return(solutions)
        stack = @timeit to "expand_frwd" expand_frwd(sm, stack, closed_list, ii, heuristic_fun)
    end

    println("Stack is empty")
    return solutions
end

solution_length(ii::Tuple) = solution_length.(ii)
solution_length(ii::SBitSet) = length(ii)
solution_length(::Nothing) = 0

function expand_frwd(sm::Subset_minimal, stack, closed_list, (I3, I2, I1), heuristic_fun)
    if I3 !== nothing
        for i in setdiff(1:784, I3)
            new_subset = (push(I3, i), I2, I1)
            if new_subset ∉ closed_list
                new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
                push!(stack, (new_heuristic, new_error, new_subset))    
            end
        end
    end
    if I2 !== nothing
        for i in setdiff(1:256, I2)
            new_subset = (I3, push(I2, i), I1)
            if new_subset ∉ closed_list
                new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
                push!(stack, (new_heuristic, new_error, new_subset))
            end
        end
    end
    if I1 !== nothing
        for i in setdiff(1:256, I1)
            new_subset = (I3, I2, push(I1, i))
            if new_subsets ∉ closed_list
                new_heuristic, new_error = @timeit to "heuristic" heuristic_fun(new_subset)
                push!(stack, (new_heuristic, new_error, new_subsets))
            end
        end
    end
    stack
end

function expand_frwd(sm::Subset_minimal, stack, closed_list, ii::SBitSet, heuristic_fun)
    for i in setdiff(1:length(sm.input), ii)
        iiᵢ = push(ii, i)
        if iiᵢ ∉ closed_list
            new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(iiᵢ)
            push!(stack, (new_heuristic, new_error, iiᵢ))    
        end
    end
    stack
end