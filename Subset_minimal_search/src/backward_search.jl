
function backward_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true) where {TT}
    println("II length: ", solution_length(ii))
    initial_heuristic, full_error = heuristic_fun(ii)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, ii)]
    closed_list = Set{TT}()

    best_subsets = ii
    best_total_len = solution_length(ii)

    steps = 0
    start_time = time()

    @timeit to "backward(length) search" while !isempty(stack)

        steps += 1

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solution")
            return best_subsets
        end

        sort!(stack, by = x -> -x[1]) # error_priority
        current_heuristic, current_error, ii = pop!(stack)
        closed_list = push!(closed_list, ii)

        v = @timeit to "isvalid" isvalid(ii)

        if v 
            println("Valid subset: $(solution_length(ii)) with error: ", current_error)
        else  # until the first invalid subset is found
            println("terminate_on_first_solution")
            terminate_on_first_solution && return(best_subsets)
        end

        println("step: $steps, length $(solution_length(ii)) Expanding state with error: $current_error, heuristic: $current_heuristic")

        total_len = solution_length(ii)
        if total_len < best_total_len
            best_subsets = ii
            best_total_len = total_len
        end

        stack = @timeit to "expand_bcwd" expand_bcwd(sm, stack, closed_list, ii, heuristic_fun)
    end
    println("Stack is empty")
    return best_subsets
end

solution_length(ii::Tuple) = solution_length.(ii)
solution_length(ii::SBitSet) = length(ii)
solution_length(::Nothing) = 0

new_subsets(ii::SBitSet, idim) = [pop(ii, i) for i in 1:idim if i in ii]

function new_subsets((I3, I2, I1)::T, idims::Tuple) where {T<:Tuple}
    new_subsets = T[]
    if I3 !== nothing
        append!(new_subsets, [(pop(I3, i), I2, I1) for i in 1:idims[1] if i in I3])
    end
    if I2 !== nothing
        append!(new_subsets, [(I3, pop(I2, i), I1) for i in 1:idims[2] if i in I2])
    end
    if I1 !== nothing
        append!(new_subsets, [(I3, I2, pop(I1, i)) for i in 1:idims[3] if i in I1])
    end
    new_subsets
end

function expand_bcwd(sm::Subset_minimal, stack, closed_list, ii, heuristic_fun)
    for new_subset in new_subsets(ii, sm.dims)
        if new_subset ∉ closed_list
            new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
            push!(stack, (new_heuristic, new_error, new_subset))    
        end
    end
    stack
end
