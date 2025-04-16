"""

    The design is that stack will contain only valid solutions.
    In every step, I will pop the best solution and filter all its childs 
    to find those, which are valid. If there is no valid child, than the
    sample is subset_minimal and I add it to the solutions.
    If there is at least one valid child, I add it to the stack.

    I think that the heuristic function should be length, as I want to be expanding 
    solutions with minimal length.
"""
function backward_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true) where {TT}
    if !isvalid(ii) 
        println("initial verification of the solution failed, returning the solution")
        return(ii)
    end
    println("II length: ", solution_length(ii))
    initial_heuristic, full_error = heuristic_fun(ii)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, ii)]
    closed_list = Set{TT}()

    solutions = Set{TT}()
    steps = 0
    start_time = time()

    println("Starting backward search...")
    @timeit to "backward(length) search" while !isempty(stack)
        steps += 1

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solution")
            return best_subset
        end

        sort!(stack, by = ii -> length(ii[end]), rev = true) # we sort of want to expand minimum length first
        current_heuristic, current_error, ii = pop!(stack)
        closed_list = push!(closed_list, ii)

        new_subsets = expand_bcwd(sm, typeof(stack)(), closed_list, ii, heuristic_fun)
        new_subsets = filter(s -> isvalid(s[3]), new_subsets)
        if isempty(new_subsets)
            terminate_on_first_solution && return(ii)
            push!(solutions, ii)
        else
            println("step: $steps, length $(solution_length(ii)) Expanding state with error: $current_error, heuristic: $current_heuristic")
            append!(stack, new_subsets)
        end
    end
    println("Stack is empty")
    return new_subsets
end

new_subsets_bcwd(ii::SBitSet, idim) = [pop(ii, i) for i in 1:idim if i in ii]

function new_subsets_bcwd((I3, I2, I1)::T, idims::Tuple) where {T<:Tuple}
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
    for new_subset in new_subsets_bcwd(ii, sm.dims)
        if new_subset ∉ closed_list
            new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
            push!(stack, (new_heuristic, new_error, new_subset))    
        end
    end
    stack
end
