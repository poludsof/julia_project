"""
    forward_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true, exclude_supersets = true, only_smaller = true, refine_with_backward=true) where {TT}

    Forward search with priority queue based on criterion value for minimal subset search for the FiRST layer of a neural network.
    ii --- initial solution
    isvalid --- function of ii returning true if the solution is valid and false otherwise
    heuristic_fun --- heuristic function estimating quality of the solution, lower is better
    time_limit=Inf --- maximum time limit to find the solution(s)
    terminate_on_first_solution=true --- if true the search stops after finding the first solution
    exclude_supersets = true --- if true the search checks if the node currently expanded is not a superset of a known solution. If yes, it is not expanded.
    only_smaller = true --- expand only solutions smaller then existing best, which is useful when looking for smallest possible 
    refine_with_backward=true --- if true, the found solution is refined by backward search to check that it is a minimal subset solution
"""
function forward_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true, exclude_supersets = true, only_smaller = true, refine_with_backward=true) where {TT}
    initial_heuristic, full_error = heuristic_fun(ii)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, ii)]
    closed_list = Set{TT}()
    solutions = Set{TT}()

    steps = 0
    start_time = time()
    smallest_solution = typemax(Int)

    @timeit to "forward search" while !isempty(stack)

        # steps > max_steps && break
        steps += 1
        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return solutions
        end

        sort!(stack, by = first, rev = true)
        current_heuristic, current_error, ii = pop!(stack)
        closed_list = push!(closed_list, ii)

        # Look only for strictly smaller solutions
        if only_smaller
            length(ii) ≥ smallest_solution && continue
        end

        # check if we have not find a solution, which is subset of solution we have find
        if exclude_supersets
            issub = @timeit to "issubset" any(Base.Fix1(issubset,ii), solutions)
            issub && println("skipping superset")
            issub && continue
        end
        
        v = @timeit to "isvalid" isvalid(ii)

        if v
            if refine_with_backward
                println("Valid subset found. Pruning with backward search...")
                ii = @timeit to "backward_search"  backward_search(sm, ii, isvalid, heuristic_fun, time_limit=200, terminate_on_first_solution=true)
                println("After pruning: ", ii)
            end

            println("Valid subset found: $(solution_length(ii)) with error: ", current_error)
            terminate_on_first_solution && return(ii)
            if solution_length(ii) < smallest_solution
                smallest_solution = solution_length(ii)
                println("new smallest solution so far: ", ii)
            end
            push!(solutions, ii)
            continue
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

new_subsets_fwrd(ii::SBitSet, idim) = [push(ii, i) for i in setdiff(1:idim, ii)]

function new_subsets_fwrd((I3, I2, I1)::T, idims::Tuple) where {T<:Tuple}
    new_subsets = T[]
    if I3 !== nothing
        append!(new_subsets, [(push(I3, i), I2, I1) for i in setdiff(1:idims[1], I3)])
    end
    if I2 !== nothing
        append!(new_subsets, [(I3, push(I2, i), I1) for i in setdiff(1:idims[2], I2)])
    end
    if I1 !== nothing
        append!(new_subsets, [(I3, I2, push(I1, i)) for i in setdiff(1:idims[3], I1)])
    end
    new_subsets
end

function expand_frwd(sm::Subset_minimal, stack, closed_list, ii, heuristic_fun)
    for new_subset in new_subsets_fwrd(ii, sm.dims)
        if new_subset ∉ closed_list
            new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
            push!(stack, (new_heuristic, new_error, new_subset))    
        end
    end
    stack
end
