
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

        sort!(stack, by = x -> -x[1])
        current_heuristic, current_error, ii = pop!(stack)
        closed_list = push!(closed_list, ii)

        v = @timeit to "isvalid" isvalid(ii)

        if v  # until the first invalid subset is found
            println("Valid subset: $(solution_length(ii)) with error: ", current_error)
        else
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

function backward_search_length_priority(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function; data_model=nothing, threshold=0.1, max_steps=1000, num_samples=100, time_limit=Inf)    
    confidence = 1 - threshold
    initial_total_err = max_error(sm, calc_func, calc_func_partial, (I3, I2, I1), confidence, data_model, num_samples)
    println("Initial max error: ", initial_total_err)

    stack = [(initial_total_err, (I3, I2, I1))]
    closed_list = Set{Tuple{typeof(I3), typeof(I2), typeof(I1)}}()

    best_subsets = (I3, I2, I1)
    best_total_len = (I3 !== nothing ? length(I3) : 0) +
                     (I2 !== nothing ? length(I2) : 0) +
                     (I1 !== nothing ? length(I1) : 0)

    steps = 0
    start_time = time()

    @timeit to "backward(length) search" while !isempty(stack)
        steps += 1
        steps > max_steps && break

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solution")
            return best_subsets
        end

        sort!(stack, by = x -> (-((x[2][1] !== nothing ? length(x[2][1]) : 0) +
                                  (x[2][2] !== nothing ? length(x[2][2]) : 0) +
                                  (x[2][3] !== nothing ? length(x[2][3]) : 0)), -x[1]))
        current_error, current_subsets = pop!(stack)
        I3, I2, I1 = current_subsets

        push!(closed_list, current_subsets)

        println("step: ", steps, ", length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) Current error: ", current_error)

        total_len = (I3 !== nothing ? length(I3) : 0) +
                    (I2 !== nothing ? length(I2) : 0) +
                    (I1 !== nothing ? length(I1) : 0)

        if total_len < best_total_len
            best_subsets = current_subsets
            best_total_len = total_len
        end
        
        stack = expand_bcwd(sm, calc_func, calc_func_partial, data_model, stack, closed_list, current_subsets, initial_total_err, confidence, num_samples)

    end
    I3, I2, I1 = best_subsets
    println("Final length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1)))")
    
    return best_subsets
end

function backward_search_error_priority(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function; data_model=nothing, threshold=0.1, max_steps=1000, num_samples=100, time_limit=Inf)    
    confidence = 1 - threshold
    initial_total_err = max_error(sm, calc_func, calc_func_partial, (I3, I2, I1), confidence, data_model, num_samples)
    println("Initial max error: ", initial_total_err)

    stack = [(initial_total_err, (I3, I2, I1))]
    closed_list = Set{Tuple{typeof(I3), typeof(I2), typeof(I1)}}()

    best_subsets = (I3, I2, I1)
    best_total_len = (I3 !== nothing ? length(I3) : 0) +
                     (I2 !== nothing ? length(I2) : 0) +
                     (I1 !== nothing ? length(I1) : 0)

    steps = 0
    start_time = time()

    @timeit to "backward(error) search" while !isempty(stack)
        steps += 1
        steps > max_steps && break

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solution")
            return best_subsets
        end

        sort!(stack, by = x -> (-x[1], -((x[2][1] !== nothing ? length(x[2][1]) : 0) +
                                         (x[2][2] !== nothing ? length(x[2][2]) : 0) +
                                         (x[2][3] !== nothing ? length(x[2][3]) : 0))))
        current_error, current_subsets = pop!(stack)
        I3, I2, I1 = current_subsets

        push!(closed_list, current_subsets)

        println("step: ", steps, ", length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) Current error: ", current_error)

        total_len = (I3 !== nothing ? length(I3) : 0) +
                    (I2 !== nothing ? length(I2) : 0) +
                    (I1 !== nothing ? length(I1) : 0)

        if total_len < best_total_len
            best_subsets = current_subsets
            best_total_len = total_len
        end
        
        stack = expand_bcwd(sm, calc_func, calc_func_partial, data_model, stack, closed_list, current_subsets, initial_total_err, confidence, num_samples)

    end
    I3, I2, I1 = best_subsets
    println("Final length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1)))")
    
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
