
function one_subset_backward_search(sm::Subset_minimal, calc_func::Function; max_steps::Int=1000, threshold::Float64=0.9, num_samples::Int=1000, time_limit=60)
    open_list = PriorityQueue{SBitSet{32, UInt32}, Float64}()
    close_list = Set{SBitSet{32, UInt32}}()
    min_solution = nothing

    full_subset = SBitSet{32, UInt32}(collect(1:length(sm.input)))
    initial_score = calc_func(sm.nn, sm.input, full_subset, num_samples)
    enqueue!(open_list, full_subset, -initial_score)

    start_time = time()

    steps = 0
    @timeit to "one-subset bcwd search" while !isempty(open_list)
        steps += 1

        if steps > max_steps && !isempty(min_solution)
            break
        end

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solution")
            return min_solution
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

        # println("Length of open_list: ", length(open_list))

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

        # while length(open_list) > 5000  # max size of priority queue (open_list)
        #     dequeue!(open_list)
        # end
    end
end


function backward_search_length_priority(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function; threshold=0.1, max_steps=1000, num_samples=100, time_limit=60)    
    confidence = 1 - threshold
    initial_total_err = max_error(sm, calc_func, calc_func_partial, (I3, I2, I1), confidence, num_samples)
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
        
        stack = expand_bcwd(sm, calc_func, calc_func_partial, stack, closed_list, current_subsets, initial_total_err, confidence, num_samples)

    end
    I3, I2, I1 = best_subsets
    println("Final length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1)))")
    
    return best_subsets
end

function backward_search_error_priority(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function; threshold=0.1, max_steps=1000, num_samples=100, time_limit=60)    
    confidence = 1 - threshold
    initial_total_err = max_error(sm, calc_func, calc_func_partial, (I3, I2, I1), confidence, num_samples)
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
        
        stack = expand_bcwd(sm, calc_func, calc_func_partial, stack, closed_list, current_subsets, initial_total_err, confidence, num_samples)

    end
    I3, I2, I1 = best_subsets
    println("Final length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1)))")
    
    return best_subsets
end
