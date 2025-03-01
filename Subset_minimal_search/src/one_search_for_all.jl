 
function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

function forward_search_for_all(sm::Subset_minimal, (I3, I2, I1); threshold_total_err=0.1, num_samples=100)
    confidence = 1 - threshold_total_err
    
    initial_heuristic, full_error = heuristic(sm, (I3, I2, I1), confidence, num_samples)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, (I3, I2, I1))]
    array_of_the_best = []
    closed_list = Set{Tuple{typeof(I3), typeof(I2), typeof(I1)}}()

    steps = 0
    # max_steps = 100

    while !isempty(stack)

        # steps > max_steps && break
        steps += 1

        sort!(stack, by = x -> -x[1])
        current_heuristic, current_error, (I3, I2, I1) = pop!(stack)
        closed_list = push!(closed_list, (I3, I2, I1))
        
        if current_error <= 0
            push!(array_of_the_best, (I3, I2, I1))
            println("Valid subset found: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) with error: ", current_error)
            return (I3, I2, I1)
        end

        println("step: $steps, length $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) Expanding state with error: $current_error, heuristic: $current_heuristic")

        stack = expand_frwd(sm, stack, closed_list, (I3, I2, I1), confidence, num_samples)
    end

    println("Stack is empty")
    return array_of_the_best
end

# Priority on the length of subsets
function backward_reduction_for_all(sm::Subset_minimal, (I3, I2, I1); threshold=0.9, num_samples=100)    
    confidence = 1 - threshold
    initial_total_err = max_error(sm, (I3, I2, I1), confidence, num_samples)
    println("Initial max error: ", initial_total_err)

    stack = [(initial_total_err, (I3, I2, I1))]
    closed_list = Set{Tuple{typeof(I3), typeof(I2), typeof(I1)}}()

    best_subsets = (I3, I2, I1)
    best_total_len = (I3 !== nothing ? length(I3) : 0) +
                     (I2 !== nothing ? length(I2) : 0) +
                     (I1 !== nothing ? length(I1) : 0)

    max_steps = 300
    steps = 0

    while !isempty(stack)
        steps += 1
        steps > max_steps && break

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
        
        stack = expand_bcwd(sm, stack, closed_list, current_subsets, initial_total_err, confidence, num_samples)

    end
    I3, I2, I1 = best_subsets
    println("Final length: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1)))")
    
    return best_subsets
end
