 
function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

function forward_search_for_all(sm::Subset_minimal, subsets; threshold_total_err=0.1, num_samples=100)
    I3, I2, I1 = subsets
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
            println("Valid subset found: $((length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) with error: ", current_error)
            return (I3, I2, I1)
        end

        println("step: $steps, length $((length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) Expanding state with error: $current_error, heuristic: $current_heuristic")

        stack = expand_frwd(sm, stack, closed_list, (I3, I2, I1), confidence, num_samples)
    end

    println("Stack is empty")
    return array_of_the_best
end

