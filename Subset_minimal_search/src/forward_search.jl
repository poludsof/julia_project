"""
Forward search with priority queue based on criterion value for minimal subset search for the FiRST layer of a neural network.
"""
function one_subset_forward_search(sm::Subset_minimal, calc_func::Function; data_model=nothing, max_steps::Int=1000, threshold::Float64=0.90, num_samples=1000, time_limit=Inf, terminate_on_first_solution=true)
    open_list = PriorityQueue{SBitSet{32, UInt32}, Float64}()
    close_list = Set{SBitSet{32, UInt32}}()
    solutions = Set{SBitSet{32, UInt32}}()

    min_solution = nothing

    expand!(sm, calc_func, data_model, open_list, close_list, SBitSet{32, UInt32}(), num_samples)

    start_time = time()

    steps = 0
    @timeit to "one-subset fwd search" while !isempty(open_list)
        steps += 1
        if steps > max_steps && !isempty(min_solution)
            println("Max steps reached ", steps)
            break
        end

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return solutions
        end

        current_subset, priority = peek(open_list)
        score = -priority
        dequeue!(open_list)

        if score ≥ threshold
            # println("Solution found: ", current_subset, " score: ", score)
            if min_solution === nothing || length(current_subset) < length(min_solution)
                min_solution = current_subset
                println("length of min solution: ", length(min_solution), " score: ", score)
                terminate_on_first_solution && return min_solution
                push!(solutions, min_solution)
            end
        else
            # println("Expanding current_subset: ", current_subset, " score: ", score)
            expand!(sm, calc_func, data_model, open_list, close_list, current_subset, num_samples)
        end

        # println("Current subset: ", current_subset, " score: ", score)
    end

    return solutions
end


""" Expand the current subset by adding one(best) feature at a time """
function expand!(sm, calc_func::Function, data_model, open_list::PriorityQueue{SBitSet{N, T}, Float64}, close_list::Set{SBitSet{N, T}}, subset::SBitSet{N, T}, num_samples) where {N, T}
    remaining_features = setdiff(1:size(sm.input, 1), subset)
    for feature in remaining_features
        new_subset = union(subset, SBitSet{32, UInt32}(feature))

        if new_subset ∈ close_list
            continue
        end
        score = calc_func(sm.nn, sm.input, new_subset, data_model, num_samples)

        if !haskey(open_list, new_subset)
            enqueue!(open_list, new_subset, -score)
        end

    end
end


function forward_search(sm::Subset_minimal, (I3, I2, I1), isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true)
    initial_heuristic, full_error = heuristic_fun((I3, I2, I1))
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, (I3, I2, I1))]
    # array_of_the_best = []
    closed_list = Set{Tuple{typeof(I3), typeof(I2), typeof(I1)}}()
    solutions = Set{Tuple{typeof(I3), typeof(I2), typeof(I1)}}()

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
        current_heuristic, current_error, (I3, I2, I1) = pop!(stack)
        closed_list = push!(closed_list, (I3, I2, I1))
        
        v = @timeit to "isvalid" isvalid((I3, I2, I1))

        if v
            println("Valid subset found: $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) with error: ", current_error)
            terminate_on_first_solution && return (I3, I2, I1)
            push!(solutions, (I3, I2, I1))
        end

        println("step: $steps, length $((I3 === nothing ? 0 : length(I3), I2 === nothing ? 0 : length(I2), I1 === nothing ? 0 : length(I1))) Expanding state with error: $current_error, heuristic: $current_heuristic")

        # steps > 2 && return(solutions)
        stack = @timeit to "expand_frwd" expand_frwd(sm, stack, closed_list, (I3, I2, I1), heuristic_fun)
    end

    println("Stack is empty")
    return solutions
end



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