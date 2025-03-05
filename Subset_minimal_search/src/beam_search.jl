
function search_best_subsets(sm::Subset_minimal, calc_func::Function, fix_inputs::SBitSet{N, T}, beam_size::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    # Initialize the worst threshold based on the last element in best_results (or 0 if empty)
    worst_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:length(sm.input)
        if i in fix_inputs
            continue # Skip fixed inputs
        end

        # Create a new subset by adding the current feature
        new_set = union(fix_inputs, SBitSet{32, UInt32}(i))
        score = calc_func(sm.nn, sm.input, new_set, num_samples) # Compute the score of the subset (ep/sdp)

        if score >= worst_threshold
            push!(best_results, (new_set, score))
            if length(best_results) > beam_size
                sort!(best_results, by=x->x[2], rev=true)
                pop!(best_results)
                worst_threshold = best_results[end][2]
            end
        end
    end

    return best_results
end


function expand_the_best_subsets(sm::Subset_minimal, calc_func::Function, best_sets::Array{Tuple{SBitSet{N,T}, Float32}}, beam_size::Int, num_samples::Int) where{N, T}
    extended_subsets = search_best_subsets(sm, calc_func, best_sets[end][1], beam_size, num_samples)
    pop!(best_sets)

    for bs in best_sets
        extended_subsets = search_best_subsets(sm, calc_func, bs[1], beam_size, num_samples, extended_subsets)
    end

    return extended_subsets
end


function one_subset_beam_search(sm::Subset_minimal, calc_func::Function; threshold=0.9, beam_size=5, num_samples=100, time_limit=Inf)
    first_best_beam = search_best_subsets(sm, calc_func, SBitSet{32, UInt32}(), beam_size, num_samples) # Initialize the first best beam
    println("FIRST BEST SETS: ")
    print_beam(first_best_beam)

    # Create extended best sets
    best_sets = expand_the_best_subsets(sm, calc_func, first_best_beam, beam_size, num_samples)
    println("THE END of 0, best score: ", best_sets[1][2])

    start_time = time() 

    iter_count = 1
    @timeit to "one-subset beam search" while best_sets[1][2] < threshold  # Continue until the criterium value meets the threshold
        if time() - start_time > time_limit
            println("TIMEOUT")
            return best_sets
        end
        best_sets = expand_the_best_subsets(sm, calc_func, best_sets, beam_size, num_samples)
        println("THE END of $iter_count, best score: ", best_sets[1][2])    
        iter_count += 1    
    end
    print_sets(best_sets)

    return best_sets
end




""" Beam search for all subsets(layers) """
function beam_search(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function; error_threshold=0.1, beam_size=5, num_samples=100, time_limit=Inf)
    confidence = 1 - error_threshold

     # Initialize the first best beam
    first_best_beam = search_for_best_expansion(sm, (I3, I2, I1), calc_func, calc_func_partial, confidence, nothing, beam_size, num_samples)
    println("FIRST BEST SETS: ")
    print_beam(first_best_beam)

    # Create expanded best sets
    best_sets = expand_beam_subsets(sm, calc_func, calc_func_partial, confidence, first_best_beam, beam_size, num_samples)
    println("THE END of 0, best score: ", best_sets[1][2])
    print_beam(best_sets)

    start_time = time() 

    iter_count = 1
    @timeit to "beam search" while best_sets[1][2] > 0
        if time() - start_time > time_limit
            println("TIMEOUT")
            return best_sets
        end

        best_sets = expand_beam_subsets(sm, calc_func, calc_func_partial, confidence, best_sets, beam_size, num_samples)
        
        println("THE END of $iter_count, best score: ", best_sets[1][2])    
        iter_count += 1
    end
    print_beam(best_sets)

    return best_sets
end


function expand_beam_subsets(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, confidence, best_sets, beam_size::Int, num_samples::Int)
    expanded_subsets = search_for_best_expansion(sm, best_sets[end][1], calc_func, calc_func_partial, confidence, nothing, beam_size, num_samples)
    pop!(best_sets)

    for bs in best_sets
        expanded_subsets = search_for_best_expansion(sm, bs[1], calc_func, calc_func_partial, confidence, expanded_subsets, beam_size, num_samples)
    end

    return expanded_subsets
end


function search_for_best_expansion(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function, confidence, best_results, beam_size::Int, num_samples::Int)
    # best_results: is Array{Tuple{Tuple{typeof(I3), typeof(I2), typeof(I1)}, Float32}, 1}
    # Initialize the worst threshold based on the last element in best_results (or 0 if empty)
    worst_error = best_results === nothing ? 1.0 : best_results[end][2]
    # println("Worst error: ", worst_error)

    if best_results === nothing
        best_results = Array{Tuple{Tuple{typeof(I3), typeof(I2), typeof(I1)}, Float32}, 1}()
    end

    if I3 !== nothing
        for i in setdiff(1:784, I3)
            new_subsets = (push(I3, i), I2, I1)
            best_results, worst_error = push_and_pop(sm, calc_func, calc_func_partial, best_results, new_subsets, confidence, beam_size, num_samples, worst_error)
        end
    end
    if I2 !== nothing
        for i in setdiff(1:256, I2)
            new_subsets = (I3, push(I2, i), I1)
            best_results, worst_error = push_and_pop(sm, calc_func, calc_func_partial, best_results, new_subsets, confidence, beam_size, num_samples, worst_error)
        end
    end
    if I1 !== nothing
        for i in setdiff(1:256, I1)
            new_subsets = (I3, I2, push(I1, i))
            best_results, worst_error = push_and_pop(sm, calc_func, calc_func_partial, best_results, new_subsets, confidence, beam_size, num_samples, worst_error)
        end
    end
    return best_results
end


function push_and_pop(sm, calc_func, calc_func_partial, best_results, new_subsets, confidence, beam_size, num_samples, worst_error)
    new_heuristic, new_error = heuristic(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
    if isvalid(nothing, new_error, worst_error)
        push!(best_results, (new_subsets, new_error))
        if length(best_results) > beam_size
            sort!(best_results, by=x->x[2])
            pop!(best_results)
            worst_error = best_results[end][2]
        end
    end
    return best_results, worst_error
end


# Helper functions
function print_beam(sets)
    println("Size of beam: ", length(sets))
    for s in sets
        println("Set: ", s[1], " Score: ", s[2])
    end
end
