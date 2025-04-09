function beam_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun;  beam_size=5, time_limit=Inf, terminate_on_first_solution=true) where {TT}
    # stack with heuristic, error, ii
    initial_heuristic, full_error = heuristic_fun(ii)
    beam_sets = [initial_heuristic, full_error, ii]
    closed_list = Set{TT}()
    beam_set = expand_beam(sm, beam_set, closed_list, ii, heuristic_fun, beam_size)

end

function expand_beam(sm::Subset_minimal, beam_sets, closed_list, ii, heuristic, beam_size)
    new_beam_set = [1.0, 1.0, ii]
    for new_subset in new_subsets(ii, sm.dims)
        if new_subset ∉ closed_list
            new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
            push!(new_beam_set, (new_heuristic, new_error, new_subset))
            if length(new_beam_set) > beam_size
                sort!(new_beam_set, by=x->x[1]) # sort by heuristic
                pop!(new_beam_set)
            end
        end
    end
    new_beam_set
end  


function beam_search(sm::Subset_minimal, (I3, I2, I1), calc_func::Function, calc_func_partial::Function; data_model=nothing, error_threshold=0.1, beam_size=5, num_samples=100, time_limit=Inf)
    confidence = 1 - error_threshold

     # Initialize the first best beam
    first_best_beam = search_for_best_expansion(sm, (I3, I2, I1), calc_func, calc_func_partial, data_model, confidence, nothing, beam_size, num_samples)
    println("FIRST BEST SETS: ")
    print_beam(first_best_beam)

    # Create expanded best sets
    best_sets = expand_beam_subsets(sm, calc_func, calc_func_partial, data_model, confidence, first_best_beam, beam_size, num_samples)
    println("THE END of 0, best score: ", best_sets[1][2])
    print_beam(best_sets)

    start_time = time() 

    iter_count = 1
    @timeit to "beam search" while best_sets[1][2] > 0
        if time() - start_time > time_limit
            println("TIMEOUT")
            return best_sets
        end

        best_sets = expand_beam_subsets(sm, calc_func, calc_func_partial, data_model, confidence, best_sets, beam_size, num_samples)
        
        println("THE END of $iter_count, best score: ", best_sets[1][2])    
        iter_count += 1
    end
    print_beam(best_sets)

    return best_sets
end


function expand_beam_subsets(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, data_model, confidence, best_sets, beam_size::Int, num_samples::Int)
    expanded_subsets = search_for_best_expansion(sm, best_sets[end][1], calc_func, calc_func_partial, data_model, confidence, nothing, beam_size, num_samples)
    pop!(best_sets)

    for bs in best_sets
        expanded_subsets = search_for_best_expansion(sm, bs[1], calc_func, calc_func_partial, data_model, confidence, expanded_subsets, beam_size, num_samples)
    end

    return expanded_subsets
end

#todo Add all to new_subsets, evaluate all, choose beam_size best ones

function new_subsets((I3, I2, I1)::T, idims::Tuple, heuristic_fun) where {T<:Tuple}
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

function push_and_pop(heuristic_fun, best_results, new_subsets, beam_size, worst_error)
    new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)

    if new_error < worst_error
        push!(best_results, (new_subsets, new_error))
        if length(best_results) > beam_size
            sort!(best_results, by=x->x[2]) # sort by error
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
