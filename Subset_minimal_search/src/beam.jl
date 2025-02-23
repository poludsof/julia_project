
function make_search_best_subsets(sm::Subset_minimal)
    function search_best_subsets(calc_func::Function, fix_inputs::SBitSet{N, T}, beam_size::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
        # Initialize the worst threshold based on the last element in best_results (or 0 if empty)
        worst_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

        for i in 1:length(sm.input)
            if i in fix_inputs
                continue # Skip fixed inputs
            end

            # Create a new subset by adding the current feature
            new_set = union(fix_inputs, SBitSet{32, UInt32}(i))
            score = calc_func(new_set, num_samples) # Compute the score for the subset (ep/sdp)

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

    return search_best_subsets
end

function make_expand_the_best_subsets(sm::Subset_minimal)
    search_best_subsets = make_search_best_subsets(sm)

    function expand_the_best_subsets(calc_func::Function, best_sets::Array{Tuple{SBitSet{N,T}, Float32}}, beam_size::Int, num_samples::Int) where{N, T}
        extended_subsets = search_best_subsets(calc_func, best_sets[end][1], beam_size, num_samples)
        pop!(best_sets)

        for bs in best_sets
            extended_subsets = search_best_subsets(calc_func, bs[1], beam_size, num_samples, extended_subsets)
        end

        return extended_subsets
    end
    return expand_the_best_subsets
end

# scoring_func:
# 1)calculate_ep
# 2)calculate_sdp
function make_beam_search(sm::Subset_minimal)
    expand_the_best_subsets = make_expand_the_best_subsets(sm)
    search_best_subsets = make_search_best_subsets(sm)

    function beam_search!(calc_func::Function; threshold=0.9, beam_size=5, num_samples=100)
        first_best_beam = search_best_subsets(calc_func, SBitSet{32, UInt32}(), beam_size, num_samples) # Initialize the first best beam
        println("FIRST BEST SETS: ")
        print_sets(first_best_beam)

        # Create extended best sets
        best_sets = expand_the_best_subsets(calc_func, first_best_beam, beam_size, num_samples)
        println("THE END of 0, best score: ", best_sets[1][2])
        # print_sets(best_sets)

        iter_count = 1
        while best_sets[1][2] < threshold  # Continue until the criterium value meets the threshold
            best_sets = expand_the_best_subsets(calc_func, best_sets, beam_size, num_samples)
            println("THE END of $iter_count, best score: ", best_sets[1][2])    
            iter_count += 1    
        end
        print_sets(best_sets)

        return best_sets
    end

    return beam_search!
end




# Helper functions
function print_sets(sets::Array{Tuple{SBitSet{N,T}, Float32}}) where {N, T}
    println("Number of sets: ", length(sets))
    for s in sets
        println("Set: ", s[1], " metric: ", s[2])
    end
end