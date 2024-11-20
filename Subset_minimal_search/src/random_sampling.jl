
function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet{N,T}) where {N, T}
    return map(idx -> idx in fix_inputs ? sm.input[idx] : rand(0:1), 1:length(sm.input))
end


function random_sampling(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_sets::Int) where {N, T}
    unique_sets = Set{Vector{Int}}()
    
    while length(unique_sets) < num_sets
        new_set = generate_random_img_with_fix_inputs(sm, fix_inputs)
        if !(new_set in unique_sets)
            push!(unique_sets, new_set)
        end
    end
    
    # println("Number of unique sets: ", length(unique_sets))
    return collect(unique_sets)
end



function calculate_ep(sm::Subset_minimal, fix_inputs::SBitSet{N, T}, num_samples::Int) where {N, T}
    num_sets = 2 ^ (length(sm.input) - length(fix_inputs))
    if num_sets < num_samples && num_sets > 0
        num_sets = num_sets
    else
        num_sets = num_samples
    end
    
    sampling_sets = random_sampling(sm, fix_inputs, num_sets)
    total_probability = 0.0

    for s in sampling_sets
        total_probability += softmax(sm.nn(s))[sm.output + 1]
    end

    return total_probability / num_sets
end


function calculate_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_samples::Int) where {N, T}
    # println("Image: ", sm.output)
    num_sets = 2 ^ (length(sm.input) - length(fix_inputs))
    if num_sets < num_samples && num_sets > 0
        num_sets = num_sets
    else
        num_sets = num_samples
    end
    # println("Number of sets: ", num_sets)

    sampling_sets = random_sampling(sm, fix_inputs, num_sets)
    # println("Number of sampling sets: ", length(sampling_sets))
    correct_classified = 0
    for s in sampling_sets
        if argmax(sm.nn(s)) - 1 == sm.output
            correct_classified += 1
        end
    end
    
    # println("Correct classified: ", correct_classified)
    return correct_classified / num_sets
end


function search_sets(sm::Subset_minimal, calc_func::Function, fix_inputs::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_ep = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:length(sm.input)
        if !(i in fix_inputs)
            new_set = SBitSet{N, T}()
            new_set = union(fix_inputs, SBitSet{32, UInt32}(i))
            ep = calc_func(sm, new_set, num_samples)
            if ep >= worst_from_best_ep
                push!(best_results, (new_set, ep))
                if length(best_results) > num_best
                    sort!(best_results, by=x->x[2], rev=true)
                    pop!(best_results)
                    worst_from_best_ep = best_results[end][2]
                end
            end
        end
    end

    return best_results
end
 

function generate_array_of_top_sets(sm::Subset_minimal, calc_func::Function, best_results::Array{Tuple{SBitSet{N,T}, Float32}}, num_best::Int, num_samples::Int) where{N, T}
    first_of_the_first = search_sets(sm, calc_func, best_results[end][1], num_best, num_samples)
    pop!(best_results)

    for bs in best_results
        first_of_the_first = search_sets(sm, calc_func, bs[1], num_best, num_samples, first_of_the_first)
    end

    return first_of_the_first
end


# scoring_func:
# 1)calculate_ep
# 2)calculate_sdp
function get_minimal_set_generic(sm::Subset_minimal, calc_func::Function, threshold=0.9, num_best=5, num_samples=100)
    fix_inputs = SBitSet{32, UInt32}()
    the_most_first = search_sets(sm, calc_func, fix_inputs, num_best, num_samples)
    println("FIRST BEST SETS: ")
    print_sets(the_most_first)

    tmp = generate_array_of_top_sets(sm, calc_func, the_most_first, num_best, num_samples)
    println("THE END of 0, best score: ", tmp[1][2])

    score_val = 0.0
    i = 1
    while score_val < threshold
        tmp = generate_array_of_top_sets(sm, calc_func, tmp, num_best, num_samples)
        println("THE END of $i, best score: ", tmp[1][2])    
        i += 1
        score_val = tmp[1][2]
    end
    print_sets(tmp)

    return tmp[1][1]  # return set with the best score
end




# Helper functions
function print_sets(sets::Array{Tuple{SBitSet{N,T}, Float32}}) where {N, T}
    println("Number of sets: ", length(sets))
    for s in sets
        println("Set: ", s[1], " metric: ", s[2])
    end
end