

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


function calculate_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}) where {N, T}
    num_sets = 2 ^ (length(sm.input) - length(fix_inputs))
    if num_sets < 1000 && num_sets > 0
        num_sets = num_sets
    else
        num_sets = 1000
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


# create an array of tuples(SBitSet, Float32) to store the best results of the sdp
function first_bunch_of_best_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_best::Int64) where {N, T}
    best_results = Array{Tuple{SBitSet{N,T}, Float32}, 1}()
    worst_from_best_sdp = 0.0

    for i in 1:length(sm.input)
        if !(i in fix_inputs)
            new_set = SBitSet{N,T}()
            new_set = union(fix_inputs, SBitSet{32, UInt32}(i))
            sdp = calculate_sdp(sm, new_set)
            if sdp >= worst_from_best_sdp
                push!(best_results, (new_set, sdp))
                if length(best_results) > num_best
                    sort!(best_results, by=x->x[2], rev=true)
                    pop!(best_results)
                    worst_from_best_sdp = best_results[end][2]
                end
            end
        end
    end
    # best_sdp_val = best_results[1][2]
    # worst_from_best_sdp = best_results[end][2]
    # println("best_sdp_val: ", best_sdp_val, " worst_from_best_sdp: ", worst_from_best_sdp)
    return best_results
end


function best_of_the_fisrt_best(sm::Subset_minimal, best_results::Array{Tuple{SBitSet{N,T}, Float32}}, fix_inputs::SBitSet{N,T}, num_best::Int) where {N, T}
    worst_from_best_sdp = best_results[end][2]
    for i in 1:length(sm.input)
        if !(i in fix_inputs)
            new_set = SBitSet{N,T}()
            new_set = union(fix_inputs, SBitSet{32, UInt32}(i))
            sdp = calculate_sdp(sm, new_set)
            # println("new_set: ", new_set, " sdp: ", sdp)
            if sdp >= worst_from_best_sdp
                push!(best_results, (new_set, sdp))
                if length(best_results) > num_best
                    sort!(best_results, by=x->x[2], rev=true)
                    pop!(best_results)
                    worst_from_best_sdp = best_results[end][2]
                end
            end
        end
    end
    return best_results
end

function get_best_sdp(sm::Subset_minimal, best_results::Array{Tuple{SBitSet{N,T}, Float32}}, num_best::Int) where {N, T}
    # fix_inputs = SBitSet{32, UInt32}()

    # print_sets(best_results)
    first_of_the_first = first_bunch_of_best_sdp(sm, best_results[end][1], num_best)
    # print_sets(first_of_the_first)
    pop!(best_results)
    # print_sets(best_results)

    for bs in best_results
        first_of_the_first = best_of_the_fisrt_best(sm, first_of_the_first, bs[1], num_best)
        # println("bs:", bs[1])
        # print_sets(first_of_the_first)
    end

    return first_of_the_first
end

function get_best_best_sdp(sm::Subset_minimal, num_best::Int) where {N, T}
    fix_inputs = SBitSet{32, UInt32}()
    the_most_first = first_bunch_of_best_sdp(sm, fix_inputs, num_best)

    tmp = get_best_sdp(sm, the_most_first, num_best)
    println("THE END, best sdp: ", tmp[1][2])

    sdp_val = 0
    i = 0
    while sdp_val < 0.9
        tmp = get_best_sdp(sm, tmp, num_best)
        println("THE END of ", i, " best sdp: ", tmp[1][2])    
        i += 1
        sdp_val = tmp[1][2]
    end
    print_sets(tmp)

    return tmp[1][1]  # return the best set
end


function print_sets(sets::Array{Tuple{SBitSet{N,T}, Float32}}) where {N, T}
    println("Number of sets: ", length(sets))
    for s in sets
        println("Set: ", s[1], " sdp: ", s[2])
    end
end