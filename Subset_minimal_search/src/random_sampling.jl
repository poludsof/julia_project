
function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet, n::Int) 
    generate_random_img_with_fix_inputs(sm, collect(fix_inputs), n)
end

function generate_random_img_with_fix_inputs(sm::Subset_minimal, ii::Vector{<:Integer}, n::Int) 
    x = rand(0:1, length(sm.input), n)
    x[ii, :] .= sm.input[ii]
    x
end

function calculate_ep(sm::Subset_minimal, fix_inputs::SBitSet, num_samples::Int)
    x = generate_random_img_with_fix_inputs(sm, fix_inputs, num_samples)
    mean(Flux.softmax(sm.nn(x))[sm.output + 1,:] )
end

function calculate_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_samples::Int) where {N, T}
    x = generate_random_img_with_fix_inputs(sm, fix_inputs, num_samples)
    mean(Flux.onecold(nn(x)) .== sm.output + 1)
end


function beam_search(sm::Subset_minimal, calc_func::Function, fix_inputs::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
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
    first_of_the_first = beam_search(sm, calc_func, best_results[end][1], num_best, num_samples)
    pop!(best_results)

    for bs in best_results
        first_of_the_first = beam_search(sm, calc_func, bs[1], num_best, num_samples, first_of_the_first)
    end

    return first_of_the_first
end


# scoring_func:
# 1)calculate_ep
# 2)calculate_sdp
function get_minimal_set_generic(sm::Subset_minimal, calc_func::Function, threshold=0.9, num_best=5, num_samples=100)
    fix_inputs = SBitSet{32, UInt32}()
    the_most_first = beam_search(sm, calc_func, fix_inputs, num_best, num_samples)
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