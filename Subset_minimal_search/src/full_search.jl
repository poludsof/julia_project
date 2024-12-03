
function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet, n::Int) 
    generate_random_img_with_fix_inputs(sm, collect(fix_inputs), n)
end

function generate_random_img_with_fix_inputs(sm::Subset_minimal, ii::Vector{<:Integer}, n::Int) 
    # x = rand(0:1, length(sm.input), n)
    # x[ii, :] .= sm.input[ii]
    # x
    x = rand([-1,1], length(sm.input), n)
	x[ii,:] .= sm.input[ii]
    x
end

function sdp_full(sm, fix_inputs, num_samples)
    x = generate_random_img_with_fix_inputs(sm, fix_inputs, num_samples)
    mean(Flux.onecold(sm.nn(x)) .== sm.output + 1)
end

function sdp_partial(sm::Subset_minimal, previous_layer_fix_inputs::SBitSet{N,T}, this_layer_fix_inputs::SBitSet{N,T}, num_samples::Int) where {N, T}
    # function criterium_partial(model, xp, ii, jj)
    x = generate_random_img_with_fix_inputs(sm, fix_inputs, num_samples)
    mean(model(x)[jj,:] .== model(sm.input)[jj,:])
end


function beam_search(sm::Subset_minimal, calc_func::Function, fix_inputs::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:length(sm.input)
        if !(i in fix_inputs)
            new_set = SBitSet{N, T}()
            new_set = union(fix_inputs, SBitSet{32, UInt32}(i))
            threshold = sdp_full(sm, new_set, num_samples) # criteruim (ep/sdp)
            if threshold >= worst_from_best_threshold
                push!(best_results, (new_set, threshold))
                if length(best_results) > num_best
                    sort!(best_results, by=x->x[2], rev=true)
                    pop!(best_results)
                    worst_from_best_threshold = best_results[end][2]
                end
            end
        end
    end

    return best_results
end


# function generate_array_of_top_sets(sm::Subset_minimal, calc_func::Function, best_results::Array{Tuple{SBitSet{N,T}, Float32}}, num_best::Int, num_samples::Int) where{N, T}
#     first_of_the_first = beam_search(sm, calc_func, best_results[end][1], num_best, num_samples)
#     pop!(best_results)

#     for bs in best_results
#         first_of_the_first = beam_search(sm, calc_func, bs[1], num_best, num_samples, first_of_the_first)
#     end

#     return first_of_the_first
# end


function full_beam_search(sm::Subset_minimal, calc_func::Function, threshold=0.9, num_best=1, num_samples=100)
    I3 = SBitSet{32, UInt32}()
    # I2 = SBitSet{32, UInt32}()
    # I1 = SBitSet{32, UInt32}()

    first_i_I3 = beam_search(sm, calc_func, I3, num_best, num_samples) # {1, 456, 752}
    
    println("done")

    # tmp = generate_array_of_top_sets(sm, calc_func, first_i_I3, num_best, num_samples)
    # score_val = 0.0
    # i = 1
    # while score_val < threshold
    #     tmp = generate_array_of_top_sets(sm, calc_func, tmp, num_best, num_samples)
    #     println("THE END of $i, best score: ", tmp[1][2])    
    #     i += 1
    #     score_val = tmp[1][2]  # get the best score
    # end
    # print_sets(tmp)

    # return tmp[1][1]  # return set with the best score
end




# Helper functions
function print_sets(sets::Array{Tuple{SBitSet{N,T}, Float32}}) where {N, T}
    println("Number of sets: ", length(sets))
    for s in sets
        println("Set: ", s[1], " metric: ", s[2])
    end
end