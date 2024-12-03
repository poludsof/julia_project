
function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet, n::Int) 
    generate_random_img_with_fix_inputs(sm, collect(fix_inputs), n)
end

function generate_random_img_with_fix_inputs(model, input, ii::Vector{<:Integer}, n::Int) 
    # x = rand(0:1, length(sm.input), n)
    # x[ii, :] .= sm.input[ii]
    # x
    x = rand([-1,1], length(sm.input), n)
	x[ii,:] .= sm.input[ii]
    x 
end

function sdp_full(model, img, ii, num_samples)
    # x = generate_random_img_with_fix_inputs(model, input, fix_inputs, num_samples)
    ii = collect(ii)
    x = rand([-1,1], length(img), num_samples)
	x[ii,:] .= img[ii]
	mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

function sdp_partial(model, img, ii, jj, num_samples) # ii = I2 --> jj = I1
    ii = collect(ii)
    jj = collect(jj)
	x = rand([-1,1], length(img), num_samples)
	x[ii,:] .= img[ii]
	mean(model(x)[jj, :] .== model(img)[jj, :])
end


function beam_search(sm::Subset_minimal, calc_func::Function, fix_inputs::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_threshold = isempty(best_results) ? 0.0 : best_results[end][2]
    # println("Worst from best threshold: ", worst_from_best_threshold)

    for i in 1:length(sm.input)
        if !(i in fix_inputs)
            new_set_I = SBitSet{N, T}()
            new_set_I = union(fix_inputs, SBitSet{32, UInt32}(i))
            threshold = sdp_full(sm.nn, sm.input, new_set_I, num_samples) # criteruim (ep/sdp)
            if threshold >= worst_from_best_threshold
                push!(best_results, (new_set_I, threshold))
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


function part_beam_search_I2(sm::Subset_minimal, calc_func::Function, I2::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:256
        if !(i in I2)
            new_set_I = SBitSet{N, T}()
            new_set_I = union(I2, SBitSet{32, UInt32}(i))
            threshold = sdp_full(sm.nn[2:3], sm.nn[1](sm.input), new_set_I, num_samples) # criteruim (ep/sdp)
            if threshold >= worst_from_best_threshold
                push!(best_results, (new_set_I, threshold))
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

function part_beam_search_I3(sm::Subset_minimal, calc_func::Function, I1::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:256
        if !(i in I1)
            new_set_I = SBitSet{N, T}()
            new_set_I = union(I1, SBitSet{32, UInt32}(i))
            threshold = sdp_full(sm.nn[3], sm.nn[1:2](sm.input), new_set_I, num_samples) # criteruim (ep/sdp)
            if threshold >= worst_from_best_threshold
                push!(best_results, (new_set_I, threshold))
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
    I2 = SBitSet{32, UInt32}()
    I1 = SBitSet{32, UInt32}()

    I3 = beam_search(sm, calc_func, I3, num_best, num_samples)
    I2 = part_beam_search_I2(sm, calc_func, I2, num_best, num_samples)
    # println(first_i_I2)
    # sdp_partial(sm.nn[1:2], sm.input, first_i_I3[1][1], first_i_I2[1][1], 10000)
    I1 = part_beam_search_I3(sm, calc_func, I1, num_best, num_samples)

    # heuristic_val = heuristic(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))
    # println("Heuristic value: ", heuristic_val)
    I3 = beam_search(sm, calc_func, I3[1][1], num_best, num_samples)
    I2 = part_beam_search_I2(sm, calc_func, I2[1][1], num_best, num_samples)
    I1 = part_beam_search_I3(sm, calc_func, I1[1][1], num_best, num_samples)

    println("I1:", I1)

    # full_error = 0.0

    # i = 1
    # while full_error > 5
    #     I3 = beam_search(sm, calc_func, I3, num_best, num_samples)
    #     I2 = part_beam_search_I2(sm, calc_func, I2, num_best, num_samples)
    #     I1 = part_beam_search_I3(sm, calc_func, I1, num_best, num_samples)
    #     heuristic_val = heuristic(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))

    #     println("THE END of $i, best score: ", tmp[1][2])    
    #     i += 1
    #     score_val = tmp[1][2]  # get the best score
    # end
    # print_sets(tmp)

    # return tmp[1][1]  # return set with the best score
end


function heuristic(model, xp, (I3, I2, I1))
	# independent criteria
	max(0, 0.95 - sdp_full(model, xp, I3, 1000)) +
	max(0, 0.95 - sdp_full(model[2:3], model[1](xp), I2, 100)) +
	max(0, 0.95 - sdp_full(model[3], model[1:2](xp), I1, 100)) +
	# independent criteria
	max(0, 0.95 - sdp_partial(model[1], xp, I3, I2, 100)) +
	max(0, 0.95 - sdp_partial(model[1:2], xp, I3, I1, 100)) +
	max(0, 0.95 - sdp_partial(model[2], model[1](xp), I2, I1, 100))
end




# Helper functions
function print_sets(sets::Array{Tuple{SBitSet{N,T}, Float32}}) where {N, T}
    println("Number of sets: ", length(sets))
    for s in sets
        println("Set: ", s[1], " metric: ", s[2])
    end
end