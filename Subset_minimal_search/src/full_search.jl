
# function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet, n::Int) 
#     generate_random_img_with_fix_inputs(sm, collect(fix_inputs), n)
# end

# function generate_random_img_with_fix_inputs(model, input, ii::Vector{<:Integer}, n::Int) 
#     # x = rand(0:1, length(sm.input), n)
#     # x[ii, :] .= sm.input[ii]
#     # x
#     x = rand([-1,1], length(sm.input), n)
# 	x[ii,:] .= sm.input[ii]
#     x 
# end

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


function beam_search(model, img, fix_inputs::SBitSet{N, T}, num_best::Int, num_samples::Int, best_results=Array{Tuple{SBitSet{N, T}, Float32}, 1}()) where {N, T}
    worst_from_best_threshold = isempty(best_results) ? 0.0 : best_results[end][2]

    for i in 1:length(img)
        if !(i in fix_inputs)
            new_set_I = SBitSet{N, T}()
            new_set_I = union(fix_inputs, SBitSet{32, UInt32}(i))
            threshold = sdp_full(model, img, new_set_I, num_samples) # criteruim (ep/sdp)
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


function full_beam_search(sm::Subset_minimal, threshold=0.9, num_best=1, num_samples=100)
    I3 = beam_search(sm.nn, sm.input, SBitSet{32, UInt32}(), num_best, num_samples)
    I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), SBitSet{32, UInt32}(), num_best, num_samples)
    I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), SBitSet{32, UInt32}(), num_best, num_samples)

    full_error = heuristic(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))

    length = 1
    while full_error > 2
        I3 = beam_search(sm.nn, sm.input, I3[1][1], num_best, num_samples)
        I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), I2[1][1], num_best, num_samples)
        I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), I1[1][1], num_best, num_samples)

        full_error = heuristic(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))
        println("Length of ii: $length, full_error: ", full_error)    
        length += 1
    end
    return I3[1][1], I2[1][1], I1[1][1]
end


function heuristic(model, xp, (I3, I2, I1))
    num_samples = 1000
    # println("num_samples: ", num_samples)
	max(0, 0.9- sdp_full(model, xp, I3, num_samples)) +
	max(0, 0.9 - sdp_full(model[2:3], model[1](xp), I2, num_samples)) +
	max(0, 0.9 - sdp_full(model[3], model[1:2](xp), I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1], xp, I3, I2, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1:2], xp, I3, I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[2], model[1](xp), I2, I1, num_samples))
end