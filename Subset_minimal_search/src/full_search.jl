
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
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
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


function full_beam_search(sm::Subset_minimal, threshold_total_err=0.1, num_best=1, num_samples=100)
    I3 = beam_search(sm.nn, sm.input, SBitSet{32, UInt32}(), num_best, num_samples)
    I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), SBitSet{32, UInt32}(), num_best, num_samples)
    I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), SBitSet{32, UInt32}(), num_best, num_samples)

    full_error = max_error(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))

    length = 1
    while full_error < threshold_total_err
        I3 = beam_search(sm.nn, sm.input, I3[1][1], num_best, num_samples)
        
        if length < 256
            I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), I2[1][1], num_best, num_samples)
            I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), I1[1][1], num_best, num_samples)
        else
            println("length is greater than 256")
        end


        full_error = max_error(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))
        println("Length of ii: $length, full_error: ", full_error)    
        length += 1
    end
    return I3[1][1], I2[1][1], I1[1][1]
end



function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end

function full_beam_search2(sm::Subset_minimal, threshold_total_err=0.1, num_samples=100)
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    full_error = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)

    while full_error > threshold_total_err
        candidate = add_best(sm, (I3,I2,I1))[1]
        I3, I2, I1 = candidate.ii

        full_error = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)
        println("Length of ii: $((length(I3), length(I2), length(I1))), full_error: ", full_error, " heuristic: ", candidate.h)
    end
    return I3, I2, I1
end


function add_best(sm::Subset_minimal, (I3, I2, I1), num_samples = 100)
    # searching through I3
    ii₁ = map(setdiff(1:784, I3)) do i
        Iᵢ = push(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm.nn, sm.input, (Iᵢ, I2, I1)), num_samples)
    end
    ii₂ = map(setdiff(1:256, I2)) do i
        Iᵢ = push(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm.nn, sm.input, (I3, Iᵢ, I1)), num_samples)
    end

    ii₃ = map(setdiff(1:256, I1)) do i
        Iᵢ = push(I1, i)
        (;ii = (I3,I2, Iᵢ), h = heuristic(sm.nn, sm.input, (I3, I2, Iᵢ)), num_samples)
    end
    sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h < j.h)
end


function backward_search(sm::Subset_minimal, (I3, I2, I1), num_samples=100)
    ii₁ = map(collect(I3)) do i
        Iᵢ = pop(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm.nn, sm.input, (Iᵢ, I2, I1), num_samples))
    end
    
    ii₂ = map(collect(I2)) do i
        Iᵢ = pop(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm.nn, sm.input, (I3, Iᵢ, I1), num_samples))
    end
    
    ii₃ = map(collect(I1)) do i
        Iᵢ = pop(I1, i)
        (;ii = (I3, I2, Iᵢ), h = heuristic(sm.nn, sm.input, (I3, I2, Iᵢ), num_samples))
    end
    
    sort(vcat(ii₁, ii₂, ii₃), lt = (i, j) -> i.h < j.h)
end

function full_backward_search(sm::Subset_minimal, (I3, I2, I1), threshold_total_err=0.1, num_samples=100)
    full_error = max_error(sm.nn, sm.input, (I3, I2, I1), num_samples)
    println("Initial max error: ", full_error)

    while true
        candidates = backward_search(sm, (I3, I2, I1), num_samples)
        best_candidate = candidates[1]
        
        I3_tmp, I2_tmp, I1_tmp = best_candidate.ii
        new_error = max_error(sm.nn, sm.input, (I3_tmp, I2_tmp, I1_tmp), num_samples)
        
        println("Length of ii: $((length(I3), length(I2), length(I1))), new_error: ", new_error, " heuristic: ", best_candidate.h)
        
        if new_error > full_error
            break
        end
        
        I3, I2, I1 = I3_tmp, I2_tmp, I1_tmp
        full_error = new_error
    end
    
    return I3, I2, I1
end


function heuristic(model, xp, (I3, I2, I1), num_samples = 1000)
	max(0, 0.9 - sdp_full(model, xp, I3, num_samples)) +
	max(0, 0.9 - sdp_full(model[2:3], model[1](xp), I2, num_samples)) +
	max(0, 0.9 - sdp_full(model[3], model[1:2](xp), I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1], xp, I3, I2, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1:2], xp, I3, I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[2], model[1](xp), I2, I1, num_samples))
end


function max_error(model, xp, (I3, I2, I1), num_samples = 1000)
    max(max(0, 1.0 - sdp_full(model, xp, I3, num_samples)),
        max(0, 1.0 - sdp_full(model[2:3], model[1](xp), I2, num_samples)),
        max(0, 1.0 - sdp_full(model[3], model[1:2](xp), I1, num_samples)),
        max(0, 1.0 - sdp_partial(model[1], xp, I3, I2, num_samples)),
        max(0, 1.0 - sdp_partial(model[1:2], xp, I3, I1, num_samples)),
        max(0, 1.0 - sdp_partial(model[2], model[1](xp), I2, I1, num_samples))
    )
end