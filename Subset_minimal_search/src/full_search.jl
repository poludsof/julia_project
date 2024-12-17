@inline function sample_input(img::AbstractVector, ii::SBitSet, num_samples::Integer)
    ii = collect(ii)
    x = similar(img, length(img), num_samples)
    @inbounds for col in axes(x, 2)
        for i in axes(x, 1)
            x[i, col] = 2*rand(Bool) - 1
        end
        for i in ii
            x[i, col] = img[i]
        end
    end
    x
end

function sdp_full(model, img, ii, num_samples)
    x = sample_input(img, ii, num_samples)
    mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

function sdp_partial(model, img, ii, jj, num_samples) # ii = I2 --> jj = I1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = sample_input(img, ii, num_samples)
    mean(model(x)[jj, :] .== model(img)[jj, :])
end

function h_vals(model, xp, (I3, I2, I1), num_samples)
    (sdp_full(model, xp, I3, num_samples),
    sdp_full(model[2:3], model[1](xp), I2, num_samples),
    sdp_full(model[3], model[1:2](xp), I1, num_samples),
    sdp_partial(model[1], xp, I3, I2, num_samples),
    sdp_partial(model[1:2], xp, I3, I1, num_samples),
    sdp_partial(model[2], model[1](xp), I2, I1, num_samples),
    )
end

function heuristic(model, xp, (I3, I2, I1), confidence, num_samples)
    hs = h_vals(model, xp, (I3, I2, I1), num_samples)
    (;
    hsum = mapreduce(x -> max(0, confidence-x), +, hs),
    hmax = mapreduce(x -> max(0, confidence-x), max, hs),
    )
end

function max_error(model, xp, (I3, I2, I1), confidence, num_samples)
    hs = h_vals(model, xp, (I3, I2, I1), num_samples)
    mapreduce(x -> max(0, confidence - x), max, hs)
end
 
function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end


function full_beam_search2(sm::Subset_minimal, threshold_total_err=0.1, num_samples=1000)
    confidence = 1-threshold_total_err
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    full_error = heuristic(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)

    while full_error.hmax >= 0
        candidate = find_best(sm, (I3,I2,I1), confidence, num_samples)[1]
        I3, I2, I1 = candidate.ii

        full_error = candidate.h
        println("Length of ii: $((length(I3), length(I2), length(I1))), full_error: ", candidate.h)
    end
    return I3, I2, I1
end

function find_best(sm::Subset_minimal, (I3,I2,I1), confidence, num_samples)
    ii₁ = map(setdiff(1:784, I3)) do i
        Iᵢ = push(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm.nn, sm.input, (Iᵢ, I2, I1), confidence, num_samples))
    end
    ii₂ = map(setdiff(1:256, I2)) do i
        Iᵢ = push(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm.nn, sm.input, (I3, Iᵢ, I1), confidence, num_samples))
    end

    ii₃ = map(setdiff(1:256, I1)) do i
        Iᵢ = push(I1, i)
        (;ii = (I3,I2, Iᵢ), h = heuristic(sm.nn, sm.input, (I3, I2, Iᵢ), confidence, num_samples))
    end
    sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h.hsum < j.h.hsum)
end


function backward_dfs_search(sm::Subset_minimal, (I3, I2, I1), num_samples=100)
    confidence = 1 - 0.05
    initial_total_err = max_error(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)
    println("Initial max error: ", initial_total_err)
    stack = [(max_error(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples), (I3, I2, I1))]
    closed_list = Set{Tuple{SBitSet, SBitSet, SBitSet}}()

    best_subsets = (I3, I2, I1)
    best_total_len = length(best_subsets[1]) + length(best_subsets[2]) + length(best_subsets[3])

    steps = 0
    max_steps = 300

    while !isempty(stack)

        steps += 1
		steps > max_steps && break

        sort!(stack, by = x -> (-(length(x[2][1]) + length(x[2][2]) + length(x[2][3])), -x[1]))
        # println("first element h: ", stack[1][1], " last element h: ", stack[end][1])
        current_error, current_subsets = pop!(stack)

        push!(closed_list, current_subsets)

        # current_error = max_error(sm.nn, sm.input, current_subsets, num_samples)
        println("step: ", steps, ", length: $((length(current_subsets[1]), length(current_subsets[2]), length(current_subsets[3]))) Current error: ", current_error)

        total_len = length(current_subsets[1]) + length(current_subsets[2]) + length(current_subsets[3])
        if total_len < best_total_len
            best_subsets = current_subsets
            best_total_len = length(best_subsets[1]) + length(best_subsets[2]) + length(best_subsets[3])
        end
        I3, I2, I1 = deepcopy(current_subsets)
        
        for i in collect(I3)
            new_subsets = (pop(I3, i), I2, I1)
            new_error = max_error(sm.nn, sm.input, new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end

        for i in collect(I2)
            new_subsets = (I3, pop(I2, i), I1)
            new_error = max_error(sm.nn, sm.input, new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end

        for i in collect(I1)
            new_subsets = (I3, I2, pop(I1, i))
            new_error = max_error(sm.nn, sm.input, new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
    end
    println("final length: $((length(best_subsets[1]), length(best_subsets[2]), length(best_subsets[3])))")
    return best_subsets
end



function full_beam_search_with_stack(sm::Subset_minimal, threshold_total_err=0.1, num_samples=100)
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    confidence = 1 - threshold_total_err
    
    # full_error = max_error(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)
    initial_heuristic, full_error = heuristic(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, (I3, I2, I1))]
    array_of_the_best = []

    steps = 0
    max_steps = 100

    while !isempty(stack)

        # steps > max_steps && break
        steps += 1

        sort!(stack, by = x -> -x[1]) 
        # println("first element: ", stack[1], " last element: ", stack[end])
        current_heuristic, current_error, (current_I3, current_I2, current_I1) = pop!(stack)
        
        if current_error <= 0
            push!(array_of_the_best, (current_I3, current_I2, current_I1))
            println("Valid subset found: $((length(current_I3), length(current_I2), length(current_I1))) with error: ", current_error)
            return current_I3, current_I2, current_I1
        end

        println("step: $steps , length $((length(current_I3), length(current_I2), length(current_I1))) Expanding state with error: $current_error, heuristic: $current_heuristic")

        for i in setdiff(1:784, current_I3)
            I3_new = push(current_I3, i)
            new_heuristic, new_error = heuristic(sm.nn, sm.input, (I3_new, current_I2, current_I1), confidence, num_samples)
            # new_error = max_error(sm.nn, sm.input, (I3_new, current_I2, current_I1), confidence, num_samples)
            push!(stack, (new_heuristic, new_error, (I3_new, current_I2, current_I1)))
        end

        for i in setdiff(1:256, current_I2)
            I2_new = push(current_I2, i)
            new_heuristic, new_error = heuristic(sm.nn, sm.input, (current_I3, I2_new, current_I1), confidence, num_samples)
            # new_error = max_error(sm.nn, sm.input, (current_I3, I2_new, current_I1), confidence, num_samples)
            push!(stack, (new_heuristic, new_error, (current_I3, I2_new, current_I1)))
        end

        for i in setdiff(1:256, current_I1)
            I1_new = push(current_I1, i)
            new_heuristic, new_error = heuristic(sm.nn, sm.input, (current_I3, current_I2, I1_new), confidence, num_samples)
            # new_error = max_error(sm.nn, sm.input, (current_I3, current_I2, I1_new), confidence, num_samples)
            push!(stack, (new_heuristic, new_error, (current_I3, current_I2, I1_new)))
        end
    end

    println("Stack is empty")
    return array_of_the_best
    # return (I3, I2, I1)
end
