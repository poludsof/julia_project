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


function greedy_subsets_search(sm::Subset_minimal, threshold_total_err=0.1, num_samples=1000)
    confidence = 1-threshold_total_err
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    full_error = heuristic(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)

    while full_error.hmax > 0
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


# Priority on the length of subsets
function backward_priority_reduction(sm::Subset_minimal, (I3, I2, I1), threshold, num_samples=100)
    confidence = 1 - threshold
    initial_total_err = max_error(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)
    println("Initial max error: ", initial_total_err)

    stack = [(max_error(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples), (I3, I2, I1))]
    closed_list = Set{Tuple{SBitSet, SBitSet, SBitSet}}()

    best_subsets = (I3, I2, I1)
    best_total_len = length(best_subsets[1]) + length(best_subsets[2]) + length(best_subsets[3])

    max_steps = 300
    steps = 0

    while !isempty(stack)
        steps += 1
		steps > max_steps && break

        sort!(stack, by = x -> (-(length(x[2][1]) + length(x[2][2]) + length(x[2][3])), -x[1]))
        current_error, current_subsets = pop!(stack)

        push!(closed_list, current_subsets)

        println("step: ", steps, ", length: $((length(current_subsets[1]), length(current_subsets[2]), length(current_subsets[3]))) Current error: ", current_error)

        total_len = length(current_subsets[1]) + length(current_subsets[2]) + length(current_subsets[3])
        if total_len < best_total_len
            best_subsets = current_subsets
            best_total_len = total_len
        end
        
        stack = expand_bcwd(sm, stack, closed_list, current_subsets, initial_total_err, confidence, num_samples)

    end
    println("Final length: $((length(best_subsets[1]), length(best_subsets[2]), length(best_subsets[3])))")
    return best_subsets
end


function expand_bcwd(sm::Subset_minimal, stack, closed_list, (I3, I2, I1), initial_total_err, confidence, num_samples)
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
    stack
end


function forward_priority_search(sm::Subset_minimal, threshold_total_err=0.1, num_samples=100)
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    confidence = 1 - threshold_total_err
    
    # full_error = max_error(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)
    initial_heuristic, full_error = heuristic(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)
    println("Initial error: ", full_error, " Initial heuristic: ", initial_heuristic)

    stack = [(initial_heuristic, full_error, (I3, I2, I1))]
    array_of_the_best = []
    closed_list = Set{Tuple{SBitSet, SBitSet, SBitSet}}()

    steps = 0
    max_steps = 100

    while !isempty(stack)

        # steps > max_steps && break
        steps += 1

        sort!(stack, by = x -> -x[1])
        current_heuristic, current_error, (I3, I2, I1) = pop!(stack)
        closed_list = push!(closed_list, (I3, I2, I1))
        
        if current_error <= 0
            push!(array_of_the_best, (I3, I2, I1))
            println("Valid subset found: $((length(I3), length(I2), length(I1))) with error: ", current_error)
            return (I3, I2, I1)
        end

        println("step: $steps , length $((length(I3), length(I2), length(I1))) Expanding state with error: $current_error, heuristic: $current_heuristic")

        stack = expand_frwd(sm, stack, closed_list, (I3, I2, I1), confidence, num_samples)
    end

    println("Stack is empty")
    return array_of_the_best
end

function expand_frwd(sm::Subset_minimal, stack, closed_list, (I3, I2, I1), confidence, num_samples)
    for i in setdiff(1:784, I3)
        new_subsets = (push(I3, i), I2, I1)
        new_heuristic, new_error = heuristic(sm.nn, sm.input, new_subsets, confidence, num_samples)
        if new_subsets ∉ closed_list
            push!(stack, (new_heuristic, new_error, new_subsets))    
        end
    end
    for i in setdiff(1:256, I2)
        new_subsets = (I3, push(I2, i), I1)
        new_heuristic, new_error = heuristic(sm.nn, sm.input, new_subsets, confidence, num_samples)
        if new_subsets ∉ closed_list
            push!(stack, (new_heuristic, new_error, new_subsets))
        end
    end
    for i in setdiff(1:256, I1)
        new_subsets = (I3, I2, push(I1, i))
        new_heuristic, new_error = heuristic(sm.nn, sm.input, new_subsets, confidence, num_samples)
        if new_subsets ∉ closed_list
            push!(stack, (new_heuristic, new_error, new_subsets))
        end
    end
    stack
end

#=
function remove(sm::Subset_minimal, (I3,I2,I1), num_samples)
    for i in I3
        Iᵢ = pop(I3, i)
        if max_error(sm.nn, sm.input, (Iᵢ, I2, I1), num_samples) == 0
            I3 = Iᵢ
        end
    end

    for i in I2
        Iᵢ = pop(I2, i)
        if max_error(sm.nn, sm.input, (I3, Iᵢ, I1), num_samples) == 0
            I2 = Iᵢ
        end
    end

    for i in I1
        Iᵢ = pop(I1, i)
        if max_error(sm.nn, sm.input, (I3, I2, Iᵢ), num_samples) == 0
            I1 = Iᵢ
        end
    end
    (I3, I2, I1)
end
=#