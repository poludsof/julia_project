function make_h_vals(sm::Subset_minimal)
    return function((I3, I2, I1), num_samples)
        (sdp_full(sm.nn, sm.input, I3, num_samples),
        sdp_full(sm.nn[2:3], sm.nn[1](sm.input), I2, num_samples),
        sdp_full(sm.nn[3], sm.nn[1:2](sm.input), I1, num_samples),
        sdp_partial(sm.nn[1], sm.input, I3, I2, num_samples),
        sdp_partial(sm.nn[1:2], sm.input, I3, I1, num_samples),
        sdp_partial(sm.nn[2], sm.nn[1](sm.input), I2, I1, num_samples),
        )
    end
end

function make_heuristic(sm::Subset_minimal)
    h_vals = make_h_vals(sm)
    return function((I3, I2, I1), confidence, num_samples)
        hs = h_vals((I3, I2, I1), num_samples)
        (;
        hsum = mapreduce(x -> max(0, confidence-x), +, hs),
        hmax = mapreduce(x -> max(0, confidence-x), max, hs),
        )
    end
end

function make_max_error(sm::Subset_minimal)
    h_vals = make_h_vals(sm)
    return function((I3, I2, I1), confidence, num_samples)
        hs = h_vals((I3, I2, I1), num_samples)
        mapreduce(x -> max(0, confidence - x), max, hs)
    end
end
 

function make_find_best(sm::Subset_minimal)
    heuristic = make_heuristic(sm)
    return function((I3,I2,I1), confidence, num_samples)
        ii₁ = map(setdiff(1:784, I3)) do i
            Iᵢ = push(I3, i)
            (;ii = (Iᵢ, I2, I1), h = heuristic((Iᵢ, I2, I1), confidence, num_samples))
        end
        ii₂ = map(setdiff(1:256, I2)) do i
            Iᵢ = push(I2, i)
            (;ii = (I3, Iᵢ, I1), h = heuristic((I3, Iᵢ, I1), confidence, num_samples))
        end
        ii₃ = map(setdiff(1:256, I1)) do i
            Iᵢ = push(I1, i)
            (;ii = (I3,I2, Iᵢ), h = heuristic((I3, I2, Iᵢ), confidence, num_samples))
        end
        sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h.hsum < j.h.hsum)
    end
end


function make_expand_bcwd(sm::Subset_minimal)
    max_error = make_max_error(sm)
    return function(stack, closed_list, (I3, I2, I1), initial_total_err, confidence, num_samples)
        for i in collect(I3)
            new_subsets = (pop(I3, i), I2, I1)
            new_error = max_error(new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
        for i in collect(I2)
            new_subsets = (I3, pop(I2, i), I1)
            new_error = max_error(new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
        for i in collect(I1)
            new_subsets = (I3, I2, pop(I1, i))
            new_error = max_error(new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
        stack
    end
end


function make_expand_frwd(sm::Subset_minimal)
    heuristic = make_heuristic(sm)
    return function(stack, closed_list, (I3, I2, I1), confidence, num_samples)
        for i in setdiff(1:784, I3)
            new_subsets = (push(I3, i), I2, I1)
            new_heuristic, new_error = heuristic(new_subsets, confidence, num_samples)
            if new_subsets ∉ closed_list
                push!(stack, (new_heuristic, new_error, new_subsets))    
            end
        end
        for i in setdiff(1:256, I2)
            new_subsets = (I3, push(I2, i), I1)
            new_heuristic, new_error = heuristic(new_subsets, confidence, num_samples)
            if new_subsets ∉ closed_list
                push!(stack, (new_heuristic, new_error, new_subsets))
            end
        end
        for i in setdiff(1:256, I1)
            new_subsets = (I3, I2, push(I1, i))
            new_heuristic, new_error = heuristic(new_subsets, confidence, num_samples)
            if new_subsets ∉ closed_list
                push!(stack, (new_heuristic, new_error, new_subsets))
            end
        end
        stack
    end
end
