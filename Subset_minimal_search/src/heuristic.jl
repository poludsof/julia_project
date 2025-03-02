
function h_vals(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3, I2, I1), num_samples)
    (calc_func(sm.nn, sm.input, I3, num_samples),
    calc_func(sm.nn[2:3], sm.nn[1](sm.input), I2, num_samples),
    calc_func(sm.nn[3], sm.nn[1:2](sm.input), I1, num_samples),
    calc_func_partial(sm.nn[1], sm.input, I3, I2, num_samples),
    calc_func_partial(sm.nn[1:2], sm.input, I3, I1, num_samples),
    calc_func_partial(sm.nn[2], sm.nn[1](sm.input), I2, I1, num_samples),
    )
end

function heuristic(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3, I2, I1), confidence, num_samples)
    hs = h_vals(sm, calc_func, calc_func_partial, (I3, I2, I1), num_samples)
    (;
    hsum = mapreduce(x -> max(0, confidence-x), +, hs),
    hmax = mapreduce(x -> max(0, confidence-x), max, hs),
    )
end

function max_error(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3, I2, I1), confidence, num_samples)
    hs = h_vals(sm, calc_func, calc_func_partial, (I3, I2, I1), num_samples)
    mapreduce(x -> max(0, confidence - x), max, hs)
end
 

function find_best(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3,I2,I1), confidence, num_samples)
    ii₁ = map(setdiff(1:784, I3)) do i
        Iᵢ = push(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm, calc_func, calc_func_partial, (Iᵢ, I2, I1), confidence, num_samples))
    end
    ii₂ = map(setdiff(1:256, I2)) do i
        Iᵢ = push(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm, calc_func, calc_func_partial, (I3, Iᵢ, I1), confidence, num_samples))
    end
    ii₃ = map(setdiff(1:256, I1)) do i
        Iᵢ = push(I1, i)
        (;ii = (I3,I2, Iᵢ), h = heuristic(sm, calc_func, calc_func_partial, (I3, I2, Iᵢ), confidence, num_samples))
    end
    sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h.hsum < j.h.hsum)
end


function expand_bcwd(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, stack, closed_list, (I3, I2, I1), initial_total_err, confidence, num_samples)
    if I3 !== nothing
        for i in collect(I3)
            new_subsets = (pop(I3, i), I2, I1)
            new_error = max_error(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
    end
    if I2 !== nothing
        for i in collect(I2)
            new_subsets = (I3, pop(I2, i), I1)
            new_error = max_error(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
    end
    if  I1 !== nothing
        for i in collect(I1)
            new_subsets = (I3, I2, pop(I1, i))
            new_error = max_error(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
            if new_error <= initial_total_err && new_subsets ∉ closed_list
                push!(stack, (new_error, new_subsets))
            end
        end
    end
    stack
end


function expand_frwd(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, stack, closed_list, (I3, I2, I1), confidence, num_samples)
    if I3 !== nothing
        for i in setdiff(1:784, I3)
            new_subsets = (push(I3, i), I2, I1)
            new_heuristic, new_error = heuristic(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
            if new_subsets ∉ closed_list
                push!(stack, (new_heuristic, new_error, new_subsets))    
            end
        end
    end
    if I2 !== nothing
        for i in setdiff(1:256, I2)
            new_subsets = (I3, push(I2, i), I1)
            new_heuristic, new_error = heuristic(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
            if new_subsets ∉ closed_list
                push!(stack, (new_heuristic, new_error, new_subsets))
            end
        end
    end
    if I1 !== nothing
        for i in setdiff(1:256, I1)
            new_subsets = (I3, I2, push(I1, i))
            new_heuristic, new_error = heuristic(sm, calc_func, calc_func_partial, new_subsets, confidence, num_samples)
            if new_subsets ∉ closed_list
                push!(stack, (new_heuristic, new_error, new_subsets))
            end
        end
    end
    stack
end
