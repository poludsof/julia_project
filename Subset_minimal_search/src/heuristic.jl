
function isvalid(criterium, current_error, threshold_error)
    if criterium == criterium_sdp || criterium == criterium_ep
        if current_error <= threshold_error
            return true
        end
    end
    return false
end


function uniform_distribution(img, ii, num_samples)
    sample_input(img, ii, num_samples)
end

function data_distribution(img, ii, data_model, num_samples)
    r = data_model
    r = condition(r, img, ii)
    sample_all(r, num_samples)
end

function mi_distribution(sm::Subset_minimal, ii, num_samples)
    #todo
end


function h_vals(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3, I2, I1), data_model, num_samples)
    (calc_func(sm.nn, sm.input, I3, data_model, num_samples),
    calc_func(sm.nn[2:3], sm.nn[1](sm.input), I2, data_model, num_samples),
    calc_func(sm.nn[3], sm.nn[1:2](sm.input), I1, data_model, num_samples),
    calc_func_partial(sm.nn[1], sm.input, I3, I2, data_model, num_samples),
    calc_func_partial(sm.nn[1:2], sm.input, I3, I1, data_model, num_samples),
    calc_func_partial(sm.nn[2], sm.nn[1](sm.input), I2, I1, data_model, num_samples),
    )
end

function heuristic(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3, I2, I1), confidence, data_model, num_samples)
    hs = h_vals(sm, calc_func, calc_func_partial, (I3, I2, I1), data_model, num_samples)
    (;
    hsum = mapreduce(x -> max(0, confidence - x), +, hs),
    hmax = mapreduce(x -> max(0, confidence - x), max, hs),
    )
end

function max_error(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3, I2, I1), confidence, data_model, num_samples)
    hs = h_vals(sm, calc_func, calc_func_partial, (I3, I2, I1), data_model, num_samples)
    mapreduce(x -> max(0, confidence - x), max, hs)
end
 



function find_best(sm::Subset_minimal, calc_func::Function, calc_func_partial::Function, (I3,I2,I1), confidence, data_model, num_samples)
    ii₁ = map(setdiff(1:784, I3)) do i
        Iᵢ = push(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm, calc_func, calc_func_partial, (Iᵢ, I2, I1), confidence, data_model, num_samples))
    end
    ii₂ = map(setdiff(1:256, I2)) do i
        Iᵢ = push(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm, calc_func, calc_func_partial, (I3, Iᵢ, I1), confidence, data_model, num_samples))
    end
    ii₃ = map(setdiff(1:256, I1)) do i
        Iᵢ = push(I1, i)
        (;ii = (I3,I2, Iᵢ), h = heuristic(sm, calc_func, calc_func_partial, (I3, I2, Iᵢ), confidence, data_model, num_samples))
    end
    sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h.hsum < j.h.hsum)
end
