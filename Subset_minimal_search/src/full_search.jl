

function greedy_subsets_search(sm::Subset_minimal; threshold_total_err=0.1, num_samples=1000)
    confidence = 1 - threshold_total_err
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    full_error = heuristic(sm, (I3, I2, I1), confidence, num_samples)

    while full_error.hmax > 0
        candidate = find_best(sm, (I3, I2, I1), confidence, num_samples)[1]
        I3, I2, I1 = candidate.ii

        full_error = candidate.h
        println("Length of ii: $((length(I3), length(I2), length(I1))), full_error: ", candidate.h)
    end
    return I3, I2, I1
end