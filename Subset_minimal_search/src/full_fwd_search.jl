

function full_forward_search(sm::Subset_minimal, threshold=0.9, num_samples=100)
    all_I3, reas = forward_search(sm.nn, sm.input, sm.output, max_steps=30, sdp_threshold=0.5, num_samples=num_samples)
    all_I2, reas = forward_search(sm.nn[2:3], sm.nn[1](sm.input), sm.output, max_steps=30, sdp_threshold=0.5, num_samples=num_samples)
    all_I1, reas = forward_search(sm.nn[3], sm.nn[1:2](sm.input), sm.output, max_steps=30, sdp_threshold=0.5, num_samples=num_samples)
    I3, sdp_I3 = choose_best_solution(all_I3, sm.nn, sm.input, num_samples)
    I2, sdp_I2 = choose_best_solution(all_I2, sm.nn[2:3], sm.nn[1](sm.input), num_samples)
    I1, sdp_I1 = choose_best_solution(all_I1, sm.nn[3], sm.nn[1:2](sm.input), num_samples)
    full_error = heuristic(sm.nn, sm.input, (I3, I2, I1))

    println("I3: ", I3)
    println("I2: ", I2)
    println("I1: ", I1)
    println("full_error: ", full_error)

    # length = 1
    # while full_error > 2
    #     I3 = beam_search(sm.nn, sm.input, I3[1][1], num_best, num_samples)
    #     I2 = beam_search(sm.nn[2:3], sm.nn[1](sm.input), I2[1][1], num_best, num_samples)
    #     I1 = beam_search(sm.nn[3], sm.nn[1:2](sm.input), I1[1][1], num_best, num_samples)

    #     full_error = heuristic(sm.nn, sm.input, (I3[1][1], I2[1][1], I1[1][1]))
    #     println("Length of ii: $length, full_error: ", full_error)    
    #     length += 1
    # end
    # return I3[1][1], I2[1][1], I1[1][1]
end

function heuristic(model, xp, (I3, I2, I1))
    num_samples = 1000
	max(0, 0.9- sdp_full(model, xp, I3, num_samples)) +
	max(0, 0.9 - sdp_full(model[2:3], model[1](xp), I2, num_samples)) +
	max(0, 0.9 - sdp_full(model[3], model[1:2](xp), I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1], xp, I3, I2, num_samples)) +
	max(0, 0.9 - sdp_partial(model[1:2], xp, I3, I1, num_samples)) +
	max(0, 0.9 - sdp_partial(model[2], model[1](xp), I2, I1, num_samples))
end