forward_search_I3 = make_forward_search(sm)

sm_I2 = Subset_minimal(sm.nn[2:3], sm.nn[1](sm.input), sm.output)
forward_search_I2 = make_forward_search(sm_I2)

sm_I1 = Subset_minimal(sm.nn[3], sm.nn[1:2](sm.input), sm.output)
forward_search_I1 = make_forward_search(sm_I1)


function full_forward_search(sm::Subset_minimal, threshold=0.9, num_samples=100)
    I3 = forward_search_I3(calc_ep; max_steps=50, threshold=0.5, num_samples=num_samples)
    I2 = forward_search_I2(calc_ep; max_steps=50, threshold=0.5, num_samples=num_samples)
    I1 = forward_search_I1(calc_ep; max_steps=50, threshold=0.5, num_samples=num_samples)
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