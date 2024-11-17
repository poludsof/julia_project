

function generate_random_img_with_fix_inputs(fix_inputs::SBitSet{N,T}, set_size::Integer) where {N, T}
    return map(idx -> idx in fix_inputs ? input[idx] : rand(0:1), 1:length(input))
end

function random_sampling(fix_inputs::SBitSet{N,T}, num_sets::Int) where {N, T}
    unique_sets = Set{Vector{Int}}()
    set_size = 784
    
    while length(unique_sets) < num_sets
        new_set = generate_random_img_with_fix_inputs(fix_inputs, set_size)
        if !(new_set in unique_sets)
            push!(unique_sets, new_set)
        end
    end
    
    return collect(unique_sets)
end


function calculate_sdp(ss::Subset_minimal, fix_inputs::SBitSet{N,T}) where {N, T}
    num_sets = 1000
    sampling_sets = random_sampling(fix_inputs, num_sets)

    correct_classified = 0
    for s in sampling_sets
        if argmax(ss.nn(s)) - 1 == ss.output
            correct_classified += 1
        end
    end
    
    return correct_classified / num_sets
end