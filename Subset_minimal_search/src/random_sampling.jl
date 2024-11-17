

function generate_random_img_with_fix_inputs(sm::Subset_minimal, fix_inputs::SBitSet{N,T}) where {N, T}
    return map(idx -> idx in fix_inputs ? sm.input[idx] : rand(0:1), 1:length(sm.input))
end

function random_sampling(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_sets::Int) where {N, T}
    unique_sets = Set{Vector{Int}}()
    
    while length(unique_sets) < num_sets
        new_set = generate_random_img_with_fix_inputs(sm, fix_inputs)
        if !(new_set in unique_sets)
            push!(unique_sets, new_set)
        end
    end
    
    println("Number of unique sets: ", length(unique_sets))
    return collect(unique_sets)
end


function calculate_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}) where {N, T}
    num_sets = 2 ^ (length(sm.input) - length(fix_inputs))
    if num_sets < 1000 && num_sets > 0
        num_sets = num_sets
    else
        num_sets = 1000
    end
    println("Number of sets: ", num_sets)

    sampling_sets = random_sampling(sm, fix_inputs, num_sets)
    println("Number of sampling sets: ", length(sampling_sets))
    correct_classified = 0
    for s in sampling_sets
        if argmax(sm.nn(s)) - 1 == sm.output
            correct_classified += 1
        end
    end
    
    println("Correct classified: ", correct_classified)
    return correct_classified / num_sets
end