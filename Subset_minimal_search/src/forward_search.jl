
# """ Simple method that adds features until a set with no adversarial samples is found. Without using MILP. """
function find_minimal_subset_forward(sm::Subset_minimal, given_input_set::SBitSet{N,T}) where {N, T}
    candidate_subset = SBitSet{N, T}()

    for i in sm.input
        candidate_subset = candidate_subset ∪ SBitSet{N, T}(i)

        # adv_img_founded = check_random_sets(sm, candidate_subset)
        # if !adv_img_founded
        #     return candidate_subset
        # end

        threshold = calc_func(sm, candidate_subset, 100)
        if threshold >= 0.1
            return candidate_subset
        end

        # println("Candidate subset: ", length(candidate_subset), " status: ", status)
    end
    
    return SBitSet{N, T}()
end

function find_minimal_subset_forward(sm::Subset_minimal, given_input_set::SBitSet)
    sufficient_subsets = []  # list to store 10 minimal subsets
    feature_list = collect(given_input_set)

    while length(sufficient_subsets) < 10
        current_subset = SBitSet{N, T}()
        
        for feature in feature_list
            push!(current_subset, feature)
            
            if !SDP(sm, current_subset)
                # Remove feature if not sufficient
                delete!(current_subset, feature)
            end
        end
        
        # Save the minimal sufficient subset
        push!(sufficient_subsets, current_subset)
        
        # Remove selected features from consideration
        feature_list = setdiff(feature_list, collect(current_subset))
    end

    return sufficient_subsets
end
