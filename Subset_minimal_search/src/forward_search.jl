
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
    end
    
    return SBitSet{N, T}()
end

function forward_search_dfs(sm::Subset_minimal, given_input_set::SBitSet{N,T}) where {N, T}

end