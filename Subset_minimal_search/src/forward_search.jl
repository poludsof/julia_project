
""" Simple method that adds features until a set with no adversarial samples is found. Without using MILP. """
function find_minimal_subset_forward(nn::Chain, given_input_set::SBitSet{N,T}, image::AbstractVector{<:Integer}, output::Int) where {N, T}
    candidate_subset = SBitSet{N, T}()

    for i in given_input_set
        candidate_subset = candidate_subset ∪ SBitSet{N, T}(i)

        adv_img_founded = check_random_sets(nn, image, output, candidate_subset)
        if !adv_img_founded
            return candidate_subset
        end

        # println("Candidate subset: ", length(candidate_subset), " status: ", status)
    end
    
    return SBitSet{N, T}()
end
