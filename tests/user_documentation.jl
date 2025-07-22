#### 1. Setup
# run debug.jl

#### 2 Set parameters:

### Sampling distribution (Unoform or Bernoulli)
sampler = UniformDistribution()
## or
# sampler_path = raw"Subset_minimal_search/models/milan_centers.jls"
# sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))

### Set image to explain
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(model(xₛ))

### Initialize rule
sm = SMS.Subset_minimal(to_gpu(model), xₛ, yₛ)
II = init_sbitset(length(xₛ))
num_samples = 1000
ϵ = 0.1
accuracy = 1 - ϵ
solution_subsets = []


#### 3. Input (first layer) explanation

### 3.1 Run search
## Forward search
time = @elapsed solution_subsets = forward_search(
            sm, II, ii -> isvalid_sdp(ii, sm, accuracy, sampler, num_samples),
            ShapleyHeuristic(sm, sampler, num_samples),
            time_limit=100,
            terminate_on_first_solution=true,
            refine_with_backward=true
        )
# Print first solution
print_solution(solution_subsets)

## Backward search
II = init_full_sbitset(xₛ) # initialize with all features for backward search
# time = @elapsed solution_subsets = backward_search(
#             sm, II, ii -> isvalid_sdp(ii, sm, accuracy, sampler, num_samples),
#             ShapleyHeuristic(sm, sampler, num_samples);
#             time_limit=30,
#             terminate_on_first_solution=false,
#         )
## Print first solution
print_solution(solution_subsets)

 
## Beam search
beam = 5 # number of subsets to keep in each step
# time = @elapsed solution_subsets = beam_search(
#             sm, II, ii -> isvalid_sdp(ii, sm, accuracy, sampler, num_samples),
#             ShapleyHeuristic(sm, sampler, num_samples),
#             time_limit=300, 
#             beam_size=beam, 
#             terminate_on_first_solution=true    
#         )
## Print first solution
print_solution(solution_subsets)


#! ADD precision and coverage calculation

#### 4. Network (all layers) explanation

### 4.1 Run search









### Helper functions
function print_solution(solution_subsets)
    if solution_subsets !== nothing && length(solution_subsets) > 0
        solution_list = collect(solution_subsets)
        best_subset = solution_list[1]
        subset_size = length(best_subset) > 0 ? length(best_subset) : 0
        println("Subset size: $subset_size, features set: $best_subset")
    else
        println("No valid subset found.")
    end
end