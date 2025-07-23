# srun -p gpufast --gres=gpu:1  --mem=16000  --pty bash -i
# cd julia/Pkg/Subset_minimal_search/tests/
#  ~/.juliaup/bn/julia --project=.
using Revise
# using ProfileCanvas, BenchmarkTools
using CUDA
using ProbAbEx
import ProbAbEx as PAE
using ProbAbEx.Flux
using ProbAbEx.LinearAlgebra
using ProbAbEx.MLDatasets
using ProbAbEx.StaticBitSets
using ProbAbEx.TimerOutputs
using Serialization
using ProbAbEx.Makie
const to = ProbAbEx.to


# CUDA.has_cuda()
# CUDA.device()

# to_gpu = gpu
to_gpu = cpu

""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model_path = "models/binary_model.jls"
model = deserialize(model_path) |> to_gpu;

""" nn for MILP search """
# nn = Chain
# (Dense(28^2, 28, relu), Dense(28,28, relu), Dense(28,10)) 
# nn = train_nn(nn, train_X_bin_neg, train_y, test_X_bin_neg, test_y)


""" Prepare data """
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)

""" Prepare image and label """
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(train_y[:, 1])
sm = Subset_minimal(model, xₛ, yₛ)

function init_sbitset(n::Int, k = 0) 
    N = ceil(Int, n / 64)
    x = SBitSet{N, UInt64}()
    k == 0 && return(x)
    for i in rand(1:n, k)
        x = push(x, i)
    end
    x
end

sampler = UniformDistribution()
# sampler = BernoulliMixture(to_gpu(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls"))))
# sampler_path = "Subset_minimal_search/models/milan_centers.jls"
# sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))
#test
# ii = init_sbitset(784)
# data_matrix = data_distribution(xₛ, ii, r, 100)

#first image
# println(data_matrix[:, 1])
#change 0 to -1 and 1 to 1
# println(2*data_matrix[:, 1] .- 1)
# plot_mnist_image(2*data_matrix[:, 20] .- 1)


""" Test conditioning """

""" Test MILP search """


# """ Test one subset(layer) backward/forward/beam search """ -- too slow
# threshold denotes the required precision of the subset
# solution_subset = one_subset_forward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=10, terminate_on_first_solution=false)
# solution_subset = one_subset_backward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=100, time_limit=60)
# solution_beam_subsets = one_subset_beam_search(sm, criterium_ep; data_model=r, threshold=0.5, beam_size=5, num_samples=100, time_limit=60)


""" Test search functions that fit one and all subsets"""
# Threshold denotes allowed error of the subset
# Valid_criterium is needed to distinguish with respect to what the validity of a subset is being checked, because it can be different, e.g. as with MILP 

reset_timer!(to)

#1. Initialization
# II = (init_sbitset(784), nothing, nothing)
# For tuple, we need to define appropriate heuristics which understand tuples
function isvalid_sdp(ii::Tuple, sm, ϵ, sampler, num_samples, verbose = false)
    acc = criterium_sdp(sm.nn, sm.input, sm.output, ii[1], sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_sdp(ii::Tuple, sm, ϵ, sampler, num_samples, verbose = false)
    h = heuristic(sm, criterium_sdp, sdp_partial, ii, ϵ, sampler, num_samples)
    verbose && println("heuristic = ",h)
    h
end


# For tuple, we need to define appropriate heuristics which understand tuples

function isvalid_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    acc = criterium_sdp(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    h = criterium_sdp(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    h = ϵ - h
    verbose && println("heuristic = ",h)
    (;hsum = h, hmax = h)
end

function isvalid_ep(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    acc = criterium_ep(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_ep(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    h = criterium_ep(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    h = ϵ - h
    verbose && println("heuristic = ",h)
    (;hsum = h, hmax = h)
end

function shapley_heuristic(ii::SBitSet, sm, sampler, num_samples, verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    # y = Flux.onecold(sm.nn(x)) .== sm.output
    y = argmax(sm.output)
    scores = sm.nn(x)
    if size(scores, 1) == 0 || size(scores, 2) == 0
        error("Empty prediction scores — cannot compute onecold.")
    end
    ŷ = Flux.onecold(scores)
    sum(==(y), ŷ) / length(ŷ)
end

function shapley_heuristic(ii::Tuple, sm, sampler, num_samples; verbose = false)
    (I3, I2, I1) = ii
    xₛ = sm.input
    # println("shapley_heuristic")
    h₃ = shapley_heuristic(I3, Subset_minimal(sm.nn, xₛ), sampler, num_samples)
    h₂ = shapley_heuristic(I2, Subset_minimal(sm.nn[2:3], sm.nn[1](xₛ)), sampler, num_samples)
    h₁ = shapley_heuristic(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), sampler, num_samples)

    h₃_h₂ = 0.0
    h₂_h₁ = 0.0

    if !isempty(I2)
        h₃_h₂ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), samplers[1], num_samples)
    end 
    if !isempty(I1)
        h₂_h₁ = shapley_heuristic(I2, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), samplers[2], num_samples)
    end    
    h₁_h₀ = shapley_heuristic(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), samplers[3], num_samples)

        # the input to the neural network has to imply h₃[I2] and h₂[I1] 
        # h₃_h₂ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), sampler, num_samples),
        # h₃_h₁ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), sampler, num_samples),
        # h₂_h₁ = shapley_heuristic(I2, (Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), I1), sampler, num_samples),
    #
    hs = (
        h₃ = h₃,
        h₂ = h₂,
        h₁ = h₁,
        h₃_h₂ = h₃_h₂,
        h₂_h₁ = h₂_h₁,
        h₁_h₀ = h₁_h₀,
    )
    (;
    hsum = mapreduce(x -> max(0, 0.99 - x), +, hs),
    hmax = mapreduce(x -> max(0, 0.99 - x), max, hs),
    )
end

struct ShapleyHeuristic{S,SM}
    sm::SM
    sampler::S
    num_samples::Int
    verbose::Bool
end

function ShapleyHeuristic(sm, sampler, num_samples, verbose = false)
    ShapleyHeuristic(sampler, sm, num_samples, verbose)
end

(sp::ShapleyHeuristic)(ii::SBitSet) = heuristic_sdp(ii, sm, 0.99, sampler, 10000)
(sp::ShapleyHeuristic)(ii::Tuple) = shapley_heuristic(ii, sm, sampler, 10000)

# function Subset_minimal_search.expand_frwd(sm::Subset_minimal_search.Subset_minimal, stack, closed_list, ii::SBitSet, heuristic_fun::ShapleyHeuristic)
#     sp = heuristic_fun
#     acc = @timeit to "heuristic" shapley_heuristic(ii, sp.sm, sp.sampler, sp.num_samples, sp.verbose)
#     for i in setdiff(1:sm.dims, ii)
#         new_subset = push(ii, i)
#         if new_subset ∉ closed_list
#             new_error = 1 -acc[i]
#             push!(stack, (new_error, new_error, new_subset))
#         end
#     end
#     stack
# end

#  serialize("/home/pevnytom/tmp/subsets.jls", (;solutions = collect(solution_subsets), x = Vector(xₛ)))

samplers = (UniformDistribution(), UniformDistribution(), UniformDistribution())
# samplers = (BernoulliMixture(centers[:,1:784,:]), BernoulliMixture(centers[:,785:1040,:]), BernoulliMixture(centers[:,1041:end,:]))
ϵ = 0.9
#2. Search
""" Prepare image and label """
img_i = 1
xₛ = train_X_bin_neg[:, img_i] |> to_gpu
yₛ = argmax(model(xₛ))

# variant with triplets
# II = (init_sbitset(784), nothing, nothing)
# sm = Subset_minimal(model, xₛ, yₛ, (784, 256, 256))

# variant with just input
# II = init_sbitset(length(xₛ))
# sm = Subset_minimal(model, xₛ, yₛ)
sm = Subset_minimal(model, xₛ, yₛ, (784, 256, 256))
II = (init_sbitset(784), init_sbitset(256), init_sbitset(256))

# t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 100),  ShapleyHeuristic(sm, sampler, 100), refine_with_backward = false)
# t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_sdp(ii, sm, ϵ, sampler, 10000))
# t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_ep(ii, sm, ϵ, sampler, 10000))

# CUDA.@time forward_search(sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_ep(ii, sm, ϵ, sampler, 10000), refine_with_backward = false)

# solutions = forward_search(sm, II, ii -> isvalid_sdp3(II, sm, ϵ, samplers, 1000),  ShapleyHeuristic(sm, samplers, 1000); terminate_on_first_solution=true, refine_with_backward = false)


# verification checks
function test_samplers()    
    # using Test
    xₛ = train_X_bin_neg[:, 1]

    ii = init_sbitset(784, 64)
    vii = collect(ii)
    cii = setdiff(1:length(xₛ), ii)

    sampler_gpu = BernoulliMixture(to_gpu(deserialize(joinpath("Subset_minimal_search", "models", "milan_centers.jls"))))
    sampler_cpu = BernoulliMixture(deserialize(joinpath("Subset_minimal_search", "models", "milan_centers.jls")))

    r_cpu = condition(sampler_cpu, cpu(xₛ), ii)
    r_gpu = condition(sampler_gpu, cu(xₛ), ii)

    for xx in [cpu(sample_all(r_cpu, 10_000)),cpu(sample_all(r_gpu, 10_000))]
        all(xx .!= 0)
        all(map(∈((-1,+1)), xx))
        all(xx[vii,:] .== xₛ[vii])
    end

    mean(xx[cii,:], dims =2 )
end

# test_samplers()

# CUDA.reclaim()