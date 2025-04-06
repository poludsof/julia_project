# srun -p gpufast --gres=gpu:1  --mem=16000  --pty bash -i
# cd julia/Pkg/Subset_minimal_search/tests/
#  ~/.juliaup/bin/julia --project=.
using Revise
using ProfileCanvas, BenchmarkTools
using CUDA
using Subset_minimal_search
import Subset_minimal_search as SMS
using Subset_minimal_search.Flux
using Subset_minimal_search.LinearAlgebra
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.TimerOutputs
using Serialization
using Subset_minimal_search.Makie
const to = Subset_minimal_search.to
# using Subset_minimal_search.Makie.Colors
# using Subset_minimal_search.Serialization
# using Subset_minimal_search.DataStructures
# using Subset_minimal_search.Distributions
# using Subset_minimal_search.Serialization

to_gpu = gpu
# to_gpu = cpu

""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
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

train_y = SMS.onehot_labels(train_y)
test_y = SMS.onehot_labels(test_y)

""" Prepare image and label """
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(train_y[:, 1]) - 1
sm = SMS.Subset_minimal(model, xₛ, yₛ)

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

#test
# ii = init_sbitset(784)
# data_matrix = SMS.data_distribution(xₛ, ii, r, 100)

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
    acc = SMS.criterium_sdp(sm.nn, sm.input, sm.output, ii[1], sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_sdp(ii::Tuple, sm, ϵ, sampler, num_samples, verbose = false)
    h = SMS.heuristic(sm, SMS.criterium_sdp, SMS.sdp_partial, ii, ϵ, sampler, num_samples)
    verbose && println("heuristic = ",h)
    h
end


# For tuple, we need to define appropriate heuristics which understand tuples

function isvalid_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    acc = SMS.criterium_sdp(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    h = SMS.criterium_sdp(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    h = ϵ - h
    verbose && println("heuristic = ",h)
    (;hsum = h, hmax = h)
end

function isvalid_ep(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    acc = SMS.criterium_ep(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_ep(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    h = SMS.criterium_ep(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    h = ϵ - h
    verbose && println("heuristic = ",h)
    (;hsum = h, hmax = h)
end

function shapley_heuristic(ii::SBitSet, sm, sampler, num_samples, verbose = false)
    r = SMS.condition(sampler, sm.input, ii)
    x = SMS.sample_all(r, num_samples)
    y = Flux.onecold(sm.nn(x)) .== sm.output

    # For us, the DAF statistic is a difference in errors when the feature is correct and wrong
    n = sum(x .== xₛ, dims = 2)
    # n₊ = sum(x .== 1, dims = 2)
    # n₋ = size(x,2) .- n₊

    # compute, how many +1 are correct
    _f(xᵢ, xₛᵢ, yᵢ) = xᵢ == xₛᵢ ? yᵢ : false
    c = sum(_f.(x, xₛ, y'), dims = 2)

    cpu(c ./ max.(1, n))
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

(sp::ShapleyHeuristic)(ii) = heuristic_sdp(ii, sm, 0.99, sampler, 10000)

function Subset_minimal_search.expand_frwd(sm::Subset_minimal_search.Subset_minimal, stack, closed_list, ii::SBitSet, heuristic_fun::ShapleyHeuristic)
    sp = heuristic_fun
    acc = @timeit to "heuristic" shapley_heuristic(ii, sp.sm, sp.sampler, sp.num_samples, sp.verbose)
    for i in setdiff(1:sm.dims, ii)
        new_subset = push(ii, i)
        if new_subset ∉ closed_list
            new_error = 1 -acc[i]
            push!(stack, (new_error, new_error, new_subset))
        end
    end
    stack
end



ϵ = 0.99
#2. Search
""" Prepare image and label """
img_i = 1
xₛ = train_X_bin_neg[:, img_i] |> to_gpu
yₛ = argmax(model(xₛ))

# variant with triplets
# II = (init_sbitset(784), nothing, nothing)
# sm = SMS.Subset_minimal(model, xₛ, yₛ, (784, 256, 256))

# variant with just input
II = init_sbitset(length(xₛ))
sm = SMS.Subset_minimal(model, xₛ, yₛ)

t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10000),  ShapleyHeuristic(sm, sampler, 10000))
t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, 10000),  ShapleyHeuristic(sm, sampler, 10000))
t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_sdp(ii, sm, ϵ, sampler, 10000))
t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_ep(ii, sm, ϵ, sampler, 10000))


# verification checks
function test_samplers()    
    using Test
    xₛ = train_X_bin_neg[:, 1]

    ii = init_sbitset(784, 64)
    vii = collect(ii)
    cii = setdiff(1:length(xₛ), ii)

    sampler_gpu = BernoulliMixture(to_gpu(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls"))))
    sampler_cpu = BernoulliMixture(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls")))

    r_cpu = SMS.condition(sampler_cpu, cpu(xₛ), ii)
    r_gpu = SMS.condition(sampler_gpu, cu(xₛ), ii)

    for xx in [cpu(SMS.sample_all(r_cpu, 10_000)),cpu(SMS.sample_all(r_gpu, 10_000))]
        @test all(xx .!= 0)
        @test all(map(∈((-1,+1)), xx))
        @test all(xx[vii,:] .== xₛ[vii])
    end

    mean(xx[cii,:], dims =2 )
end