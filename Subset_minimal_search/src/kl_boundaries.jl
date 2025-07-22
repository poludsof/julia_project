#KL-LUCB
using Test

function kl_bernoulli(p, q)
    p = clamp(p, 1e-7, 1 - 1e-16)
    q = clamp(q, 1e-7, 1 - 1e-16)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

# kl_bernoulli(p, q) = p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))


# binary search
    
#upper bound +- [p, 1]
function dup_bernoulli(p, confidence; max_iter=16)
    lower = p
    upper = min(1.0, p + sqrt(confidence / 2))

    for _ in 1:max_iter
        mid = (lower + upper) / 2
        if kl_bernoulli(p, mid) > confidence
            upper = mid
        else
            lower = mid
        end
    end

    return upper
end

#lower bound +- [0, p]
function dlow_bernoulli(p, confidence; max_iter=16)
    lower = max(0.0, p - sqrt(confidence / 2))
    upper = p

    for _ in 1:max_iter
        mid = (lower + upper) / 2
        if kl_bernoulli(p, mid) > confidence
            lower = mid
        else
            upper = mid
        end
    end

    return lower
end

function update_bounds(t, means, n_samples, n_features, delta, top_n, ub, lb)
    sorted_means = sortperm(means)
    beta = compute_beta(n_features, t, delta)
    J = sorted_means[end-top_n+1:end]
    not_J = sorted_means[1:end-top_n]

    for f in not_J
        ub[f] = dup_bernoulli(means[f], beta / n_samples[f])
    end
    for f in J
        lb[f] = dlow_bernoulli(means[f], beta / n_samples[f])
    end

    ut = not_J[argmax(ub[not_J])]
    lt = J[argmin(lb[J])]
    return ut, lt
end

function compute_beta(n_features::Int, t::Real, delta::Real)
    alpha = 1.1
    k = 405.5
    temp = log(k * n_features * (t^alpha) / delta)
    return temp + log(temp)
end

function greedy_lucb(sm, ii::SBitSet, sample_fn, delta, epsilon; num_samples=1000)
    anchors = expand_frwd(sm, [], [], ii, sample_fn)
    println("size of anchors: ", typeof(anchors), " first:" , anchors[1])
    means, sets = map(x -> x[1], anchors), map(x -> x[2], anchors)
    println("Means: ", means[1])
    println("Sets: ", sets[1])

    n_samples = fill(num_samples, length(anchors))  ###! modify
    ub = zeros(length(anchors))
    lb = zeros(length(anchors))
    top_n = 2 # number of top features to consider
    t = 1
    n_features = length(anchors)
    
    function update_bounds(t::Integer)
        inds = sortperm(anchors, by = x -> x[1])
        
        beta = compute_beta(n_features, t, delta)
        
        # split indexes into top_n best and others
        J = inds[end - top_n + 1:end]
        not_J = inds[1:end - top_n]
        
        for f in not_J
            ub[f] = dup_bernoulli(anchors[f][1], beta / n_samples[f])
        end
        for f in J
            lb[f] = dlow_bernoulli(anchors[f][1], beta / n_samples[f])
        end
        
        ut = not_J[argmax(ub[not_J])]
        lt = J[argmin(lb[J])]
        
        return ut, lt
    end
    
    #calculate bounds
    ut, lt = update_bounds(t)
    println("Initial bounds: ut: $(ut) with ub: $(ub[ut]), lt: $(lt) with lb: $(lb[lt])")
    while ub[ut] - lb[lt] > epsilon
        println("while ", ub[ut] - lb[lt], " > ", epsilon)
    
        # println("means before: ", means[lt])
        new_mean_ut = sample_fn(sets[ut])
        means[ut] = (n_samples[ut] * means[ut] + num_samples * new_mean_ut) / (n_samples[ut] + num_samples)
        n_samples[ut] += num_samples

        new_mean_lt = sample_fn(sets[lt])
        means[lt] = (n_samples[lt] * means[lt] + num_samples * new_mean_lt) / (n_samples[lt] + num_samples)
        n_samples[lt] += num_samples
        # println("means new: ", new_mean_lt)
        # println("means after: ", means[lt])

        t += 1
        ut, lt = update_bounds(t)
        println("Updated bounds: ut: $(ut) with ub: $(ub[ut]), lt: $(lt) with lb: $(lb[lt])")
        # if t == 30
        #     println("Stopping after 30 iterations")
        #     return
        # end
    end
    println("Final bounds: ut: $(ut) with ub: $(ub[ut]), lt: $(lt) with lb: $(lb[lt])")
    
end

img_i = 1
xₛ = train_X_bin_neg[:, img_i]
yₛ = argmax(model(xₛ))
II = init_sbitset(length(xₛ))
sm = SMS.Subset_minimal(model, xₛ, yₛ)
greedy_lucb(sm, II, CriteriumSdp(sm, sampler, 1000, false), 0.5, 0.1, num_samples=1000)


function bernoulli_bounds_test()
    @testset "bernoulli bounds test" begin
        println("testing KL bounds")
        for p in 0.1:0.1:0.9
            confidence = 0.9
            lower = dlow_bernoulli(p, confidence)
            upper = dup_bernoulli(p, confidence)
            println("p: $p, lower: $lower, upper: $upper, confidence: $confidence")
            
            # p must be within the interval
            @test lower ≤ p ≤ upper
            
            println("KL lower bound: ", kl_bernoulli(p, lower))
            println("KL upper bound: ", kl_bernoulli(p, upper))
        end
    end
end

# bernoulli_bounds_test()