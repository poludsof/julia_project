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



@testset "Bernoulli bounds test" begin
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