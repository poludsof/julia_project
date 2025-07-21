#KL-LUCB

function kl_bernoulli(p, q)
    p = clamp(p, 1e-7, 1 - 1e-16)
    q = clamp(q, 1e-7, 1 - 1e-16)
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

# binary search
function dup_bernoulli(p, level)
    #upper bound +- [p, 1]
    lm = p
    um = min(1.0, p + sqrt(level / 2))
    for _ in 1:16
        qm = (um + lm) / 2
        if kl_bernoulli(p, qm) > level
            um = qm
        else
            lm = qm
        end
    end
    return um
end

function dlow_bernoulli(p, level)
    #lower bound +- [0, p]
    lm = max(0.0, p - sqrt(level / 2))
    um = p
    for _ in 1:16
        qm = (um + lm) / 2
        if kl_bernoulli(p, qm) > level
            lm = qm
        else
            um = qm
        end
    end
    return lm
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

#! attempt to calculate the boundaries
