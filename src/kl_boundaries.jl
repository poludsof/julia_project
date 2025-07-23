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

function compute_beta(n_features::Int, t::Real, delta::Real)
    alpha = 1.1
    k = 405.5
    temp = log(k * n_features * (t^alpha) / delta)
    return temp + log(temp)
end

function fill_missing_samples(sample_fn, n_samples, positives, means, anchor_sets::Vector)
        for i in eachindex(n_samples)
        if n_samples[i] == 0
            means[i] = sample_fn(anchor_sets[i])
            n_samples[i] += 100
            positives[i] = means[i] * n_samples[i]
        end
    end
end

function greedy_lucb(sm, ii::SBitSet, sampler, sample_fn, delta, epsilon; num_samples=1000)
    positives, n_samples = batch_heuristic(ii, sm, sampler, num_samples)
    anchor_sets = PAE.new_subsets_fwrd(ii, sm.dims)
    means = positives ./ n_samples
    fill_missing_samples(sample_fn, n_samples, positives, means, anchor_sets)

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
    # println("Initial bounds: ut: $(ut) with ub: $(ub[ut]), lt: $(lt) with lb: $(lb[lt])")
    while ub[ut] - lb[lt] > epsilon
        # println("while ", ub[ut] - lb[lt], " > ", epsilon)
    
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
        # println("Updated bounds: ut: $(ut) with ub: $(ub[ut]), lt: $(lt) with lb: $(lb[lt])")
    end
    # println("Final bounds: ut: $(ut) with ub: $(ub[ut]), lt: $(lt) with lb: $(lb[lt])")
    # println("returning: ", sets[ut], " with mean: ", means[ut], " and ud:", ub[ut])
    return means[ut], sets[ut]
end


function lucb_forward_search(sm, ii::SBitSet, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true, num_samples=1000)
    println("forward_search_lucb")
    initial_heuristic = heuristic_fun(ii)

    if typeof(initial_heuristic) == Tuple
        initial_h, initial_hmax_err = initial_heuristic
    elseif typeof(initial_heuristic) == Number 
        initial_h = initial_heuristic
    else
        initial_h = sum(initial_heuristic)
    end
    println("Initial h: ", initial_h)

    stack = [(initial_h, ii)]
    closed_list = Set{SBitSet}()
    solutions = Set{SBitSet}()

    steps = 0
    start_time = time()
    smallest_solution = typemax(Int)

    while !isempty(stack)

        steps += 1
        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return steps, solutions
        end

        sort!(stack, by = x -> -x[1])
        current_h, ii = pop!(stack)
        closed_list = push!(closed_list, ii)
        
        v = @timeit to "isvalid" isvalid(ii)
        # println("isvalid: ", v)

        if v
            println("Valid subset found...")
            push!(solutions, ii)
            if terminate_on_first_solution
                println("Terminating on first solution")
                return solutions
            end
            continue
        end

        println("step: $steps, length $(solution_length(ii)) with mean: ", current_h)
        push!(stack, greedy_lucb(sm, ii, heuristic_fun, 0.9, 0.1, num_samples=num_samples))
    end

    println("Stack is empty")
    return solutions
end


sampler = UniformDistribution()
img_i = 1
xₛ = train_X_bin_neg[:, img_i]
yₛ = argmax(model(xₛ))
II = init_sbitset(length(xₛ))
sm = Subset_minimal(model, xₛ, yₛ)
greedy_lucb(sm, II, sampler, CriteriumSdp(sm, sampler, 100, false), 0.9, 0.1, num_samples=1000)
# lucb_forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 100), CriteriumSdp(sm, sampler, 1000, false); time_limit=Inf, terminate_on_first_solution=true)        
# c, n = batch_heuristic(II, sm, sampler, 1000)
# println("Batch heuristic: ", length(c), " with set: ", length(n))

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