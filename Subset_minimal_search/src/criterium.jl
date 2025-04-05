function ep_score(logits, y::Integer)
    logŷ = softmax(logits)
    mean(logŷ[y,:])
end

function sdp_score(logits, y::Integer)
    ŷ =  Flux.onecold(logits)
    sum(==(y), ŷ) / length(ŷ)
end

# criterium_sdp(model, img, y, ii::Nothing, sampler::Nothing, num_samples) = 1

function criterium_sdp(model, xₛ, y, ii, sampler, num_samples)
    r = condition(sampler, xₛ, ii)
    xx = sample_all(r, num_samples)
    sdp_score(model(xx), y)
end

function criterium_ep(model, xₛ, y, ii, sampler, num_samples)
    r = condition(sampler, xₛ, ii)
    xx = sample_all(r, num_samples)
    ep_score(model(xx), y)
end

criterium_sdp(model, img, y, ii::Nothing, sampler, num_samples) = 1

""" Partial SDP and EP criteria for evaluating the implication score of a subset of one layer(ii) on another layer(jj) """
function sdp_partial(model, img, y, ii, jj, sampler, num_samples)
    (jj === nothing || ii === nothing) && return 1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = uniform_distribution(img, ii, num_samples)
    sdp_score(model(x), y)
end

sdp_partial(model, img, y, ii, jj::Nothing, sampler, num_samples) = 1
sdp_partial(model, img, y, ii::Nothing, jj, sampler, num_samples) = 1
sdp_partial(model, img, y, ii::Nothing, jj::Nothing, sampler, num_samples) = 1

# TODO ep_partial doesn't work
function ep_partial(model, img, y, ii, jj, sampler, num_samples)
    (jj === nothing || ii === nothing) && return 1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = uniform_distribution(img, ii, num_samples)
    ep_score(model(x), y)
end
