@inline function sample_input(img::AbstractVector, ii::SBitSet, num_samples::Integer)
    ii = collect(ii)
    x = rand(eltype(img), length(img), num_samples)
    map!(x -> 2*round(x) - 1, x, x)
    x[ii,:] .= img[ii]
    x
end

function ep_score(logits, y::Integer)
    logŷ = softmax(logits)
    mean(logŷ[y,:])
end

function sdp_score(logits, y::Integer)
    ŷ =  Flux.onecold(logits)
    sum(==(y), ŷ) / length(ŷ)
end

""" SDP and EP criteria for evaluating subset(ii) robustness """

function criterium_ep(model, img, y, ii, data_model, num_samples)
    x = uniform_distribution(img, ii, num_samples)
    ep_score(model(x), y)
end

criterium_ep(model, img, y, ii::Nothing, data_model, num_samples) = 1

function criterium_sdp(model, img, y, ii, data_model::Nothing, num_samples)
    x = uniform_distribution(img, ii, num_samples)
    sdp_score(model(x), y)
end


criterium_sdp(model, img, y, ii::Nothing, data_model::Nothing, num_samples) = 1

function criterium_sdp(model, img, y, ii, data_model, num_samples)
    x = data_distribution(img, ii, data_model, num_samples)
    sdp_score(model(x), y)
end

criterium_sdp(model, img, y, ii::Nothing, data_model, num_samples) = 1

""" Partial SDP and EP criteria for evaluating the implication score of a subset of one layer(ii) on another layer(jj) """
function sdp_partial(model, img, y, ii, jj, data_model, num_samples)
    (jj === nothing || ii === nothing) && return 1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = uniform_distribution(img, ii, num_samples)
    sdp_score(model(x), y)
end

sdp_partial(model, img, y, ii, jj::Nothing, data_model, num_samples) = 1
sdp_partial(model, img, y, ii::Nothing, jj, data_model, num_samples) = 1
sdp_partial(model, img, y, ii::Nothing, jj::Nothing, data_model, num_samples) = 1

# TODO ep_partial doesn't work
function ep_partial(model, img, y, ii, jj, data_model, num_samples)
    (jj === nothing || ii === nothing) && return 1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = uniform_distribution(img, ii, num_samples)
    ep_score(model(x), y)
end
