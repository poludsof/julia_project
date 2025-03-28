@inline function sample_input(img::AbstractVector, ii::SBitSet, num_samples::Integer)
    ii = collect(ii)
    x = rand(eltype(img), length(img), num_samples)
    map!(x -> 2*round(x) - 1, x, x)
    x[ii,:] .= img[ii]
    x
end



""" SDP and EP criteria for evaluating subset(ii) robustness """

function criterium_ep(model, img, ii, data_model, num_samples)
    x = uniform_distribution(img, ii, num_samples)
    mean(Flux.softmax(model(x))[argmax(model(img)), :])
end

criterium_ep(model, img, ii::Nothing, data_model, num_samples) = 1

function criterium_sdp(model, img, ii, data_model::Nothing, num_samples)
    x = uniform_distribution(img, ii, num_samples)
    mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

criterium_sdp(model, img, ii::Nothing, data_model::Nothing, num_samples) = 1

function criterium_sdp(model, img, ii, data_model, num_samples)
    x = data_distribution(img, ii, data_model, num_samples)
    mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

criterium_sdp(model, img, ii::Nothing, data_model, num_samples) = 1

""" Partial SDP and EP criteria for evaluating the implication score of a subset of one layer(ii) on another layer(jj) """
function sdp_partial(model, img, ii, jj, data_model, num_samples)
    (jj === nothing || ii === nothing) && return 1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = uniform_distribution(img, ii, num_samples)
    mean(model(x)[jj, :] .== model(img)[jj, :])
end

# TODO ep_partial doesn't work
function ep_partial(model, img, ii, jj, data_model, num_samples)
    (jj === nothing || ii === nothing) && return 1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = uniform_distribution(img, ii, num_samples)
    # mean(Flux.softmax(model(x))[jj, :] .== Flux.softmax(model(img))[jj, :])
end
