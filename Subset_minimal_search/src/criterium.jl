
function compute_sdp_fwd(model, img, ii, num_samples)
    ii = collect(ii)
    x = rand([-1,1], length(img), num_samples)
    x[ii,:] .= img[ii]
    mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

function calculate_ep(sm::Subset_minimal, fix_inputs::SBitSet, num_samples::Int)
    x = generate_random_img_with_fix_inputs(sm, fix_inputs, num_samples)
    mean(Flux.softmax(sm.nn(x))[sm.output + 1,:] )
end

function calculate_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_samples::Int) where {N, T}
    x = generate_random_img_with_fix_inputs(sm, fix_inputs, num_samples)
    mean(Flux.onecold(nn(x)) .== sm.output + 1)
end

function sdp_full(model, img, ii, num_samples)
    x = sample_input(img, ii, num_samples)
    mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

function sdp_partial(model, img, ii, jj, num_samples)
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = sample_input(img, ii, num_samples)
    mean(model(x)[jj, :] .== model(img)[jj, :])
end