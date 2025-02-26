@inline function sample_input(img::AbstractVector, ii::SBitSet, num_samples::Integer)
    ii = collect(ii)
    x = similar(img, length(img), num_samples)
    @inbounds for col in axes(x, 2)
        for i in axes(x, 1)
            x[i, col] = 2*rand(Bool) - 1
        end
        for i in ii
            x[i, col] = img[i]
        end
    end
    x
end


function criterium_ep(sm::Subset_minimal, fix_inputs::SBitSet, num_samples::Int, τ)
    x = sample_input(sm.input, fix_inputs, num_samples)
    mean(Flux.softmax(sm.nn(x))[sm.output, :]) > τ
end

function criterium_sdp(sm::Subset_minimal, fix_inputs::SBitSet, num_samples::Int, τ)
    x = sample_input(sm.input, fix_inputs, num_samples)
    mean(Flux.onecold(sm.nn(x)) .== sm.output) > τ
end

function sdp_partial(model, img, ii, jj, num_samples)
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = sample_input(img, ii, num_samples)
    mean(model(x)[jj, :] .== model(img)[jj, :])
end