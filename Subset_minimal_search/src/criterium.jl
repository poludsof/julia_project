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


function make_calculate_ep(sm::Subset_minimal)
    return function(fix_inputs::SBitSet, num_samples::Int)
        x = sample_input(sm.input, fix_inputs, num_samples)
        mean(Flux.softmax(sm.nn(x))[sm.output + 1, :])
    end
end

function make_calculate_sdp(sm::Subset_minimal)
    return function(fix_inputs::SBitSet, num_samples::Int)
        x = sample_input(sm.input, fix_inputs, num_samples)
        mean(Flux.onecold(sm.nn(x)) .== sm.output + 1)
    end
end

# function calculate_ep(sm::Subset_minimal, fix_inputs::SBitSet, num_samples::Int)
#     x = sample_input(sm.input, fix_inputs, num_samples)
#     mean(Flux.softmax(sm.nn(x))[sm.output + 1,:] )
# end

# function calculate_sdp(sm::Subset_minimal, fix_inputs::SBitSet{N,T}, num_samples::Int) where {N, T}
#     x = sample_input(sm.input, fix_inputs, num_samples)
#     mean(Flux.onecold(sm.nn(x)) .== sm.output + 1)
# end


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