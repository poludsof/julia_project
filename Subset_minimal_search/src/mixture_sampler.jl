using StatsBase

"""
struct BernoulliMixture{T<:Real} <: AbstractSampler
    log_p::Array{T, 3}
    p::Matrix{T}
end

Implements BernoulliMixture with uniform weights.
`log_p` holds logarithms (`logsoftmax`) of probabilities 
(the third dimension is probability of one and zero, i.e. `size(log_p,3)=2`),
which simplifies the conditioning, which is crucial operation.

The conditioning is
"""
struct BernoulliMixture{T<:Real}
    log_p::Array{T, 3}
    p::Matrix{T}
end

function BernoulliMixture(centers)
    log_p = logsoftmax(centers, dims = 1)
    p = softmax(centers, dims = 1)[2,:,:]
    BernoulliMixture(log_p, p)
end

struct ConditionedBernoulliMixture{T<:Real,B,W}
    r::BernoulliMixture{T}
    mask::Vector{Bool}
    free2orig::Vector{Int64}    # maps the free indices to their location in original
    w::W
    xₛ::Vector{B}
end


function condition(r::BernoulliMixture, xₛ, known_ii::SBitSet)
    idim = length(xₛ)
    mask = fill(false, idim)
    for i in known_ii
        mask[i] = true
    end
    condition(r, [xᵢ >0 for xᵢ in xs], mask)
end

function condition(r::BernoulliMixture, xₛ, mask::Vector{Bool})
    free2orig = collect(1:length(mask))[.!mask]
    _xₛ = vcat(1 .- xₛ', xₛ')
    pzx = softmax(vec(sum(mask' .* _xₛ .* r.log_p, dims = (1,2))))
    w = StatsBase.Weights(pzx)
    ConditionedBernoulliMixture(r, mask, free2orig, w, xₛ)
end

"""
    sample_all(r::AbstractSampler, n::Integer)

    Sample `n` samples including fixed (condition) part, which is 
    copied
"""
function sample_all(r::ConditionedBernoulliMixture, n::Integer)
    u = zeros(length(r.xₛ), n)
    sample_all!(u, r)
end

"""
    sample_all!(u, r::AbstractSampler)

    Sample `size(u,2)` samples including fixed (condition) part, which is 
    copied
"""
function sample_all!(u, r::ConditionedBernoulliMixture)
    mask = r.mask
    p = r.r.p
    size(u,1) != length(mask) && error("dimension of u does not match the dimension of the sampler")
    for col in axes(u, 2)
        cid = sample(r.w)
        for row in axes(u, 1)
            u[row, col] = mask[row] ? r.xₛ[row] : (rand() < p[row, cid])
        end
    end
    u
end

"""
    sample_free!(u, r::AbstractSampler)

    Sample `size(u,2)` samples without fixed (condition) part.
"""
function sample_free!(u::Matrix, r::ConditionedBernoulliMixture)
    mask = r.mask
    size(u,1) != length(mask) - sum(mask) && error("dimension of u does not match the dimension of the sampler")
    for col in axes(u, 2)
        cid = sample(r.w)   # may-be, we can make this faster by reordering centers, since the weight will be cumulated on top
        u_row = 1
        for row in 1:length(r.mask)
            r.mask[row] && continue
            u[u_row, col] = (rand() < r.r.p[row, cid])
            u_row += 1
        end
    end
    u
end

function sample_free!(u::PackedMatrix{N,I}, r::ConditionedBernoulliMixture) where {N,I}
    mask = r.mask
    D = u.nrows
    D != length(mask) - sum(mask) && error("Dimension of `u` does not match the dimension of the sampler. `u` has $(D) rows while sampler has dimension $(length(r.free2orig))")
    bits = 8*sizeof(I)
    @inbounds for col in axes(u.x, 2)
        cid = sample(r.w)
        for j in 1:N
            y = zero(I)
            upper_bound = (j*bits > D) ? D : j*bits 
            for b in upper_bound:-1:bits*(j-1) + 1
                y <<= 1
                y |= (rand() < r.r.p[r.free2orig[b], cid])
            end
            u.x[j,col] = y
        end
    end
    u
end