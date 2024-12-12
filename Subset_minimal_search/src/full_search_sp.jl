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

function sdp_full(model, img, ii, num_samples)
    # x = generate_random_img_with_fix_inputs(model, input, fix_inputs, num_samples)
    x = sample_input(img, ii, num_samples)
    mean(Flux.onecold(model(x)) .== Flux.onecold(model(img)))
end

function sdp_partial(model, img, ii, jj, num_samples) # ii = I2 --> jj = I1
    isempty(jj) && return(1.0)
    isempty(ii) && return(0.0)
    jj = collect(jj)
    x = sample_input(img, ii, num_samples)
    mean(model(x)[jj, :] .== model(img)[jj, :])
end


function init_sbitset(n::Int) 
    N = ceil(Int, n / 64)
    SBitSet{N, UInt64}()
end


function full_beam_search2(sm::Subset_minimal; threshold_total_err=0.1, num_samples=1000)
    println("NOT MINE")
    confidence = 1-threshold_total_err
    I3, I2, I1 = init_sbitset(784), init_sbitset(256), init_sbitset(256)
    full_error = heuristic(sm.nn, sm.input, (I3, I2, I1), confidence, num_samples)

    while full_error.hmax > 0
        candidate = find_best(sm, (I3,I2,I1), confidence, num_samples)[1]
        I3, I2, I1 = candidate.ii

        full_error = candidate.h
        println("Length of ii: $((length(I3), length(I2), length(I1))), full_error: ", candidate.h)
    end
    return I3, I2, I1
end


function find_best(sm::Subset_minimal, (I3,I2,I1), confidence, num_samples)
    # searching through I3
    ii₁ = map(setdiff(1:784, I3)) do i
        Iᵢ = push(I3, i)
        (;ii = (Iᵢ, I2, I1), h = heuristic(sm.nn, sm.input, (Iᵢ, I2, I1), confidence, num_samples))
    end
    ii₂ = map(setdiff(1:256, I2)) do i
        Iᵢ = push(I2, i)
        (;ii = (I3, Iᵢ, I1), h = heuristic(sm.nn, sm.input, (I3, Iᵢ, I1), confidence, num_samples))
    end

    ii₃ = map(setdiff(1:256, I1)) do i
        Iᵢ = push(I1, i)
        (;ii = (I3,I2, Iᵢ), h = heuristic(sm.nn, sm.input, (I3, I2, Iᵢ), confidence, num_samples))
    end
    sort(vcat(ii₃, ii₂, ii₁), lt = (i,j) -> i.h.hsum < j.h.hsum)
end

function remove(sm::Subset_minimal, (I3,I2,I1), num_samples)
    for i in I3
        Iᵢ = pop(I3, i)
        if max_error(sm.nn, sm.input, (Iᵢ, I2, I1), num_samples) == 0
            I3 = Iᵢ
        end
    end

    for i in I2
        Iᵢ = pop(I2, i)
        if max_error(sm.nn, sm.input, (I3, Iᵢ, I1), num_samples) == 0
            I2 = Iᵢ
        end
    end

    for i in I1
        Iᵢ = pop(I1, i)
        if max_error(sm.nn, sm.input, (I3, I2, Iᵢ), num_samples) == 0
            I1 = Iᵢ
        end
    end
    (I3, I2, I1)
end

function h_vals(model, xp, (I3, I2, I1), num_samples)
    # println("num_samples: ", num_samples)
    (sdp_full(model, xp, I3, num_samples),
    sdp_full(model[2:3], model[1](xp), I2, num_samples),
    sdp_full(model[3], model[1:2](xp), I1, num_samples),
    sdp_partial(model[1], xp, I3, I2, num_samples),
    sdp_partial(model[1:2], xp, I3, I1, num_samples),
    sdp_partial(model[2], model[1](xp), I2, I1, num_samples),
    )
end

function heuristic(model, xp, (I3, I2, I1), confidence, num_samples)
    hs = h_vals(model, xp, (I3, I2, I1), num_samples)
    (;
    hsum = mapreduce(x -> max(0, confidence-x), +, hs),
    hmax = mapreduce(x -> max(0, confidence-x), max, hs),
    )
end

function max_error(model, xp, (I3, I2, I1), confidence, num_samples)
    hs = h_vals(model, xp, (I3, I2, I1), num_samples)
    mapreduce(x -> max(0, confidence - x), max, hs)
end