using CairoMakie.ColorTypes

const orange = RGBA{Float32}(1.0f0,0.64705884f0,0.0f0,1.0f0)

small_pixel = 8
big_pixel = 20

"""
    arrange(x)

    arrane `x` to a matrix and reverse the axes to make the image compati
"""
arrange(x::Vector) = reverse(reshape(x, 28, 28), dims = 2)
arrange(x::BitVector) = reverse(reshape(x, 28, 28), dims = 2)
arrange(x::Matrix) = x
arrange(x::BitMatrix) = x

function arrange(ii::SBitSet)
    x = zeros(Bool, 784)
    for i in ii
        x[i] = true
    end
    arrange(x)
end

"""
    points(x)

    convert the matrix (or a solution) into a set of `Point2f` for the scatter plot
"""
points(x) = points(arrange(x))
function points(x::Matrix)
    [Point2f(i,j) for i in 1:28, j in 1:28 if x[i,j] > 0]
end

function point_sizes(x)
    x = arrange(x)
    vec([x[i,j] == 1 ? 15 : 8 for i in 1:28, j in 1:28])
end

function point_colors(x)
    x = arrange(x)
    vec([x[i,j] == 1 ? (:gray, 0.7) : (:gray, 0.5) for i in 1:28, j in 1:28])
end

function dotted_digit!(ax::Axis, x)
    x = arrange(x)
    mp = vec([Point2f(i,j) for i in 1:28, j in 1:28])
    ms = point_sizes(x)
    mc = point_colors(x)
    pl = scatter!(ax, mp, color = mc, marker = :circle, markersize = ms)
    ax
end

function dotted_digit(x)
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (500, 500))
    ga = fig[1, 1] = GridLayout()
    ax = GLMakie.Axis(ga[1, 1], backgroundcolor=(0.0, 0.0, 0.0))
    hidedecorations!(ax)
    dotted_digit!(ax, x)
    fig
end

function animate_solutions(solutions, x, ofile)        
    i = Observable(1)
    mask = Makie.@lift(points(solutions[$(i)]))

    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (500, 500))
    ga = fig[1, 1] = GridLayout()
    ax = GLMakie.Axis(ga[1, 1])
    image!(ax, arrange(x), colormap = :grays, alpha = 0.5)
    scatter!(ax, mask, color = :red, marker = :rect)
    # scatter!(ax, points(x), color = :blue)

    framerate = 30
    timestamps = range(1, length(solutions), step=1)

    record(fig, ofile, timestamps;
            framerate = framerate) do t
        i[] = t
    end
end

function animate_images(xx::AbstractMatrix, ofile; framerate = 5)        
    i = Observable(1)
    mp = vec([Point2f(i,j) for i in 1:28, j in 1:28])
    ms = Makie.@lift(point_sizes(xx[:,$(i)]))
    mc = Makie.@lift(point_colors(xx[:,$(i)]))
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (500, 500))
    ga = fig[1, 1] = GridLayout()
    ax = GLMakie.Axis(ga[1, 1])
    hidedecorations!(ax)
    pl = scatter!(ax, mp, color = mc, marker = :circle, markersize = ms)

    timestamps = range(1, size(xx,2), step=1)
    record(fig, ofile, timestamps; framerate) do t
        i[] = t
    end
end

function tile_images(xx::AbstractMatrix; n = 4)        
    fig = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98), size = (n*500, n*500))
    ga = fig[1, 1] = GridLayout()
    for (i, (row,col)) in enumerate(Iterators.product(1:n,1:n))
        ax = GLMakie.Axis(ga[row,col], backgroundcolor = RGBf(0.0, 0.0, 0.0))
        hidedecorations!(ax)
        dotted_digit!(ax, xx[:,i])
    end
    fig
end

dotted_solutions(solutions::AbstractVector{<:AbstractArray}, x) = dotted_solutions(mean(arrange(solutions)), x)

function rescale(x, lo, up)
    @assert up > lo
    mn, mx = minimum(vec(x)), maximum(vec(x))
    δ = mx == mn ? 1 : mx - mn
    x = (x .- mn) ./ δ
    δ = up - lo
    x = x .* δ .+ lo
end
    
function dotted_solutions(mask, x)    
    x = arrange(x)    
    fig = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (600, 500))
    ga = fig[1, 1] = GridLayout()
    ax = GLMakie.Axis(ga[1, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
    ax = dotted_digit!(ax, x)

    # find the points and their value
    mask = arrange(mask)
    mp = [Point2f(i,j) for i in 1:28, j in 1:28 if mask[i,j] != 0]
    mv = [mask[i,j] for i in 1:28, j in 1:28 if mask[i,j] != 0]
    ms = [x[i,j] == 1 ? big_pixel : small_pixel for i in 1:28, j in 1:28 if mask[i,j] != 0]
    # ms = rescale([abs(mask[i,j]) for i in 1:28, j in 1:28 if mask[i,j] != 0], 6, 20)
    hidedecorations!(ax)
    pl = scatter!(ax, mp, color = mv, marker = :circle, markersize = ms, colormap = cgrad(reverse(to_colormap(:heat))))
    Colorbar(fig[1, 2], pl)
    fig
end

function add_rule!(ax, x, ii; explanation_color = orange)
    mask = arrange(ii) 
    mpp = [Point2f(i,j) for i in 1:28, j in 1:28 if mask[i,j] != 0]
    ms = [x[i,j] == 1 ? big_pixel : small_pixel for i in 1:28, j in 1:28 if mask[i,j] != 0]
    scatter!(ax, mpp, color = explanation_color, marker = :circle, markersize = ms)
end

function plot_rule(x, rule; show_digit = true)
    x = arrange(x)    
    fig = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (500, 500))
    ga = fig[1, 1] = GridLayout()
    ax = Axis(ga[1, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
    if show_digit
        ax = dotted_digit!(ax, x)
    else
        scatter!(ax, [Point2f(1,1), Point2f(28,28)], color = :black, marker = :circle, markersize = 6)
    end

    add_rule!(ax, x, rule)
    hidedecorations!(ax)
    fig
end

function add_rule!(ax, sampler, x, ii; explanation_color = orange)
    # find the points and their value
    xₛ = arrange(x)
    d = condition(sampler, x, ii)
    p = sum(softmax(d.r.log_p, dims = 1)[2,:,:] .* d.w', dims = 2)
    p = arrange(vec(p))

    mp = vec([Point2f(i,j) for i in 1:28, j in 1:28])
    mv = vec(rescale([p[i,j] for i in 1:28, j in 1:28], 6, 20))
    scatter!(ax, mp, color = mv, marker = :circle, markersize = mv, colormap = cgrad(to_colormap(:grays)))

    mask = arrange(ii) 
    mpp = [Point2f(i,j) for i in 1:28, j in 1:28 if mask[i,j] != 0]
    ms = [xₛ[i,j] == 1 ? big_pixel : small_pixel for i in 1:28, j in 1:28 if mask[i,j] != 0]
    scatter!(ax, mpp, color = explanation_color, marker = :circle, markersize = ms)
end

function plot_rule(x, rule, sampler)
    fig = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (500, 500))
    ga = fig[1, 1] = GridLayout()
    ax = GLMakie.Axis(ga[1, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
    add_rule!(ax, sampler, x, rule)
    hidedecorations!(ax)
    fig
end

function plot_rule_as_band(x, rule, sampler)
    target_ii = collect(rule)
    fig = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (500*length(target_ii), 500));
    ga = fig[1, 1] = GridLayout()
    xₛ = arrange(x)
    ii = empty_sbitset(x)

    ax = GLMakie.Axis(ga[1, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
    add_rule!(ax, sampler, x, ii)
    for (j,i) in enumerate(target_ii)
        ii = push(ii,i)
        ax = GLMakie.Axis(ga[1, j+1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
        add_rule!(ax, sampler, x, ii)
        hidedecorations!(ax)
    end
    fig
end

function plot_rule_as_sequence(x, rule, sampler)
    target_ii = collect(rule)
    xₛ = arrange(x)
    ii = empty_sbitset(x)
    fig = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (500, 500));
    ga = fig[1, 1] = GridLayout()
    ax = GLMakie.Axis(ga[1, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
    add_rule!(ax, sampler, x, ii)
    figs =[fig]
    for (j,i) in enumerate(target_ii)
        ii = push(ii,i)
        fig = Figure(backgroundcolor = RGBf(1.0, 1.0, 1.0), size = (500, 500));
        ga = fig[1, 1] = GridLayout()
        ax = GLMakie.Axis(ga[1, 1], backgroundcolor = RGBf(0.0, 0.0, 0.0))
        add_rule!(ax, sampler, x, ii)
        hidedecorations!(ax)
        push!(figs, fig)
    end
    figs
end

