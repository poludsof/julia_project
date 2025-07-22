using DataFrames, Serialization, StaticBitSets, Statistics, GLMakie
using Flux
import ProbAbEx as PAE
using ProbAbEx
using ProbAbEx: condition
using ProbAbEx.TimerOutputs
using RelevancePropagation
using ExplainableAI
using ShapML
using Random
include("heuristics_criteria.jl")
include("visualization_tools.jl")
include("visualize_skeleton.jl")
# include("training_utilities.jl")
function prepare_data()
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]
    xtrain = map(x -> 2*(x > 0.5) - 1, reshape(xtrain, :, size(xtrain,3)))
    xtest = map(x -> 2*(x > 0.5) - 1, reshape(xtest, :, size(xtest,3)))
    ytrain = ytrain .+ 1
    ytest = ytest .+ 1
    ((xtrain, ytrain), (xtest, ytest))
end


function attack(model, xₑ)
    x = Float32.(xₑ)
    y = Flux.onecold(model(x))
    yo = Flux.onehot(y, 1:10)
    for i in 1:10000
        # @show (onecold(model(x)), Flux.logitcrossentropy(model(x), yo))
        J = gradient(x -> -Flux.logitcrossentropy(model(x), yo), x)[1]

        # change one pixel
        changed = false
        for i in sortperm(abs.(J), rev = true)
            x[i] == -1 && J[i] > 0 && continue
            x[i] == 1 && J[i] < 0 && continue
            x[i] = - x[i]
            changed = true
            break
        end
        changed || break
        Flux.onecold(model(x)) != y && break
    end
    Flux.onecold(model(x)) == y && println("The attack has failed")
    x
end

function min_length_max_recall(df)
    ll = length.(df.rule)
    l = minimum(ll)
    df = df[ll .== l, :]
    df.rule[argmax(df.train_recall)]
end

empty_sbitset(n::Int) = SBitSet{ceil(Int, n / 64), UInt64}()
empty_sbitset(x::AbstractArray) = empty_sbitset(length(x))


############
#   prior art in explanations
############
let 
    x = deserialize("mnist_proper/model_noise_forward_beamsearch.jls")[1].x;
    model = deserialize("mnist_proper/model_default.jls")

    # Let's plot the solution offered by Layerwise relevance propagation
    composite = EpsilonPlus()
    analyzer = LRP(model, composite)
    expl = analyze(Float32.(reshape(x, :,1)), analyzer)
    save("latex/images/lre.pdf", dotted_solutions(vec(expl.val), x))

    # Let's add GradCAM
    analyzer = SmoothGrad(model)
    expl = analyze(Float32.(reshape(x, :,1)), analyzer)  # or: expl = analyzer(input)
    save("latex/images/smoothgrad.pdf", dotted_solutions(vec(expl.val), x))


    # Compute stochastic Shapley values.
    # train, test = prepare_data();
    # df_data = DataFrame(Matrix(train[1]'), :auto)
    # data_shap = ShapML.shap(explain = df_data[2:2,:],
    #                         reference = df_data,
    #                         model = model,
    #                         predict_function = (model, x) -> Flux.onecold(model(Matrix(x)')),
    #                         sample_size = 10_000,
    #                         seed = 1
    #                         )
    # serialize("mnist_proper/model_default_data_shaple.jls", (;data_shap, x))
    data_shap, x = deserialize("mnist_proper/model_default_data_shaple.jls")
    save("latex/images/shap.pdf", dotted_solutions(data_shap.shap_effect, x))


    # Let's do a binary counterfactual
    c = attack(model, x)
    save("latex/images/counterfactual.pdf", dotted_solutions(c - x, x))
end


############
#   Let's have examples of few explanations
############
let 
    df = DataFrame(deserialize("mnist_proper/model_default_uniform_forward_beamsearch.jls"));
    for y in 1:10
        img_id = first(unique(filter(r -> r.y == y, df).img_id))
        subdf = filter(r -> r.img_id == img_id, df)
        x = subdf.x[1]  
        rule = min_length_max_recall(df)
        save("latex/images/explanation_default_uniform_$(y-1).pdf", plot_rule(x, rule))
    end
end


############
#   Let's view distributions of explanations
############
let 
    df = DataFrame(deserialize("mnist_proper/model_default_uniform_forward_img_2.jls").stats);
    hist(length.(df.rule), bins = 80)
    # hist(df.train_recall, bins = 5)
    # hist(df.train_precision, bins = 20)
    f = Figure(size = (500,500));
    ax = Axis(f[1, 1], xlabel = "rule length", ylabel = "count");
    hist!(ax, length.(df.rule), bins = 80);
    save("latex/images/histogram_2.pdf", f)
end



############
#   differences between samplers
############
let 
    x = deserialize("mnist_proper/model_noise_forward_beamsearch.jls")[1].x;
    sampler = UniformDistribution()
    xx = PAE.sample_all(condition(sampler, x, empty_sbitset(x)), 100)
    # animate_images(xx, "sample_uniform.gif")        
    save("latex/images/sample_uniform.pdf", tile_images(xx, n = 2))        
    sampler = BernoulliMixture(deserialize(joinpath("mnist_proper", "milan_centers.jls")))
    Random.seed!(1)
    xx = PAE.sample_all(condition(sampler, x, empty_sbitset(x)), 100)
    save("latex/images/sample_data.pdf", tile_images(xx, n = 2))          
end

############
#   visualize explanations from each method
############
let 
    border_id = [(j-1)*28+i for i in 1:28, j in 1:28 if i ∈ (1,28) || j ∈ (1,28)]
    df = DataFrame(deserialize("mnist_proper/model_default_uniform_forward_beamsearch.jls"));
    df = filter(df -> df.img_id==2, df)
    x = df.x[1]
    # ii = argmin(length, df.rule)
    ii = df.rule[findfirst(ii -> any(∈(border_id), ii), df.rule)]
    save("latex/images/minlength_uniform.pdf", plot_rule(x, ii))

    df = DataFrame(deserialize("mnist_proper/model_default_forward_beamsearch.jls"));
    df = filter(df -> df.img_id==2, df)
    x = df.x[1]
    ii = argmin(length, df.rule)
    save("latex/images/minlength_data.pdf", plot_rule(x, ii))
end

############
#   Let's visualize the step-by-step process of finding the explanation
############
let 
    sampler = BernoulliMixture(deserialize(joinpath("mnist_proper", "milan_centers.jls")))
    df = DataFrame(deserialize("mnist_proper/model_default_forward_beamsearch.jls"));
    df = filter(df -> df.img_id==2, df)
    x = df.x[1]

    rule = argmin(length, df.rule)
    let ii = empty_sbitset(x)
        save("latex/images/evolution_1.pdf",  plot_rule(x, ii,sampler))
        save("latex/images/evolution_rule_1.pdf",  plot_rule(x, ii;show_digit=false))
        for (i, j) in enumerate(rule)
            ii = push(ii, j)
            save("latex/images/evolution_$(i+1).pdf",  plot_rule(x, ii,sampler))
            save("latex/images/evolution_rule_$(i+1).pdf",  plot_rule(x, ii;show_digit=false))
        end
    end

    # generate just the data distribution as an explanation as a introduction
    save("latex/images/evolution_0.pdf", plot_rule(x, empty_sbitset(x),sampler))
end
############
#  Let's show the conditional distribution for few points to explain 
#  the effect of the correlation and constants.
############
let 
    sampler = BernoulliMixture(deserialize(joinpath("mnist_proper", "milan_centers.jls")))
    border_id = [(j-1)*28+i for i in 1:28, j in 1:28 if i ∈ (1,28) || j ∈ (1,28)]
    df = DataFrame(deserialize("mnist_proper/model_default_uniform_forward_beamsearch.jls"));
    df = filter(df -> df.img_id==2, df)
    x = df.x[1]
    uni_ii = df.rule[findfirst(ii -> any(∈(border_id), ii), df.rule)]

    # Let's create a small subset of the uni_ii distribution
    ii = foldl((x,y) -> push(x, y), sort(collect(uni_ii))[8:10:end], init = typeof(uni_ii)())
    save("latex/images/correlation.pdf", plot_rule(x, ii, sampler))
end


############
#   Let's visualize the effect of the regularization
############
let 
    for model in ["model_default", "model_l1", "model_l2", "model_lasso", "model_noise"]
        df = DataFrame(deserialize("mnist_proper/$(model)_uniform_forward_beamsearch.jls"));
        df = filter(df -> df.img_id==2, df)
        x = df.x[1]
        ii = argmin(length, df.rule)
        save("latex/images/$(model)_uniform.pdf", plot_rule(x, ii))
    end
end
############
#   Let's visualize the effect of adversarial samples
############
let 
    sampler = BernoulliMixture(deserialize(joinpath("mnist_proper", "milan_centers.jls")))
    df = DataFrame(deserialize("mnist_proper/model_l2_uniform_forward_beamsearch.jls"));
    df = filter(df -> df.img_id==2, df)
    x = df.x[1]
    ii = argmin(length, df.rule)
    save("latex/images/adversarial_x_digit.pdf", dotted_digit(x))
    save("latex/images/adversarial_x_digit_with_rule.pdf", plot_rule(x, ii))
    save("latex/images/adversarial_x_rule.pdf", plot_rule(x, ii, sampler))

    df = DataFrame(deserialize("mnist_proper/model_l2_adversarial_uniform_forward_beamsearch.jls"));
    df = filter(df -> df.img_id==2, df)
    x = df.xa[1]
    ii = argmin(length, df.rule)
    save("latex/images/adversarial_xa_digit.pdf", dotted_digit(x))
    save("latex/images/adversarial_xa_digit_with_rule.pdf", plot_rule(x, ii))
    save("latex/images/adversarial_xa_rule.pdf", plot_rule(x, ii, sampler))
end


############
#   Let's plot the skeleton of a network explaining a sample
############
sampler = BernoulliMixture(centers);
model = deserialize("mnist/binary_model.jls")
rule, skeleton, xₛ, _ = deserialize("mnist_proper/skeleton_90.jls")
r = PAE.condition(sampler, xₛ, rule[1])
xx = PAE.sample_all(r, 10_000);

l₀ = vec(mean(xx .> 0, dims = 2))
l₁ = vec(mean(model[1:1](xx) .> 0, dims = 2))
l₂ = vec(mean(model[1:2](xx) .> 0, dims = 2))
l₃ = vec(mean(model[1:3](xx) .> 0, dims = 2))

layers = [l₁, l₂]
pos = mapreduce(vcat, enumerate(layers)) do (i, ls)
    jitter = 0.2
    [Point2f(i + jitter * (mod(j, 10) / 10),length(ls) ÷ 2 - j) for (j, v) in enumerate(ls)]
end;
pc = reduce(vcat, layers);

fig = Figure(backgroundcolor = RGBf(1, 1, 1), size = (500, 500));
ga = fig[1, 1] = GridLayout()
ax = GLMakie.Axis(ga[1, 1])
scatter!(ax, pos; color = pc)

pos = [Point2f(3,10*(length(l₃) ÷ 2 - j)) for (j, v) in enumerate(l₃)]
pc = reduce(vcat, l₃);
sc = scatter!(ax, pos; color = pc)
hidedecorations!(ax)
Colorbar(fig[1, 2], sc, flipaxis = true)
fig

save("latex/images/skeleton_motivation.pdf", fig)


############
#   Let's plot the skeleton of a network explaining a sample
############
centers = deserialize(joinpath("mnist_proper", "binary_model_hidden_centers.jls"))
samplers = (BernoulliMixture(centers[:,1:784,:]), BernoulliMixture(centers[:,785:1040,:]), BernoulliMixture(centers[:,1041:end,:]))
rule, skeleton, xₛ, _ = deserialize("mnist_proper/skeleton_90.jls")

fig = plot_skeleton(rule, skeleton, xₛ, samplers;explanation_color = :red)
save("latex/images/skeleton.pdf", fig)
