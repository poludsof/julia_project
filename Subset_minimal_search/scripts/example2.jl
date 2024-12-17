
"""
Beam search with b=1 with partial and full probability criterion
"""

using Subset_minimal_search

using Subset_minimal_search.Flux
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.Makie
using Subset_minimal_search.CairoMakie
using Subset_minimal_search.Serialization

using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy, Subset_minimal, full_beam_search

model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)
# println(model)


# Prepare data and model
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

# println("Digit: ", argmax(train_y[:, 1]) - 1, "\nOtput of the model: ", model(train_X_bin_neg[:, 1]), "\nModel's digit: ", argmax(model(train_X_bin_neg[:, 1])) - 1)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)
 
# test accuracy of the model
# println("Train Accuracy: ", accuracy(model, train_X_bin_neg, train_y) * 100, "%")
# println("Test Accuracy: ", accuracy(model, test_X_bin_neg, test_y) * 100, "%")

# Test random sampling
img = train_X_bin_neg[:, 1]
label_img = argmax(train_y[:, 1]) - 1
threshold=0.1
# num_best=1
num_samples=1000

# Length of ii: (271, 80, 78), full_error: 0.09962499999999996 heuristic: 0.0 --- 0.9
# Length of ii: (309, 86, 74), full_error: 0.09732558139534886 heuristic: 0.07475801382778113 --- 0.95

# updated sampling + stop condition:
# Length of ii: (72, 25, 35), full_error: 0.0 heuristic: 0.0
# updated sampling + my stop condition:
# Length of ii: (66, 26, 32), full_error: 0.09959375000000004 heuristic: 0.0
#previous sampling + my stop condition:
# Length of ii: (73, 24, 41), full_error: 0.09899999999999998 heuristic: 0.0
best_set = full_beam_search2(Subset_minimal(model, img, label_img), threshold, num_samples)
reduced_sets = backward_dfs_search(Subset_minimal(model, img, label_img), best_set, num_samples)
# length: (306, 84, 70) Current error: 0.08714285714285719 Current heuristic: 0.04973809523809514
memory_reduced_sets = deepcopy(reduced_sets)

#TEST reduced sets
max_error(model, img, best_set, 0.9, num_samples) + 0.9
max_error(model, img, reduced_sets, 0.9, num_samples) + 0.9

# I3_set, I2_set, I1_set = best_set
# println("Subset I3: ", I3_set)
# println("Subset I2: ", I2_set)
# println("Subset I1: ", I1_set)


# for i in collect(I3_test)
#     println("Subset: ", i)
# end

# I3_set_tmp = deepcopy(I3_set)
# I3_tmp, I2_tmp, I1_tmp = deepcopy(best_set)
# popped = pop(I3_tmp, 4)
# I3_test = push(I3_test, 4)
# println("Subset I3: ", I3_test)


# TEST backward
# I3_test = SBitSet{32, UInt32}(collect(5:10))
# I2_test = SBitSet{32, UInt32}(collect(1:40))
# I1_test = SBitSet{32, UInt32}(collect(1:68))
# red_test_sets = backward_dfs_search(Subset_minimal(model, img, label_img), (I3_test, I2_test, I1_test), 0.7, num_samples)

# TEST stack 
# stack_test = [(SBitSet{32, UInt32}(collect(1:5)), 10, 0.1), (SBitSet{32, UInt32}(collect(1:4)), 9, 0.2), (SBitSet{32, UInt32}(collect(1:6)), 8, 0.3)]
# sort!(stack_test, by = x -> -length(x[1]))
# println(pop!(stack_test))
# max_stack_size = 2
# stack_test = stack_test[end-max_stack_size+1:end]

# img_bin = train_X_binary[:, 1]
# plot_set = Set([i for i in best_set[1]])
# plot_mnist_with_active_pixels(img_bin, plot_set)

# best_set = forward_beam_search2(Subset_minimal(model, img, label_img), 0.7, num_samples, 3)

best_set = full_beam_search_with_stack(Subset_minimal(model, img, label_img), 0.8, num_samples)


# Length of ii: (71, 25, 31), full_error: (hsum = 0.0, hmax = 0.0)

# I3 = SBitSet{13,UInt64}{4,21,46,55,71,72,79,100,101,102,103,109,116,133,134,145,158,163,169,191,197,200,201,229,232,242,257,262,264,267,268,269,296,300,301,302,310,312,319,320,330,332,333,359,360,361,391,404,412,431,447,456,538,588,602,604,606,626,627,648,667,669,670,679,698,704,705,735,736,778,781}
# I2 = SBitSet{4,UInt64}{2,27,42,52,55,60,67,69,82,83,84,85,98,104,147,156,183,186,191,197,209,213,224,230,234}
# I1 = SBitSet{4,UInt64}{8,9,12,15,19,23,28,31,52,56,63,67,80,89,92,110,124,126,151,154,176,189,196,203,208,235,240,246,248,252,255}

I3_array = [7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729]
I2_array = [11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]
I1_array = [5,7,8,9,12,19,23,28,32,48,51,52,76,113,151,154,204,206,208,215,216,235,246,248,252,255]

function man_push(array, sbitset)
    for i in array
        sbitset = push(sbitset, i)
    end
    return sbitset
end

I3 = man_push(I3_array, SBitSet{13, UInt64}())
I2 = man_push(I2_array, SBitSet{4, UInt64}())
I1 = man_push(I1_array, SBitSet{4, UInt64}())
# redcd = [158,169,201,264,269,300,312,333,391,404,412,447,604,606,669,705,735]
# subsubset_I2 = man_push(redcd, SBitSet{32, UInt32}())

subset_threshold = 0.98
num_samples = 1000
I3, I2, I1 = memory_reduced_sets
subsubset_I2 = subset_for_I2(Subset_minimal(model, img, label_img), I3, I2, subset_threshold, num_samples)
println("Subsubset I2: ", subsubset_I2)

#TEST
sdp_partial(model[1], img, subsubset_I2[1], 2, 1000)

println("Subset I2: ", I2)
subsubset_I1 = subset_for_I1(Subset_minimal(model, img, label_img), I3, I2, I1, subset_threshold, num_samples)
println("Subsubset I1: ", subsubset_I1)




num_samples = 1000
fwd_best_set = full_beam_search_with_stack(Subset_minimal(model, img, label_img), 0.1, num_samples)
fwd_reduced_sets = backward_dfs_search(Subset_minimal(model, img, label_img), fwd_best_set, num_samples)
max_error(model, img, fwd_best_set, 0.9, num_samples)
max_error(model, img, stets_to_vis, 0.9, num_samples)


stets_to_vis = deepcopy(fwd_reduced_sets)
I3, I2, I1 = stets_to_vis
# final length: (54, 21, 26) 
# (SBitSet{13,UInt64}{7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{5,7,8,9,12,19,23,28,32,48,51,52,76,113,151,154,204,206,208,215,216,235,246,248,252,255,})
subsubset_I2 = subset_for_I2(Subset_minimal(model, img, label_img), I3, I2, subset_threshold, num_samples)
subsubset_I1 = subset_for_I1(Subset_minimal(model, img, label_img), I3, I2, I1, subset_threshold, num_samples)


# Subsubset I2: Any[SBitSet{13,UInt64}{7,96,101,110,155,161,174,201,237,242,257,298,301,326,328,330,338,378,432,559,602,604,626,729,}, SBitSet{13,UInt64}{110,133,145,161,163,174,242,257,296,299,301,302,378,404,417,538,559,636,}, SBitSet{13,UInt64}{79,96,101,145,155,161,174,201,207,232,301,302,326,338,359,432,441,489,559,567,647,700,705,729,}, SBitSet{13,UInt64}{7,73,79,96,145,155,161,174,300,302,330,360,626,700,705,}, SBitSet{13,UInt64}{73,79,174,302,330,355,357,359,360,489,626,705,729,}, SBitSet{13,UInt64}{7,63,101,155,201,257,300,326,328,338,357,390,441,538,559,567,705,}, SBitSet{13,UInt64}{7,63,101,110,145,155,163,237,296,298,299,301,338,538,567,602,643,705,729,}, SBitSet{13,UInt64}{7,63,79,101,110,133,155,174,191,207,237,242,298,300,326,328,330,359,360,432,559,602,604,626,647,705,729,}, SBitSet{13,UInt64}{73,101,110,133,145,161,163,191,232,242,298,300,302,328,338,357,359,360,404,417,432,538,626,647,729,}, SBitSet{13,UInt64}{63,73,79,133,145,155,161,174,191,201,232,242,296,298,299,301,302,330,355,357,358,390,432,441,489,559,626,700,}, SBitSet{13,UInt64}{63,79,133,161,174,201,232,242,296,298,301,328,355,358,359,360,441,489,559,602,636,}, SBitSet{13,UInt64}{79,101,110,155,207,237,296,299,300,301,302,326,330,355,359,390,432,538,559,602,604,643,705,}, SBitSet{13,UInt64}{7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729,}, SBitSet{13,UInt64}{96,133,145,163,191,207,232,237,257,298,390,441,602,}, SBitSet{13,UInt64}{96,110,133,242,298,300,301,328,355,359,378,404,417,432,538,559,567,602,636,643,705,729,}, SBitSet{13,UInt64}{79,96,161,174,191,207,242,257,296,338,355,358,360,404,417,559,567,602,636,}, SBitSet{13,UInt64}{73,96,101,155,161,191,232,237,257,298,300,302,326,328,338,357,359,360,378,390,441,602,604,636,643,647,700,}, SBitSet{13,UInt64}{7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729,}, SBitSet{13,UInt64}{63,73,79,101,110,133,155,161,163,174,207,237,296,302,338,357,359,378,390,602,626,636,643,700,729,}, SBitSet{13,UInt64}{73,96,101,110,133,155,161,163,201,207,301,326,328,330,358,360,390,404,432,538,602,626,636,643,729,}, SBitSet{13,UInt64}{7,63,73,79,96,110,145,155,174,191,207,232,242,302,328,330,338,357,358,359,360,390,417,432,489,559,647,700,}]
# Subsubset I1: Any[SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{42,85,112,183,190,}, SBitSet{4,UInt64}{27,42,186,190,197,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{27,67,157,183,186,190,197,217,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{11,60,69,85,209,}, SBitSet{4,UInt64}{27,42,60,84,85,156,183,209,217,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{11,32,42,60,84,112,156,157,191,197,217,}, SBitSet{4,UInt64}{11,67,83,84,112,156,186,191,209,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{42,60,69,83,156,157,191,224,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{32,42,60,67,69,83,85,156,157,186,190,191,209,217,224,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{27,84,85,183,217,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}, SBitSet{4,UInt64}{32,60,67,69,83,156,157,}, SBitSet{4,UInt64}{27,156,186,209,224,}, SBitSet{4,UInt64}{42,84,85,157,209,217,}, SBitSet{4,UInt64}{11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224,}]