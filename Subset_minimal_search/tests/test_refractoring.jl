
``` Test refactoring of the code ```


""" Prepare model and data """
using Subset_minimal_search
using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy

model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)

train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)


""" Prepare image and label """
img = train_X_bin_neg[:, 1]
label_img = argmax(train_y[:, 1]) - 1

Subset_minimal(model, img, label_img)


""" Test backward search """


""" Test forward search """

