
"""
"""

model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)

# Prepare data and model
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_bin_neg = preprocess_bin_neg(preprocess_binary(train_X))
test_X_bin_neg = preprocess_bin_neg(preprocess_binary(test_X))

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)
 
img = train_X_bin_neg[:, 1]
label_img = argmax(test_y[:, 1]) - 1
threshold=0.1
num_samples=100

using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy, Subset_minimal, full_forward_search
full_forward_search(Subset_minimal(model, img, label_img), threshold, num_samples)
