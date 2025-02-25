include("dataset.jl")
include("networks.jl") 

using Flux
using Flux: softmax, onehotbatch, onecold, crossentropy, trainmode!, testmode!, params, setup, update!
using ParameterSchedulers

# Define model
model = VGG16(num_features, num_classes)

# Use CPU only
device = cpu
model = model |> device

# Optimizer setup with Flux ADAM
optim = Flux.setup(Adam(learning_rate), model)

# Accuracy function
function compute_accuracy(model, data_loader)
    correct_pred, num_examples = 0, 0
    for (features, targets) in data_loader
        features = features |> device
        targets = targets |> device
        logits = model(features)  # Use logits directly

        probas = softmax(logits, dims=1)
        predicted_labels = onecold(probas, 0:1)  # Assuming labels are 0 and 1
        num_examples += length(targets)
        correct_pred += sum(predicted_labels .== targets)
    end
    return 100 * correct_pred / num_examples
end

# Training loop
num_epochs = 2  # Replace with your num_epochs
start_time = time()

for epoch in 1:num_epochs
    trainmode!(model)
    for (batch_idx, (features, targets)) in enumerate(train_loader)
        features = features |> device
        targets = targets |> device

        # Debug input and output sizes
        if batch_idx == 1 && epoch == 1
            println("Input size to model: ", size(features))
            logits = model(features)
            probas = softmax(logits, dims=1)
            println("Logits size: ", size(logits))
            println("Logits size: ", size(probas))
            println("Targets size: ", size(targets))
        end

        # Forward and backprop with Flux.withgradient
        loss, grads = Flux.withgradient(model) do m
            y_hat = softmax(m(features))
            crossentropy(y_hat, onehotbatch(targets, 0:1)) 
        end
        Flux.update!(optim, model, grads[1])

        # Logging
        if batch_idx % 50 == 0
            println("Epoch: $epoch/$num_epochs | Batch: $batch_idx/$(length(train_loader)) | Loss: $(round(loss, digits=4))")
        end
    end

    # Evaluation
    testmode!(model)
    train_acc = compute_accuracy(model, train_loader)
    valid_acc = compute_accuracy(model, valid_loader)
    println("Epoch: $epoch/$num_epochs | Train: $(round(train_acc, digits=3))% | Valid: $(round(valid_acc, digits=3))%")
    println("Time elapsed: $(round((time() - start_time) / 60, digits=2)) min")
end

println("Total Training Time: $(round((time() - start_time) / 60, digits=2)) min")