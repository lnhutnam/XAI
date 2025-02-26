include("dataset.jl")

using Flux, BSON
@load "resnet18_model.bson" model
model = model |> cpu
# Test with a batch from train_loader
features, targets = first(train_loader)
logits = model(features)
println(size(logits))  # Should be (2, 64)