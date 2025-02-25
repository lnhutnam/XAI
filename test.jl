using Flux
using Flux: Chain, Conv, MaxPool, Dense, flatten, softmax, onehotbatch, onecold, crossentropy, throttle, DataLoader, setup
using DataFrames
using CSV
using Images
using ImageTransformations
using Random
using Statistics

# Set random seed
Random.seed!(42)  # Replace with your random_seed

# Define VGG16 struct
struct VGG16
    blocks::Chain
    classifier::Chain
end

function VGG16(num_features, num_classes)
    block_1 = Chain(
        Conv((3, 3), 3 => 64, relu, pad=1, stride=1),
        Conv((3, 3), 64 => 64, relu, pad=1, stride=1),
        MaxPool((2, 2), stride=2)
    )
    block_2 = Chain(
        Conv((3, 3), 64 => 128, relu, pad=1, stride=1),
        Conv((3, 3), 128 => 128, relu, pad=1, stride=1),
        MaxPool((2, 2), stride=2)
    )
    block_3 = Chain(
        Conv((3, 3), 128 => 256, relu, pad=1, stride=1),
        Conv((3, 3), 256 => 256, relu, pad=1, stride=1),
        Conv((3, 3), 256 => 256, relu, pad=1, stride=1),
        Conv((3, 3), 256 => 256, relu, pad=1, stride=1),
        MaxPool((2, 2), stride=2)
    )
    block_4 = Chain(
        Conv((3, 3), 256 => 512, relu, pad=1, stride=1),
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        MaxPool((2, 2), stride=2)
    )
    block_5 = Chain(
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        Conv((3, 3), 512 => 512, relu, pad=1, stride=1),
        MaxPool((2, 2), stride=2)
    )
    blocks = Chain(block_1, block_2, block_3, block_4, block_5)
    classifier = Chain(
        flatten,
        Dense(512 * 4 * 4, 4096, relu),
        Dense(4096, 4096, relu),
        Dense(4096, num_classes)
    )
    model = VGG16(blocks, classifier)

    # Weight initialization
    for layer in Flux.modules(model)
        if layer isa Conv
            Flux.params(layer)[1] .= randn(Float32, size(layer.weight)) .* 0.05
            if !isnothing(layer.bias)
                Flux.params(layer)[2] .= 0
            end
        elseif layer isa Dense
            Flux.params(layer)[1] .= randn(Float32, size(layer.weight)) .* 0.05
            Flux.params(layer)[2] .= 0
        end
    end
    return model
end

function (model::VGG16)(x)
    x = model.blocks(x)
    logits = model.classifier(x)
    return logits
end

Flux.@functor VGG16

# Dataset definition
struct CelebaDataset
    img_dir::String
    csv_path::String
    img_names::Vector{String}
    labels::Vector{Int}
    transform::Union{Function, Nothing}
end

function CelebaDataset(csv_path::String, img_dir::String; transform=nothing)
    df = DataFrame(CSV.File(csv_path))
    img_names = df[!, :Filename]
    labels = df[!, "Male"]
    return CelebaDataset(img_dir, csv_path, img_names, labels, transform)
end

Base.length(dataset::CelebaDataset) = length(dataset.labels)

function Base.getindex(dataset::CelebaDataset, index::Int)
    img_path = joinpath(dataset.img_dir, dataset.img_names[index])
    img = load(img_path)
    if !isnothing(dataset.transform)
        img = dataset.transform(img)
    end
    label = dataset.labels[index]
    return img, label
end

function Base.getindex(dataset::CelebaDataset, indices::Vector{Int})
    images = [dataset[i][1] for i in indices]
    labels = dataset.labels[indices]
    batch_images = cat(images..., dims=4)
    batch_images = permutedims(batch_images, (2, 3, 1, 4))
    return batch_images, labels
end

function custom_transform(img)
    h, w = size(img)
    top = (h - 178) ÷ 2 + 1
    left = (w - 178) ÷ 2 + 1
    img_cropped = img[top:top+177, left:left+177]
    img_resized = imresize(img_cropped, (128, 128))
    img_array = channelview(img_resized)
    img_array = Float32.(img_array) ./ 255.0
    return img_array
end

# Create datasets and loaders
train_dataset = CelebaDataset("celeba-gender-train.csv", "img_align_celeba/", transform=custom_transform)
valid_dataset = CelebaDataset("celeba-gender-valid.csv", "img_align_celeba/", transform=custom_transform)
test_dataset = CelebaDataset("celeba-gender-test.csv", "img_align_celeba/", transform=custom_transform)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batchsize=BATCH_SIZE, shuffle=true)
valid_loader = DataLoader(valid_dataset, batchsize=BATCH_SIZE, shuffle=false)
test_loader = DataLoader(test_dataset, batchsize=BATCH_SIZE, shuffle=false)

# Model setup
num_features = 512  # Not used directly, kept for compatibility
num_classes = 2     # Binary gender classification
model = VGG16(num_features, num_classes)

# Move to GPU if available
device = cpu
model = model |> device

# Optimizer setup with Flux ADAM
opt = Adam(0.001)  # Replace with your learning_rate
opt_state = Flux.setup(opt, model)

# Accuracy function
function compute_accuracy(model, data_loader)
    correct_pred, num_examples = 0, 0
    for (features, targets) in data_loader
        features = features |> device
        targets = targets |> device
        logits = model(features)
        probas = softmax(logits, dims=1)
        predicted_labels = onecold(probas, 0:1)
        num_examples += length(targets)
        correct_pred += sum(predicted_labels .== targets)
    end
    return 100 * correct_pred / num_examples
end

# Training loop
num_epochs = 10  # Replace with your num_epochs
start_time = time()

for epoch in 1:num_epochs
    Flux.trainmode!(model)
    for (batch_idx, (features, targets)) in enumerate(train_loader)
        features = features |> device
        targets = targets |> device

        # Debug input and output sizes
        if batch_idx == 1 && epoch == 1
            println("Input size to model: ", size(features))
            logits = model(features)
            println("Logits size: ", size(logits))
            println("Targets size: ", size(targets))
        end

        # Forward and backprop
        logits = model(features)
        loss = crossentropy(logits, onehotbatch(targets, 0:1))
        
        # Gradient computation and update with Flux.setup
        gs = gradient(() -> loss, params(model))
        opt_state, model = Flux.update!(opt_state, model, gs)

        # Logging
        if batch_idx % 50 == 0
            println("Epoch: $epoch/$num_epochs | Batch: $batch_idx/$(length(train_loader)) | Loss: $(round(loss, digits=4))")
        end
    end

    # Evaluation
    Flux.testmode!(model)
    train_acc = compute_accuracy(model, train_loader)
    valid_acc = compute_accuracy(model, valid_loader)
    println("Epoch: $epoch/$num_epochs | Train: $(round(train_acc, digits=3))% | Valid: $(round(valid_acc, digits=3))%")
    println("Time elapsed: $(round((time() - start_time) / 60, digits=2)) min")
end

println("Total Training Time: $(round((time() - start_time) / 60, digits=2)) min")