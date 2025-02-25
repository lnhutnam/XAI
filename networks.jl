using Flux
using Flux: Chain, Conv, MaxPool, Dense, flatten, softmax, onehotbatch, onecold, crossentropy, throttle, DataLoader

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
            Flux.params(layer)[1] .= randn(Float32, size(layer.weight)) .* 0.05  # Changed to Float32 for consistency
            if !isnothing(layer.bias)
                Flux.params(layer)[2] .= 0
            end
        elseif layer isa Dense
            Flux.params(layer)[1] .= randn(Float32, size(layer.weight)) .* 0.05  # Changed to Float32
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

# Testing
# model = VGG16(512, 2)  # num_features isn’t used directly, kept for compatibility

# # Test with a random input (batch of 64, 3 channels, 128x128 images)
# x = rand(Float32, 128, 128, 3, 64)
# logits = model(x)
# probas = softmax(logits, dims=1)

# println("Logits size: ", size(logits))  # Should be (2, 64)
# println("Probas size: ", size(probas))  # Should be (2, 64)
# println("Logits range: ", minimum(logits), " to ", maximum(logits))  # Check raw logits range
# println("Probas range: ", minimum(probas), " to ", maximum(probas))  # Should be [0, 1]