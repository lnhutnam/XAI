using Flux
using Flux: DataLoader
using DataFrames
using CSV
using Images
using ImageTransformations

include("hyperparams.jl")

using Random
# Set random seed
Random.seed!(random_seed)  # Replace `random_seed` with an integer, e.g., 42

# CelebA dataset
struct CelebaDataset
    img_dir::String
    csv_path::String
    img_names::Vector{String}
    labels::Vector{Int}
    transform::Union{Function, Nothing}
end

# Constructor function for CelebaDataset
function CelebaDataset(csv_path::String, img_dir::String; transform=nothing)
    df = DataFrame(CSV.File(csv_path))
    img_names = df[!, :Filename]
    labels = df[!, "Male"]
    return CelebaDataset(img_dir, csv_path, img_names, labels, transform)
end

# Length dataset
Base.length(dataset::CelebaDataset) = length(dataset.labels)

# Getitem dataset
function Base.getindex(dataset::CelebaDataset, index::Int)
    # Load the image
    img_path = joinpath(dataset.img_dir, dataset.img_names[index])
    img = load(img_path)

    # Apply transforms if provided
    if !isnothing(dataset.transform)
        img = dataset.transform(img)
    end

    # Get the label
    label = dataset.labels[index]
    return img, label
end

# Get batch
function Base.getindex(dataset::CelebaDataset, indices::Vector{Int})
    images = [dataset[i][1] for i in indices]
    labels = dataset.labels[indices]
    batch_images = cat(images..., dims=4)  # Should be (channels, height, width, batch_size)
    # Ensure correct order: (width, height, channels, batch_size)
    batch_images = permutedims(batch_images, (2, 3, 1, 4))
    return batch_images, labels
end

# Custom transform function
function custom_transform(img)
    h, w = size(img)
    top = (h - 178) ÷ 2 + 1
    left = (w - 178) ÷ 2 + 1
    img_cropped = img[top:top+177, left:left+177]
    img_resized = imresize(img_cropped, (128, 128))
    img_array = channelview(img_resized)  # (channels, height, width)
    img_array = Float32.(img_array) ./ 255.0
    return img_array
end

# Create datasets
train_dataset = CelebaDataset("data/celeba-gender-train.csv", 
                             "data/img_align_celeba/",
                             transform=custom_transform)

valid_dataset = CelebaDataset("data/celeba-gender-valid.csv",
                             "data/img_align_celeba/",
                             transform=custom_transform)

test_dataset = CelebaDataset("data/celeba-gender-test.csv",
                            "data/img_align_celeba/",
                            transform=custom_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, 
                         batchsize=BATCH_SIZE, 
                         shuffle=true)

valid_loader = DataLoader(valid_dataset,
                         batchsize=BATCH_SIZE,
                         shuffle=false)

test_loader = DataLoader(test_dataset,
                        batchsize=BATCH_SIZE,
                        shuffle=false)


# for (images, labels) in train_loader
#     println("Images size: ", size(images))  # Should be (3, 128, 128, 64) for RGB images
#     println("Labels size: ", size(labels))  # Should be (64,)
#     break
# end