function alpha_matting_cutout(
    img, mask, foreground_threshold, background_threshold, erode_structure_size
) end

function naive_cutout(img, mask)
    cutout = colorview(RGBA, RGBA.(img) .* channelview(mask))
    return cutout
end

function post_process(mask) end

function apply_background_color(img, color)
    img_size = size(img)
    r = fill(color[1] / 255, img_size)
    g = fill(color[2] / 255, img_size)
    b = fill(color[3] / 255, img_size)
    a = fill(color[4] / 255, img_size)
    bg = colorview(RGBA{Float32}, r, g, b, a)
    mask = alpha.(img)
    bg = clamp01!(bg .* (1 .- mask) + img .* mask)
    return bg
end

"""
    remove(img; kwargs...)

Remove background from image.

# Examples
```julia
remove(img)
remove(img; session=new_session(U2Net))  # or use specific session
```
"""
function remove(img::Array{<:Number}; kwargs...)::Array{<:Number}
    # unify image channels to 3
    if ndims(img) == 2
        # Change size (hight, width) to (1, hight, width)
        img = reshape(img, 1, size(img)...)
    end
    if size(img, 1) == 1
        # Change size (1, hight, width) to (3, hight, width)
        img = repeat(img, 3)
    elseif size(img, 1) == 4
        # Change size (4, hight, width) to (3, hight, width)
        img = img[1:3, :, :]
    end
    img = copy(colorview(RGB, img))  # convert to Matrix{<:Colorant}

    cutout = remove(img; kwargs...)

    return copy(channelview(cutout))
end

function remove(
    img::Matrix{<:Colorant};
    session::Union{InferenceSession,Nothing}=nothing,
    only_mask::Bool=false,
    bgcolor::Union{Tuple{Int,Int,Int,Int},Nothing}=nothing,
)::Matrix{<:Colorant}
    # get onnx session
    if isnothing(session)
        session = new_session(U2Net)
    end

    # predict mask
    masks = predict(session, img)

    # get cutouts from all masks
    cutouts = []
    for mask in masks
        if only_mask
            cutout = mask
        else
            cutout = naive_cutout(img, mask)
        end
        push!(cutouts, cutout)
    end

    # get cutout by concatenating cutouts vertically
    if length(cutouts) > 0
        cutout = reduce(vcat, cutouts)
    end

    # apply background color
    if !isnothing(bgcolor) && !only_mask
        cutout = apply_background_color(cutout, bgcolor)
    end

    return cutout
end
