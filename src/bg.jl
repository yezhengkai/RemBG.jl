function alpha_matting_cutout(
    img, mask, foreground_threshold, background_threshold, erode_structure_size
) end

function naive_cutout(img, mask)
    cutout = colorview(RGB, img .* channelview(mask))
    return cutout
end

function get_concat_v_multi(imgs)
    return reduce(get_concat_v, imgs)
end

function get_concat_v(img1, img2)
    return vcat(img2, img1)
end

function post_process(mask) end

function apply_background_color(img, color) end

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
    img::Matrix{<:Colorant}; session::Union{InferenceSession,Nothing}=nothing
)::Matrix{<:Colorant}
    # get onnx session
    if isnothing(session)
        session = new_session(U2Netp)
    end

    # predict mask
    masks = predict(session, img)

    # get cutouts from all masks
    cutouts = []
    for mask in masks
        cutout = naive_cutout(img, mask)
        push!(cutouts, cutout)
    end

    # get cutout by concatenating cutouts
    if length(cutouts) > 0
        cutout = get_concat_v_multi(cutouts)
    end

    # apply background color

    return cutout
end
