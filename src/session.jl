U2NET_ONNX_PATH = joinpath(artifact"u2net", "u2net.onnx")
U2NETP_ONNX_PATH = joinpath(artifact"u2netp", "u2netp.onnx")
U2NET_CLOTH_SEG_ONNX_PATH = joinpath(artifact"u2net_cloth_seg", "u2net_cloth_seg.onnx")
U2NET_HUMAN_SEG_ONNX_PATH = joinpath(artifact"u2net_human_seg", "u2net_human_seg.onnx")
SILUETA_ONNX_PATH = joinpath(artifact"silueta", "silueta.onnx")
ISNET_GENERAL_USE_ONNX_PATH = joinpath(
    artifact"isnet-general-use", "isnet-general-use.onnx"
)

abstract type MattingModel end
struct U2NetModel <: MattingModel end
struct U2NetpModel <: MattingModel end
struct U2NetClothSegModel <: MattingModel end
struct U2NetHumanSegModel <: MattingModel end
struct SiluetaModel <: MattingModel end
struct ISNetGeneralUseModel <: MattingModel end

const U2Net = U2NetModel()
const U2Netp = U2NetpModel()
const U2NetClothSeg = U2NetClothSegModel()
const U2NetHumanSeg = U2NetHumanSegModel()
const Silueta = SiluetaModel()
const ISNetGeneralUse = ISNetGeneralUseModel()

abstract type InferenceSession end

struct SimpleSession <: InferenceSession
    model_name::String
    onnx_session::OX.InferenceSession
end
struct ClothSession <: InferenceSession
    model_name::String
    onnx_session::OX.InferenceSession
end
struct DisSession <: InferenceSession
    model_name::String
    onnx_session::OX.InferenceSession
end

function new_session(::U2NetModel)
    return SimpleSession("U2Net", OX.load_inference(U2NET_ONNX_PATH))
end
function new_session(::U2NetpModel)
    return SimpleSession("U2Netp", OX.load_inference(U2NETP_ONNX_PATH))
end
function new_session(::U2NetClothSegModel)
    return ClothSession("U2NetClothSeg", OX.load_inference(U2NET_CLOTH_SEG_ONNX_PATH))
end
function new_session(::U2NetHumanSegModel)
    return SimpleSession("U2NetHumanSeg", OX.load_inference(U2NET_HUMAN_SEG_ONNX_PATH))
end
function new_session(::SiluetaModel)
    return SimpleSession("Silueta", OX.load_inference(SILUETA_ONNX_PATH))
end
function new_session(::ISNetGeneralUseModel)
    return DisSession("ISNetGeneralUse", OX.load_inference(ISNET_GENERAL_USE_ONNX_PATH))
end

function predict(session::SimpleSession, img::Matrix{<:Colorant})

    # Preprocessing
    small_img = imresize(img, 320, 320)
    img_array = copy(channelview(small_img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
    img_array = copy(rawview(img_array))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)
    img_array = img_array / maximum(img_array)
    tmp_img = zeros(Float32, 1, 3, 320, 320)  # input size: (batch_size, 3, 320, 320)
    tmp_img[1, 1, :, :] = (img_array[1, :, :] .- 0.485) / 0.229
    tmp_img[1, 2, :, :] = (img_array[2, :, :] .- 0.456) / 0.224
    tmp_img[1, 3, :, :] = (img_array[3, :, :] .- 0.406) / 0.225

    # Predict
    input = Dict(session.onnx_session.input_names[1] => tmp_img)
    # output size: (batch_size, 1, height, width)
    pred = session.onnx_session(input)[session.onnx_session.output_names[1]][:, 1, :, :]

    # Postprocessing
    ma = maximum(pred)
    mi = minimum(pred)
    pred = (pred .- mi) / (ma .- mi)
    pred = dropdims(pred; dims=1)

    # Creates a view of the numeric array `pred`,
    # interpreting successive elements of `pred` as if they were channels of Colorant `Gray`.
    mask = colorview(Gray, pred)
    mask = imresize(mask, size(img))
    return [mask]
end

function predict(session::DisSession, img::Matrix{<:Colorant})

    # Preprocessing
    small_img = imresize(img, 1024, 1024)
    img_array = copy(channelview(small_img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
    img_array = copy(rawview(img_array))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)
    img_array = img_array / maximum(img_array)
    tmp_img = zeros(Float32, 1, 3, 1024, 1024)  # input size: (batch_size, 3, 320, 320)
    tmp_img[1, 1, :, :] = (img_array[1, :, :] .- 0.485) / 1.0
    tmp_img[1, 2, :, :] = (img_array[2, :, :] .- 0.456) / 1.0
    tmp_img[1, 3, :, :] = (img_array[3, :, :] .- 0.406) / 1.0

    # Predict
    input = Dict(session.onnx_session.input_names[1] => tmp_img)
    # output size: (batch_size, 1, height, width)
    pred = session.onnx_session(input)[session.onnx_session.output_names[1]][:, 1, :, :]

    # Postprocessing
    ma = maximum(pred)
    mi = minimum(pred)
    pred = (pred .- mi) / (ma .- mi)
    pred = dropdims(pred; dims=1)

    # Creates a view of the numeric array `pred`,
    # interpreting successive elements of `pred` as if they were channels of Colorant `Gray`.
    mask = colorview(Gray, pred)
    mask = imresize(mask, size(img))
    return [mask]
end

# TODO
# function predict(session::ClothSession, img::Matrix{<:Colorant})

#     # Preprocessing
#     small_img = imresize(img, 768, 768)
#     img_array = copy(channelview(small_img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
#     img_array = copy(rawview(img_array))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)
#     img_array = img_array / maximum(img_array)
#     tmp_img = zeros(Float32, 1, 3, 768, 768)  # input size: (batch_size, 3, 768, 768)
#     tmp_img[1, 1, :, :] = (img_array[1, :, :] .- 0.485) / 0.229
#     tmp_img[1, 2, :, :] = (img_array[2, :, :] .- 0.456) / 0.224
#     tmp_img[1, 3, :, :] = (img_array[3, :, :] .- 0.406) / 0.225

#     # Predict
#     input = Dict(session.onnx_session.input_names[1] => tmp_img)
#     # output size: (batch_size, 1, height, width)
#     pred = session.onnx_session(input)
#     println(size(pred[session.onnx_session.output_names[1]]))
#     pred = log_softmax(pred[session.onnx_session.output_names[1]], 1)
#     # pred = argmax(pred; dims=1)
#     println(size(pred))
#     pred = dropdims(pred; dims=1)
#     println(size(pred))

#     mask = colorview(RGBA, pred)
#     mask = imresize(mask, size(img))

#     masks = []

#     mask1 = copy(mask)
#     # mask1.putpalette(pallete1)
#     # mask1 = mask1.convert("RGB").convert("L")
#     push!(masks, mask1)

#     mask2 = copy(mask)
#     # mask1.putpalette(pallete1)
#     # mask1 = mask1.convert("RGB").convert("L")
#     push!(masks, mask2)

#     mask3 = copy(mask)
#     # mask1.putpalette(pallete1)
#     # mask1 = mask1.convert("RGB").convert("L")
#     push!(masks, mask3)

#     return masks
# end
