U2NET_ONNX_PATH = joinpath(artifact"u2net", "u2net.onnx")
U2NETP_ONNX_PATH = joinpath(artifact"u2netp", "u2netp.onnx")
U2NET_CLOTH_SEG_ONNX_PATH = joinpath(artifact"u2net_cloth_seg", "u2net_cloth_seg.onnx")
U2NET_HUMAN_SEG_ONNX_PATH = joinpath(artifact"u2net_human_seg", "u2net_human_seg.onnx")
SILUETA_ONNX_PATH = joinpath(artifact"silueta", "silueta.onnx")
ISNET_GENERAL_USE_ONNX_PATH = joinpath(artifact"isnet-general-use", "isnet-general-use.onnx")

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
    return SimpleSession("U2NetClothSeg", OX.load_inference(U2NET_CLOTH_SEG_ONNX_PATH))
end
function new_session(::U2NetHumanSegModel)
    return ClothSession("U2NetHumanSeg", OX.load_inference(U2NET_CLOTH_SEG_ONNX_PATH))
end
function new_session(::SiluetaModel)
    return SimpleSession("Silueta", OX.load_inference(SILUETA_ONNX_PATH))
end
function new_session(::ISNetGeneralUseModel)
    return DisSession("ISNetGeneralUse", OX.load_inference(ISNET_GENERAL_USE_ONNX_PATH))
end

function predict(session::SimpleSession, img::Matrix{<:Colorant})

    # input size: (batch_size, 3, 320, 320)
    small_img = imresize(img, 320, 320)
    img_array = collect(channelview(small_img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
    img_array = collect(rawview(img_array))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)
    img_array = img_array / maximum(img_array)

    tmp_img = zeros(Float32, 1, 3, 320, 320)
    tmp_img[1, 1, :, :] = (img_array[1, :, :] .- 0.485) / 0.229
    tmp_img[1, 2, :, :] = (img_array[2, :, :] .- 0.456) / 0.224
    tmp_img[1, 3, :, :] = (img_array[3, :, :] .- 0.406) / 0.225
    input = Dict("input.1" => tmp_img)
    # # output size: (batch_size, 1, height, width)
    # result = INFERENCE_SESSION[](input)
    # mask = result["1959"]
    pred = session.onnx_session(input)["1959"][:, 1, :, :]

    ma = maximum(pred)
    mi = minimum(pred)
    pred = (pred .- mi) / (ma .- mi)
    pred = dropdims(pred; dims=1)

    mask = colorview(Gray, pred)
    mask = imresize(mask, size(img))
    return [mask]
end
