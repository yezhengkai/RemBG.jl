U2NET_ONNX_PATH = joinpath(artifact"u2net", "u2net.onnx")
U2NETP_ONNX_PATH = joinpath(artifact"u2netp", "u2netp.onnx")
U2NET_CLOTH_SEG_ONNX_PATH = joinpath(artifact"u2net_cloth_seg", "u2net_cloth_seg.onnx")
U2NET_HUMAN_SEG_ONNX_PATH = joinpath(artifact"u2net_human_seg", "u2net_human_seg.onnx")
SILUETA_ONNX_PATH = joinpath(artifact"silueta", "silueta.onnx")
ISNET_GENERAL_USE_ONNX_PATH = joinpath(
    artifact"isnet-general-use", "isnet-general-use.onnx"
)
const EXECUTION_PROVIDER = Ref{Symbol}(:cpu)

abstract type MattingModel end
struct U2NetModel <: MattingModel end
struct U2NetpModel <: MattingModel end
struct U2NetClothSegModel <: MattingModel end
struct U2NetHumanSegModel <: MattingModel end
struct SiluetaModel <: MattingModel end
struct ISNetGeneralUseModel <: MattingModel end

"""`U2NetModel` object. `U2NetModel` is a subtype of `MattingModel`."""
const U2Net = U2NetModel()

"""`U2NetpModel` object. `U2NetpModel` is a subtype of `MattingModel`."""
const U2Netp = U2NetpModel()

"""`U2NetClothSegModel` object. `U2NetClothSegModel` is a subtype of `MattingModel`."""
const U2NetClothSeg = U2NetClothSegModel()

"""`U2NetHumanSegModel` object. `U2NetHumanSegModel` is a subtype of `MattingModel`."""
const U2NetHumanSeg = U2NetHumanSegModel()

"""`SiluetaModel` object. `SiluetaModel` is a subtype of `MattingModel`."""
const Silueta = SiluetaModel()

"""`ISNetGeneralUseModel` object. `ISNetGeneralUseModel` is a subtype of `MattingModel`."""
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

"""
    get_onnx_execution_provider()

Get the current execution provider for the ONNX inference session.
"""
function get_onnx_execution_provider()
    return EXECUTION_PROVIDER[]
end

"""
    set_onnx_execution_provider(execution_provider::Symbol)

Set the execution provider for ONNX inference session. Now the "execution_provider" only accepts ":cpu" and ":cuda".
"""
function set_onnx_execution_provider(execution_provider::Symbol)
    if execution_provider === :cpu
        EXECUTION_PROVIDER[] = execution_provider
    elseif execution_provider === :cuda
        EXECUTION_PROVIDER[] = execution_provider
    else
        error("Unsupported execution_provider $execution_provider")
    end
end

"""
    new_session(::MattingModel)

Constructs an `InferenceSession` object by input type.

# Examples
```julia
session = new_session(U2Net)
```
"""
function new_session(::U2NetModel)
    return SimpleSession(
        "U2Net", OX.load_inference(U2NET_ONNX_PATH; execution_provider=EXECUTION_PROVIDER[])
    )
end
function new_session(::U2NetpModel)
    return SimpleSession(
        "U2Netp",
        OX.load_inference(U2NETP_ONNX_PATH; execution_provider=EXECUTION_PROVIDER[]),
    )
end
function new_session(::U2NetClothSegModel)
    return ClothSession(
        "U2NetClothSeg",
        OX.load_inference(
            U2NET_CLOTH_SEG_ONNX_PATH; execution_provider=EXECUTION_PROVIDER[]
        ),
    )
end
function new_session(::U2NetHumanSegModel)
    return SimpleSession(
        "U2NetHumanSeg",
        OX.load_inference(
            U2NET_HUMAN_SEG_ONNX_PATH; execution_provider=EXECUTION_PROVIDER[]
        ),
    )
end
function new_session(::SiluetaModel)
    return SimpleSession(
        "Silueta",
        OX.load_inference(SILUETA_ONNX_PATH; execution_provider=EXECUTION_PROVIDER[]),
    )
end
function new_session(::ISNetGeneralUseModel)
    return DisSession(
        "ISNetGeneralUse",
        OX.load_inference(
            ISNET_GENERAL_USE_ONNX_PATH; execution_provider=EXECUTION_PROVIDER[]
        ),
    )
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
    # output size: (batch_size, 1, 320, 320)
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
    # output size: (batch_size, 1, 1024, 1024)
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

function predict(session::ClothSession, img::Matrix{<:Colorant})

    # Preprocessing
    small_img = imresize(img, 768, 768)
    img_array = copy(channelview(small_img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
    img_array = copy(rawview(img_array))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)
    img_array = img_array / maximum(img_array)
    tmp_img = zeros(Float32, 1, 3, 768, 768)  # input size: (batch_size, 3, 768, 768)
    tmp_img[1, 1, :, :] = (img_array[1, :, :] .- 0.485) / 0.229
    tmp_img[1, 2, :, :] = (img_array[2, :, :] .- 0.456) / 0.224
    tmp_img[1, 3, :, :] = (img_array[3, :, :] .- 0.406) / 0.225

    # Predict
    input = Dict(session.onnx_session.input_names[1] => tmp_img)
    # output size: (batch_size, 4, 768, 768)
    # The 4 channels represent in order "Background", "Upper Body Clothes", "Lower Body Clothes" and "Full Body Clothes".
    pred = session.onnx_session(input)[session.onnx_session.output_names[1]]
    pred = softmax(pred, 2)
    pred = dropdims(pred; dims=1) # size: (4, 768, 768)

    mask = imresize(pred, 4, size(img)...)
    mask_upper = colorview(Gray, mask[2, :, :])
    mask_lower = colorview(Gray, mask[3, :, :])
    mask_full = colorview(Gray, mask[4, :, :])

    return [mask_upper, mask_lower, mask_full]
end
