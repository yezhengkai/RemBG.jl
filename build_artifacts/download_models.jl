using Downloads

uris = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx",
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
)
mkpath("models")

@sync for uri in uris
    model_path = joinpath("models", rsplit(uri, "/")[end])
    isfile(model_path) && continue
    @async @show Downloads.download(uri, model_path)
end
