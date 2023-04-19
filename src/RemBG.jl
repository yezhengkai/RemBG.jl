module RemBG

import ONNXRunTime as OX
using Images
using Artifacts
using LazyArtifacts

include("session.jl")
export U2Net,
    U2Netp,
    U2NetClothSeg,
    U2NetHumanSeg,
    Silueta,
    ISNetGeneralUse,
    new_session,
    set_onnx_execution_provider,
    get_onnx_execution_provider

include("bg.jl")
export remove

include("utils.jl")

end
