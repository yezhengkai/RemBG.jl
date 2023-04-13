#=
References:
- https://www.juliafordatascience.com/artifacts/
- https://github.com/simeonschaub/ArtifactUtils.jl
- https://github.com/FluxML/Metalhead.jl
- https://github.com/FluxML/MetalheadWeights
- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
- https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection
=#

using Pkg.Artifacts
import HuggingFaceHub as HF

model_dir = abspath(joinpath(@__DIR__, "models"))
build_dir = abspath(joinpath(@__DIR__, "build"))
artifact_toml = abspath(joinpath(@__DIR__, "..", "Artifacts.toml"))

# Clean up prior builds.
ispath(build_dir) && rm(build_dir; recursive=true, force=true)
mkdir(build_dir)

# Create model repo
repo_id = "yezhengkai/RemBG"
HF.create(HF.Model(id=repo_id, private=false))  # BUG
repo = HF.info(HF.Model(id=repo_id))

# Package up models
for model_path in readdir(model_dir; join=true)
    model_name = splitext(basename(model_path))[1]
    artifact_filename = "$model_name.tar.gz"
    artifact_filepath = joinpath(build_dir, artifact_filename)
    @info "Creating:" model_name

    product_hash = create_artifact() do artifact_dir
        cp(model_path, joinpath(artifact_dir, basename(model_path)); force=true)
    end
    @info "Create artifact:" product_hash

    download_hash = archive_artifact(product_hash, joinpath(build_dir, artifact_filename))
    @info "Archive artifact:" download_hash
    remote_url = "https://huggingface.co/$repo_id/resolve/main/$artifact_filename"

    @info "Upload to Hugging Face models" artifact_filename remote_url
    HF.file_upload(repo, artifact_filename, artifact_filepath)  # BUG

    bind_artifact!(
        artifact_toml,
        model_name,
        product_hash,
        force=true,
        lazy=true,
        download_info=Tuple[(remote_url, download_hash)]
    )
end
