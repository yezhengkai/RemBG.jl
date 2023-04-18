using RemBG
using Documenter

DocMeta.setdocmeta!(RemBG, :DocTestSetup, :(using RemBG); recursive=true)

makedocs(;
    modules=[RemBG],
    authors="Zheng-Kai Ye <supon3060@gmail.com> and contributors",
    repo="https://github.com/yezhengkai/RemBG.jl/blob/{commit}{path}#{line}",
    sitename="RemBG.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yezhengkai.github.io/RemBG.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/yezhengkai/RemBG.jl", devbranch="main")
