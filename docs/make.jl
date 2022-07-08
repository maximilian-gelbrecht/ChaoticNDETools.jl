using ChaoticNDETools
using Documenter

DocMeta.setdocmeta!(ChaoticNDETools, :DocTestSetup, :(using ChaoticNDETools); recursive=true)

makedocs(;
    modules=[ChaoticNDETools],
    authors="Maximilian Gelbrecht <maximilian.gelbrecht@posteo.de> and contributors",
    repo="https://github.com/maximilian-gelbrecht/ChaoticNDETools.jl/blob/{commit}{path}#{line}",
    sitename="ChaoticNDETools.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://maximilian-gelbrecht.github.io/ChaoticNDETools.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maximilian-gelbrecht/ChaoticNDETools.jl",
    devbranch="main",
)
