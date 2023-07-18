using RLTypes
using Documenter

DocMeta.setdocmeta!(RLTypes, :DocTestSetup, :(using RLTypes); recursive=true)

makedocs(;
    modules=[RLTypes],
    authors="Sven Duve <svenduve@gmail.com> and contributors",
    repo="https://github.com/SvenDuve/RLTypes.jl/blob/{commit}{path}#{line}",
    sitename="RLTypes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://SvenDuve.github.io/RLTypes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SvenDuve/RLTypes.jl",
    devbranch="main",
)
