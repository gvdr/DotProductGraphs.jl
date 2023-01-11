using Documenter
using DotProductGraphs

push!(LOAD_PATH,"../src/")
makedocs(sitename="DotProductGraphs.jl",
    authors = "Giulio Valentino Dalla RIva",
    pages = [
        "Home" => "index.md",
        "Manual" => [
        "man/embeddings.md",
#        "man/dimensionality.md",
#        "man/alignments.md",
#        "man/omniembedding.md"
        ]
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
)
)

deploydocs(
    repo = "github.com/gvdr/DotProductGraphs.jl.git",
    devbranch = "main"
)