using Documenter
using LiftAndLearn
using DocumenterCitations

PAGES = [
    "Home" => "index.md",
    "API Reference" => "api.md",
    "Paper Reference" => "paper.md",
    "Manual" => [
        "Utilities" => "manual/Utilities.md",
        "Options" => "manual/Options.md",
        "Lift" => "manual/Lift.md",
        "Intrusive" => "manual/Intrusive.md",
        "Non-Intrusive" => [
            "Standard OpInf" => "manual/nonintrusive/LS.md",
            "Lift And Learn" => "manual/nonintrusive/LnL.md",
            "Energy Preserving" => "manual/nonintrusive/EPOpInf.md",
        ],
    ],
    "Models" => [
        "Heat1D" => "models/Heat1D.md",
        "Burgers" => "models/Burgers.md",
        "FHN" => "models/FHN.md",
        "KS" => "models/KS.md",
    ],
]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "LiftAndLearn.jl",
    authors = "Tomoki Koike",
    modules = [LiftAndLearn],
    clean = true, doctest = false, linkcheck = false,
    format = Documenter.HTML(
        assets=String["assets/citations.css"]
    ),
    pages = PAGES,
    plugins=[bib],
)

deploydocs(
    repo = "github.com/smallpondtom/LiftAndLearn.jl.git",
    target = "build",
    push_preview = true,
    # devbranch = "main"
    # Add other deployment options as needed
)