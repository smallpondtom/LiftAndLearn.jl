using Documenter
using LiftAndLearn
using DocumenterCitations

ENV["JULIA_DEBUG"] = "Documenter"

PAGES = [
    "Home" => "index.md",
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
    "API Reference" => "api.md",
    "Paper Reference" => "paper.md",
]

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "LiftAndLearn.jl",
    clean = true, doctest = false, linkcheck = false,
    authors = "Tomoki Koike <tkoike45@gmail.com>",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "https://github.com/smallpondtom/LiftAndLearn.jl",
        assets=String[
            "assets/citations.css",
            "assets/favicon.ico",
        ],
        analytics = "G-B2FEJZ9J99",
    ),
    modules = [
        LiftAndLearn,
    ],
    pages = PAGES,
    plugins=[bib],
)

deploydocs(
    repo = "github.com/smallpondtom/LiftAndLearn.jl.git",
    branch = "gh-pages",
    devbranch = "main",
    # Add other deployment options as needed
)