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
        "Streaming OpInf" => [
            "Overview" => "manual/streaming/Overview.md",
            "Two-Pass" => "manual/streaming/TwoPass.md",
            "One-Pass" => "manual/streaming/OnePass.md",
        ],
        "Least-Squares Solvers" => "manual/Solvers.md",
    ],
    "API Reference" => "api.md",
    "Paper Reference" => "paper.md",
]

# Objects already documented in a manual `@docs` block. The API page (`api.md`) lists every
# remaining docstring with `@autodocs`, filtering these out to avoid duplicate docs.
function manual_documented_objects(dir)
    objs = Set{Any}()
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".md") || continue
        inblock = false
        for line in eachline(joinpath(root, f))
            s = strip(line)
            if startswith(s, "```@docs")
                inblock = true
            elseif inblock && startswith(s, "```")
                inblock = false
            elseif inblock && !isempty(s)
                name = replace(first(split(s, '(')), "LiftAndLearn." => "")
                sym = Symbol(strip(name))
                isdefined(LiftAndLearn, sym) && push!(objs, getfield(LiftAndLearn, sym))
            end
        end
    end
    return objs
end
const MANUAL_DOCUMENTED = manual_documented_objects(joinpath(@__DIR__, "src", "manual"))

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename = "LiftAndLearn.jl",
    clean = true, doctest = false, linkcheck = false,
    authors = "Tomoki Koike <tkoike45@gmail.com>",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        edit_link = "https://github.com/smallpondtom/LiftAndLearn.jl",
        assets=String[
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