using Documenter
using LiftAndLearn

PAGES = [
    "Home" => "index.md",
    "API Reference" => "api.md",
    "Manual" => [
        "Utilities" => "manual/Utilities.md",
        "Options" => "manual/Options.md",
        "Models" => "manual/Models.md",
        "Optimizers" => "manual/Optimizers.md",
        "Lift" => "manual/Lift.md",
        "Learn" => "manual/Learn.md",
        "Intrusive ROM" => "manual/Intrusive_ROM.md",
    ],
    "Examples" => [
        "Heat1D" => "examples/Heat1D.md",
        "Burgers" => "examples/Burgers.md",
        "FHN" => "examples/FHN.md",
        "KS" => "examples/KS.md",
    ],
]

makedocs(
    sitename = "LiftAndLearn.jl",
    authors = "Tomoki Koike",
    modules = [LiftAndLearn],
    clean = true, doctest = false, linkcheck = true,
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        # "API" => "api.md",
        # Add more pages as needed
    ]
)

# deploydocs(
#     repo = "github.com/smallpondtom/LiftAndLearn.jl.git",
#     devbranch = "main"
#     # Add other deployment options as needed
# )