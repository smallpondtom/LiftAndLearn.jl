using Documenter

push!(LOAD_PATH,"../src/")

makedocs(
    modules = [LiftAndLearn],
    sitename = "LiftAndLearn.jl",
    authors = ["Tomoki Koike"],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API" => "api.md"
        # Add more pages as needed
    ]
)

deploydocs(
    repo = "github.com/smallpondtom/LiftAndLearn.jl",
    devbranch = "main"
    # Add other deployment options as needed
)