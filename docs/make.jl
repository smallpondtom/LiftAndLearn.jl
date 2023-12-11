using Documenter

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "MyPackage Documentation",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        # Add more pages as needed
    ]
)

# deploydocs(
#     repo = "github.com/smallpondtom/LiftAndLearn.jl.git",  # Replace with your repository
#     devbranch = "main"
#     # Add other deployment options as needed
# )