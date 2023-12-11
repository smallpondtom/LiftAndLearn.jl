using Documenter

makedocs(sitename="My Documentation")

makedocs(
    sitename = "MyPackage Documentation",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        # Add more pages as needed
    ]
)

# deploydocs(
#     repo = "github.com/username/MyPackage.jl.git",  # Replace with your repository
#     # Add other deployment options as needed
# )