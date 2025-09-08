using Documenter
using SizeCheck

makedocs(
    sitename = "SizeCheck",
    format = Documenter.HTML(sidebar_sitename=false),
    pages =[],
    modules = [SizeCheck]
)

deploydocs(repo = "github.com/samanklesaria/SizeCheck.jl.git")
