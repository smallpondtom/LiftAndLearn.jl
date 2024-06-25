include("Heat1D.jl")
include("Burgers.jl")
include("FitzHughNagumo.jl")
include("KuramotoSivashinsky.jl")
include("FisherKPP.jl")
include("Heat2D.jl")
include("ChafeeInfante.jl")
include("AllenCahn.jl")
using .Heat1D: Heat1DModel
using .Burgers: BurgersModel
using .FitzHughNagumo: FitzHughNagumoModel
using .KuramotoSivashinsky: KuramotoSivashinskyModel
using .FisherKPP: FisherKPPModel
using .Heat2D: Heat2DModel
using .ChafeeInfante: ChafeeInfanteModel
using .AllenCahn: AllenCahnModel
