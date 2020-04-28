module Metrics
using LinearAlgebra: tr, diag

export
    confusion_matrix,
    accuracy,
    cohen_kappa

include("classification.jl")

end # module
