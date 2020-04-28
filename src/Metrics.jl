module Metrics
using LinearAlgebra: tr, diag

export
    confusion_matrix,
    accuracy,
    cohen_kappa,
    f_beta,
    f1_score

include("classification.jl")

end # module
