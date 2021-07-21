module Metrics
using LinearAlgebra: tr, diag
using Statistics: mean

export
    confusion_matrix,
    accuracy, accuracy_per_class,
    cohen_kappa,
    f_beta, f_beta_per_class,
    f1_score, f1_per_class

include("classification.jl")

end # module
