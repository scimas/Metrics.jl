"""
    confusion_matrix(y, ŷ, classes=unique([y; ŷ]))

Creates the confusion matrix for targets y and predictions ŷ. classes
are the unique target values. It is recommended to specify this explicitly to
avoid ambiguities about the order of the target labels in the confusion matrix.
"""
function confusion_matrix(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    length(y) == length(ŷ) || throw(DimensionMismatch("targets and predictions should have same number of elements"))
    mat = zeros(Int, length(classes), length(classes))
    @inbounds for (i, j) in Iterators.product(1:length(classes), 1:length(classes))
        mat[i, j] = sum(ŷ[y .== classes[j]] .== classes[i])
    end
    mat
end

"""
    accuracy(y, ŷ, classes=unique([y; ŷ]))

Calculates accuracy score for targets y and predictions ŷ. classes are the
unique target values. It is mathematically equiavalent to #(true positives) ÷
#(observations).
"""
function accuracy(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    conf_mat = confusion_matrix(y, ŷ, classes)
    accuracy(conf_mat)
end

"""
    accuracy(conf_mat)

Calculates accuracy score using the confusion matrix conf_mat.
"""
function accuracy(conf_mat::Matrix{<:Integer})
    tr(conf_mat) / sum(sum(conf_mat; dims=1))
end

"""
    cohen_kappa(y, ŷ, classes=unique([y; ŷ]))

Calculates Cohen's Kappa statistic for y and ŷ. classes are the unique target
values. It is mathematically equiavalent to

κ = (pₒ - pₑ)/(1 - pₑ)

where pₒ and pₑ are the observed and expected probabilities of agreement,
respectively.
"""
function cohen_kappa(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    conf_mat = confusion_matrix(y, ŷ, classes)
    cohen_kappa(conf_mat)
end

"""
    cohen_kappa(conf_mat)

Calculates Cohen's Kappa statistic using the confusion matrix conf_mat.
"""
function cohen_kappa(conf_mat::Matrix{<:Integer})
    pₑ = only(sum(conf_mat; dims=1) * sum(conf_mat; dims=2) / sum(sum(conf_mat; dims=1))^2)
    (accuracy(conf_mat) - pₑ) / (1 - pₑ)
end

"""
    f_beta(y, ŷ, classes=unique([y; ŷ]); β=1)

Calculates Fᵦ score for the targets y, predictions ŷ and β value. classes are
the unique target values. It is mathematically equiavalent to

Fᵦ = (1 + (β ^ 2 * FN + FP) / ((1 + β ^ 2) * TP)) ^ -1

where TP, FP and FN are true positives, false positives and false negatives
respectively. Currently only supports global score calculation, per class
averaging is not available.
"""
function f_beta(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]); β=1)
    β >= 0 || throw(DomainError(β, "β must be non-negative for Fᵦ score"))
    conf_mat = confusion_matrix(y, ŷ, classes)
    f_beta(conf_mat; β=β)
end

"""
    f_beta(conf_mat; β=1)

Calculates Fᵦ score using the confusion matrix conf_mat for the given β value.
"""
function f_beta(conf_mat::Matrix{<:Integer}; β=1)
    β >= 0 || throw(DomainError(β, "β must be non-negative for Fᵦ score"))
    tp = tr(conf_mat)
    fp = sum(sum(conf_mat; dims=2) - diag(conf_mat))
    fn = sum(permutedims(sum(conf_mat; dims=1)) - diag(conf_mat))
    inv(1 + (β ^ 2 * fn + fp) / ((1 + β ^ 2) * tp))
end
