"""
    confusion_matrix(y, ŷ, classes=unique([y; ŷ]))

Creates the confusion matrix for targets y and predictions ŷ. `classes` are the
unique target values. It is recommended to specify this explicitly to avoid
ambiguities about the order of the target labels in the confusion matrix.
"""
function confusion_matrix(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    length(y) == length(ŷ) || throw(DimensionMismatch("targets and predictions must have same number of elements"))
    mat = Matrix{Int64}(undef, length(classes), length(classes))
    @inbounds for (i, j) in Iterators.product(1:length(classes), 1:length(classes))
        mat[i, j] = count(ŷ[y .== classes[i]] .== classes[j])
    end
    mat
end

"""
    accuracy(y, ŷ, classes=unique([y; ŷ]))
    accuracy(conf_mat)

Calculates accuracy score for targets y and predictions ŷ or the confusion
matrix `conf_mat`. `classes` are the unique target values. It is mathematically
equiavalent to

accuracy = TP / N

where TP and N are true positives and total number of observations respectively.
"""
function accuracy(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    conf_mat = confusion_matrix(y, ŷ, classes)
    accuracy(conf_mat)
end

function accuracy(conf_mat::Matrix{<:Integer})
    tr(conf_mat) / sum(conf_mat)
end

"""
    accuracy_per_class(y, ŷ, classes=unique([y; ŷ]))
    accuracy_per_class(conf_mat)

Calculates accuracy score for targets y and predictions ŷ or the confusion
matrix conf_mat for each class. `classes` are the unique target values. It is
mathematically equiavalent to

accuracyᵢ = TPᵢ / Nᵢ

where TPᵢ and Nᵢ are true positives and total number of observations for class
i, respectively.
"""
function accuracy_per_class(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    conf_mat = confusion_matrix(y, ŷ, classes)
    accuracy_per_class(conf_mat)
end

function accuracy_per_class(conf_mat::Matrix{<:Integer})
    diag(conf_mat) ./ sum(conf_mat; dims=2)
end

@doc raw"""
    cohen_kappa(y, ŷ, classes=unique([y; ŷ]))
    cohen_kappa(conf_mat)

Calculates Cohen's Kappa statistic for y and ŷ or using the confusion matrix
conf_mat. classes are the unique target values. It is mathematically equiavalent
to

``κ = \frac{pₒ - pₑ}{1 - pₑ}``

where pₒ and pₑ are the observed and expected probabilities of agreement,
respectively.
"""
function cohen_kappa(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    conf_mat = confusion_matrix(y, ŷ, classes)
    cohen_kappa(conf_mat)
end

function cohen_kappa(conf_mat::Matrix{<:Integer})
    pₑ = only(sum(conf_mat; dims=1) * sum(conf_mat; dims=2)) / sum(conf_mat)^2
    (accuracy(conf_mat) - pₑ) / (1 - pₑ)
end

"""
    f_beta(y, ŷ, classes=unique([y; ŷ]); β, mode=:binary)
    f_beta(conf_mat; β, mode=:binary)

Calculates Fᵦ score for the targets y, predictions ŷ or using the confusion
matrix conf_mat and β value. classes are the unique target values. It is
mathematically equiavalent to

Fᵦ = (1 + (β ^ 2 * FN + FP) / ((1 + β ^ 2) * TP)) ^ -1

where TP, FP and FN are true positives, false positives and false negatives
respectively. Currently only supports global score calculation, per class
averaging is not available.

`mode`s:
 - :binary : Assumes binary classification
 - :micro  : Intermediate metrics are calculated globally across all classes
 - :macro  : Fᵦ is calculated for each class and then averaged
"""
function f_beta(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]); β, mode=:binary)
    β >= 0 || throw(DomainError(β, "β must be non-negative for Fᵦ score"))
    conf_mat = confusion_matrix(y, ŷ, classes)
    f_beta(conf_mat; β, mode)
end

function f_beta(conf_mat::Matrix{<:Integer}; β, mode=:binary)
    β >= 0 || throw(DomainError(β, "β must be non-negative for Fᵦ score"))
    _f_beta(Val(mode), conf_mat; β)
end

function _f_beta(::Val{:binary}, conf_mat::Matrix{<:Integer}; β)
    size(conf_mat) == (2, 2) || throw(ArgumentError("confusion matrix must be 2 × 2 for the binary case"))
    println("Here")
    tp = tr(conf_mat)
    fp = conf_mat[2, 1]
    fn = conf_mat[1, 2]
    (1 + β^2) * tp / ((1 + β^2) * tp + β^2 * fn + fp)
end

function _f_beta(::Val{:micro}, conf_mat::Matrix{<:Integer}; β)
    tp = tr(conf_mat)
    fp = fn = sum(conf_mat) - tp
    (1 + β^2) * tp / ((1 + β^2) * tp + β^2 * fn + fp)
end

function _f_beta(::Val{:macro}, conf_mat::Matrix{<:Integer}; β)
    mean(f_beta_per_class(conf_mat; β))
end

"""
    f_beta_per_class(y, ŷ, classes=unique([y; ŷ]); β)
    f_beta_per_class(conf_mat; β)

Calculates Fᵦ score per class.
"""
function f_beta_per_class(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]); β)
    β >= 0 || throw(DomainError(β, "β must be non-negative for Fᵦ score"))
    conf_mat = confusion_matrix(y, ŷ, classes)
    f_beta_per_class(conf_mat; β)
end

function f_beta_per_class(conf_mat::Matrix{<:Integer}; β)
    β >= 0 || throw(DomainError(β, "β must be non-negative for Fᵦ score"))
    tp = diag(conf_mat)
    fp = permutedims(sum(conf_mat; dims=1)) .- tp
    fn = sum(conf_mat; dims=2) .- tp
    @. (1 + β^2) * tp / ((1 + β^2) * tp + β^2 * fn + fp)
end

"""
    f1_score(y, ŷ, classes=unique([y; ŷ]); mode=:binary)
    f1_score(conf_mat; mode=:binary)

Calculates Fᵦ score with β=1.
"""
f1_score(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]); mode=:binary) = f_beta(y, ŷ, classes; β=1, mode)
f1_score(conf_mat::Matrix{<:Integer}; mode=:binary) = f_beta(conf_mat; β=1, mode)

"""
    f1_per_class(y, ŷ, classes=unique([y; ŷ]); β)
    f1_per_class(conf_mat; β)

Calculates F₁ score per class.
"""
f1_per_class(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ])) = f_beta_per_class(y, ŷ, classes; β=1)
f1_per_class(conf_mat::Matrix{<:Integer}) = f_beta_per_class(conf_mat; β=1)
