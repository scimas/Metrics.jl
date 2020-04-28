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

Calculates accuracy score for targets y and predictions ŷ. classes are the unique
target values. It is recommended to specify this explicitly to avoid ambiguities
about the order of the target labels in the confusion matrix. It is
mathematically equiavalent to #(true positives) ÷ #(observations).
"""
function accuracy(y::AbstractVector, ŷ::AbstractVector, classes::AbstractVector=unique([y; ŷ]))
    conf_mat = confusion_matrix(y, ŷ, classes)
    sum(conf_mat[1:length(classes) + 1:length(classes) * length(classes)]) / length(y)
end

"""
    accuracy(conf_mat)

Calculates accuracy score using the confusion matrix conf_mat.
"""
function accuracy(conf_mat::Matrix)
    sum(conf_mat[1:size(conf_mat, 1) + 1:size(conf_mat, 1) * size(conf_mat, 1)]) / sum(sum(conf_mat; dims=1))
end