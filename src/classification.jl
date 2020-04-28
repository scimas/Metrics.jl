function confusion_matrix(y::AbstractVector, ŷ::AbstractVector, classes=unique(y))
    length(y) == length(ŷ) || throw(DimensionMismatch("targets and predictions should have same number of elements"))
    mat = zeros(Int, length(classes), length(classes))
    @inbounds for i in 1:length(classes)
        for j in 1:length(classes)
            mat[i, j] = sum(ŷ[y .== classes[j]] .== classes[i])
        end
    end
    mat
end
