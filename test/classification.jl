y = [2, 0, 2, 2, 0, 1]
ŷ = [0, 0, 2, 2, 0, 2]
classes = [0, 1, 2]
mat = [2 0 0; 0 0 1; 1 0 2]

@testset "Confusin matrix" begin
    @test confusion_matrix(y, ŷ, classes) == mat
    @test_throws DimensionMismatch confusion_matrix(y, [1, 2])
    @test_throws DimensionMismatch confusion_matrix([0, 1, 2], ŷ)
end

@testset "Accuracy" begin
    @test accuracy(y, ŷ, classes) == 4 / 6
    @test accuracy(y, ŷ, classes) == accuracy(mat)
    @test accuracy(mat) == 4 / 6
end

@testset "Cohen's kappa" begin
    @test cohen_kappa(y, ŷ) == 0.42857142857142855
    @test cohen_kappa(y, ŷ) == cohen_kappa(ŷ, y)
    @test cohen_kappa(y, ŷ) == cohen_kappa(mat)
end

@testset "F₁ / Fᵦ score" begin
    @test f1_score(y, ŷ, classes) == 4 / 6
    @test f1_score(y, ŷ, classes) == f1_score(mat)
    @test f1_score(mat) == 4 / 6
    @test f_beta(y, ŷ, classes; β=1) == f1_score(y, ŷ, classes)
    @test f_beta(mat, β=1) == 4/6
    @test_throws DomainError f_beta(y, ŷ; β=-1)
    @test_throws DomainError f_beta(mat; β=-1)
end
