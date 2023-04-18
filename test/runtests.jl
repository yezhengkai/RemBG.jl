using RemBG
using Test
using RemBG.Images

@testset "Model type" begin
    @test isa(U2Net, RemBG.U2NetModel)
    @test isa(U2Netp, RemBG.U2NetpModel)
    @test isa(U2NetClothSeg, RemBG.U2NetClothSegModel)
    @test isa(U2NetHumanSeg, RemBG.U2NetHumanSegModel)
    @test isa(Silueta, RemBG.SiluetaModel)
    @test isa(ISNetGeneralUse, RemBG.ISNetGeneralUseModel)
end

@testset "Session type" begin
    @test isa(new_session(U2Net), RemBG.SimpleSession)
    @test isa(new_session(U2Netp), RemBG.SimpleSession)
    @test isa(new_session(U2NetClothSeg), RemBG.ClothSession)
    @test isa(new_session(U2NetHumanSeg), RemBG.SimpleSession)
    @test isa(new_session(Silueta), RemBG.SimpleSession)
    @test isa(new_session(ISNetGeneralUse), RemBG.DisSession)
end

exampledir(args...) = joinpath(@__DIR__, "..", "examples", args...)
img_animal = load(exampledir("animal-1.jpg"))
img_cloth = load(exampledir("cloth-1.jpg"))
@testset "remove" begin

    session = new_session(U2Netp)
    @test size(img_animal) == size(remove(img_animal; session=session))
    @test size(img_animal) == size(remove(img_animal; session=session, only_mask=true))

    session = new_session(ISNetGeneralUse)
    @test size(img_animal) == size(remove(img_animal; session=session))
    @test size(img_animal) == size(remove(img_animal; session=session, only_mask=true))

    session = new_session(U2NetClothSeg)
    @test size(img_cloth) .* (3, 1) ==
        size(remove(img_cloth; session=session))
    @test size(img_cloth) .* (3, 1) ==
        size(remove(img_cloth; session=session, only_mask=true))
end
