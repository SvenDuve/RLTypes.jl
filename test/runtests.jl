using RLTypes
using Test

@testset "RLTypes.jl" begin
    # Write your tests here.
    @test Acrobot() isa DiscreteEnvironment
    @test LunarLanderDiscrete() isa DiscreteEnvironment
    @test ModelParameter().train == false
end
