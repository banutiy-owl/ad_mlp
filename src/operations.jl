import Base: ^, sin, *, sum, max
import LinearAlgebra: mul!

Base.Broadcast.broadcasted(^, x::GraphNode, n::GraphNode) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = x .^ n
backward(::BroadcastedOperator{typeof(^)}, x, n, g) = (
    g .* n .* (x .^ (n .- 1)),
    g .* log.(abs.(x)) .* (x .^ n)
)

Base.Broadcast.broadcasted(sin, x::GraphNode) = BroadcastedOperator(sin, x)
forward(::BroadcastedOperator{typeof(sin)}, x) = sin.(x)
backward(::BroadcastedOperator{typeof(sin)}, x, g) = (g .* cos.(x),)

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = (g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = (
    g .* y,
    g .* x
)

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(op::BroadcastedOperator{typeof(log)}, x) = log.(x)
backward(op::BroadcastedOperator{typeof(log)}, x, g) = g ./ x

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = (g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = (g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = (fill(g, size(x)),)

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(::BroadcastedOperator{typeof(/)}, x, y, g) = (
    g ./ y,
    -g .* x ./ (y .^ 2)
)

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = (
    g .* (x .> y),
    g .* (y .> x)
)

σ(x) = BroadcastedOperator(σ, x)
forward(::BroadcastedOperator{typeof(σ)}, x) = 1.0 ./ (1.0 .+ exp.(-x))
backward(node::BroadcastedOperator{typeof(σ)}, x, g) = begin
    y = node.output
    (g .* y .* (1 .- y),)
end

reLU(x) = BroadcastedOperator(reLU, x)
forward(::BroadcastedOperator{typeof(reLU)}, x) = max.(0, x)
backward(::BroadcastedOperator{typeof(reLU)}, x, g) = (
    g .* (x .> 0),
)