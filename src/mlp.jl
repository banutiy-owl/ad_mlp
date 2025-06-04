include("ad.jl")
include("operations.jl")

using LinearAlgebra

mutable struct AdamState
    m::Array{Float64}
    v::Array{Float64}
    t::Int
end

function AdamState(param_shape::Tuple)
    AdamState(zeros(param_shape), zeros(param_shape), 0)
end

function adam!(param, grad, state, lr, β1, β2, ϵ)
    if (grad===nothing)
        return
    end
    state.t += 1
    state.m .= β1 .* state.m .+ (1 .- β1) .* grad
    state.v .= β2 .* state.v .+ (1 .- β2) .* (grad .^ 2)
    m_hat = state.m ./ (1 .- β1^state.t)
    v_hat = state.v ./ (1 .- β2^state.t)
    param.output .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)
end


function dense(w::GraphNode, x::GraphNode, activation)
    return activation(w*x)
end

function xavier_init(fan_in::Int, fan_out::Int)
    limit = sqrt(6.0 / (fan_in + fan_out))
    return rand(Float32, fan_out, fan_in) .* (2f0 * limit) .- limit
end

function create_mlp(input_dim::Int, hidden_dim::Int, output_dim::Int)
    Wh = Variable(xavier_init(input_dim, hidden_dim), name="Wh")
    Wo = Variable(xavier_init(hidden_dim, output_dim), name="Wo")
    return Wh, Wo
end


function compute_loss(y::GraphNode, ŷ::GraphNode)

    loss_array = Constant([0]).- (y .* log.(ŷ) .+ (Constant(1) .- y) .* log.(Constant(1) .- ŷ))
    loss_val = sum(loss_array)

    return loss_val
end



function build_graph(x_var, y_var, Wh, Wo, dense, σ)
    x̂ = dense(Wh, x_var, reLU); x̂.name = "x̂"
    ŷ = dense(Wo, x̂, σ); ŷ.name = "ŷ"
    loss_node = compute_loss(y_var, ŷ)
    graph = topological_sort(loss_node)
    return graph, x_var, y_var, ŷ
end


function train_step!(x_var, y_var, Wh, Wo, lr, states, β1, β2, ϵ, graph)
    loss_node = last(graph)  
    forward!(graph)
    backward!(graph)
    adam!(Wh, Wh.gradient, states[1], lr, β1, β2, ϵ)
    adam!(Wo, Wo.gradient, states[2], lr, β1, β2, ϵ)
    forward!(graph)
    return loss_node.output[1]
end


function test!(X_test, y_test, graph, x_var, y_var, ŷ)
    _, n_samples = size(X_test)
    total_loss = 0.0
    correct = 0
    for i in 1:n_samples
        x_var.output .= X_test[:, i]
        y_var.output .= y_test[:, i]
        loss_val = forward!(graph)
        total_loss += loss_val[1]
        predicted = ŷ.output[1] > 0.5 ? 1.0 : 0.0
        if predicted == y_test[1, i]
            correct += 1
        end
    end
    avg_loss = total_loss / n_samples
    accuracy = correct / n_samples * 100
    return avg_loss, accuracy
end