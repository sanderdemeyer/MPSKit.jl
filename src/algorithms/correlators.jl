"""
    correlator(ψ, O1, O2, i, j)
    correlator(ψ, O12, i, j)

Compute the 2-point correlator <ψ|O1[i]O2[j]|ψ> for inserting `O1` at `i` and `O2` at `j`.
Also accepts ranges for `j`.
"""
function correlator end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int, j::Int)
    return first(correlator(state, O₁, O₂, i, j:j))
end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, i::Int,
                    js::AbstractRange{Int})
    first(js) > i || @error "i should be smaller than j ($i, $(first(js)))"
    S₁ = _firstspace(O₁)
    S₁ == oneunit(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
    S₂ = _lastspace(O₂)
    S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))

    G = similar(js, scalartype(state))
    U = Tensor(ones, S₁)

    @tensor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * O₁[1 2; 4 -2] *
                             conj(state.AC[i][3 2; -1])
    ctr = i + 1

    for (k, j) in enumerate(js)
        if j > ctr
            Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)])
        end
        G[k] = @tensor Vₗ[2; 3 5] * state.AR[j][5 6; 7] * O₂[3 4; 6 1] * U[1] *
                       conj(state.AR[j][2 4; 7])
        ctr = j
    end
    return G
end

# function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle::MPOTensor, i::Int, js::AbstractRange{Int})
#     first(js) > i || @error "i should be smaller than j ($i, $(first(js)))"
#     S₁ = _firstspace(O₁)
#     S₁ == oneunit(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
#     S₂ = _lastspace(O₂)
#     S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))

#     G = similar(js, scalartype(state))
#     U = Tensor(ones, S₁)

#     @tensor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * O₁[1 2; 4 -2] *
#                 conj(state.AC[i][3 2; -1])
#     ctr = i + 1

#     for (k, j) in enumerate(js)
#         if j > ctr
#             println("j is $(j)")
#             Vₗ = Vₗ * TransferMatrix(state.AR[ctr:(j - 1)], fill(middle, j-i-1), state.AR[ctr:(j - 1)])
#         end
#         G[k] = @tensor Vₗ[2; 3 5] * state.AR[j][5 6; 7] * O₂[3 4; 6 1] * U[1] * conj(state.AR[j][2 4; 7])
#         ctr = j
#     end
#     return G
# end

function correlator(state::AbstractMPS, O₁₂::AbstractTensorMap{S,2,2}, i::Int, j) where {S}
    O₁, O₂ = decompose_localmpo(add_util_leg(O₁₂))
    return correlator(state, O₁, O₂, i, j)
end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle, i::Int, N::Int)
    S₁ = _firstspace(O₁)
    S₁ == oneunit(S₁) || throw(ArgumentError("O₁ should start with a trivial leg."))
    S₂ = _lastspace(O₂)
    S₂ == S₁' || throw(ArgumentError("O₂ should end with a trivial leg."))

    G = similar(1:N, scalartype(state))
    U = Tensor(ones, S₁)

    @tensor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * O₁[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
    @tensor Vᵣ[-1 -2; -3] := state.AC[i][-1 2; 1] * O₂[-2 4; 2 3] * U[3] * conj(state.AC[i][-3 4; 1])

    for j = i-1:-1:1 # j < i ==> factor -i
        if j < i-1
            @tensor Vᵣ[-1 -2; -3] := (-2im) *Vᵣ[1 -2; 4] * (state.AL[j+1])[-1 2; 1] * middle[3; 2] * conj((state.AL[j+1])[-3 3; 4])
        end
        G[j] = 1im*(@tensor Vᵣ[4; 5 7] * state.AL[j][1 2; 4] * O₁[3 6; 2 5] * conj(U[3]) * conj(state.AL[j][1 6; 7]))
    end
    G[i] = @tensor (state.AC[i])[1 2; 8] * O₂[5 4; 2 3] * U[3] * O₁[6 7; 4 5] * conj(U[6]) * conj((state.AC[i])[1 7; 8])
    for j = i+1:N # j > i ==> factor i and conjugate
        if j > i+1
            @tensor Vₗ[-1 -2; -3] := (2im) * Vₗ[1 -2; 4] * (state.AR[j-1])[4 5; -3] * middle[3; 5] * conj((state.AR[j-1])[1 3; -1])
        end
        G[j] = -1im*(@tensor Vₗ[2; 3 5] * state.AR[j][5 6; 7] * O₂[3 4; 6 1] * U[1] * conj(state.AR[j][2 4; 7]))
    end
    return G
end

function correlator_old(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle, left::MPOTensor, right::MPOTensor, i::Int, N::Int)
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)
    
    U_space₁ = Tensor(ones, space(O₁)[1])
    U_space₂ = Tensor(ones, space(O₂)[4])
    
    unit_left = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], space(left)[1] ⊗ pspace, pspace ⊗ space(left)[1])
    unit_right = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], conj(space(right)[4]) ⊗ pspace, pspace ⊗ conj(space(right)[4]))
    
    G = similar(1:N, scalartype(state))
    
    for op_index = 1:N-1 # divide by N-1?
    
        println("op_index = $(op_index) of N = $(N)")
        H_list = fill(unit_left, N)
    
        H_list[op_index] = left
        H_list[op_index+1] = right
    
        U_H_left = Tensor(ones, space(H_list[i])[1])
        U_H_right = Tensor(ones, space(H_list[i])[4])
    
        for i₂ = op_index+2:N
            H_list[i₂] = unit_right
        end
    
        @tensor Vₗ[-1 -2; -3 -4] := state.AC[i][1 2; -4] * conj(U_space₁[3]) * O₁[3 4; 2 -3] * H_list[i][5 6; 4 -2] * conj(U_H_left[5]) * conj(state.AC[i][1 6; -1])
        @tensor Vᵣ[-1 -2; -3 -4] := state.AC[i][-1 2; 1] * H_list[i][-2 4; 2 3] * conj(U_H_right[3]) * O₂[-3 6; 4 5] * conj(U_space₂[5]) * conj(state.AC[i][-4 6; 1])
    
        for j = i-1:-1:1 # j < i ==> factor -i
            # global Vᵣ
            if j < i-1
                @tensor Vᵣ[-1 -2; -3 -4] := (-2im) * Vᵣ[1 4; -3 6] * (state.AL[j+1])[-1 2; 1] * middle[3; 2] * H_list[j+1][-2 5; 3 4] * conj((state.AL[j+1])[-4 5; 6])
            end
            U_H_left = Tensor(ones, space(H_list[j])[1])
            G[j] += 1im*(@tensor Vᵣ[4 8; 5 10] * state.AL[j][1 2; 4] * O₁[3 6; 2 5] * conj(U_space₁[3]) * H_list[j][7 9; 6 8] * conj(U_H_left[7]) * conj(state.AL[j][1 9; 10]))
        end
        U_H_left = Tensor(ones, space(H_list[i])[1])
        U_H_right = U_H_right = Tensor(ones, space(H_list[i+1])[4])
        G[i] += @tensor (state.AC[i])[1 2; 10] * O₂[5 4; 2 3] * conj(U_space₂[3]) * H_list[i][6 7; 4 12] * conj(U_H_left[6]) * O₁[8 9; 7 5] * conj(U_space₁[8]) * conj((state.AC[i])[1 9; 15]) * (state.AR[i+1])[10 11; 16] * H_list[i+1][12 14; 11 13] * conj(U_H_right[13]) * conj((state.AR[i+1])[15 14; 16])
        for j = i+1:N # j > i ==> factor i and conjugate
            # global Vₗ
            # global U_H_right
            if j > i+1
                @tensor Vₗ[-1 -2; -3 -4] := (2im) * Vₗ[1 4; -3 1] * (state.AR[j-1])[1 2; -4] * middle[3; 2] * H_list[j-1][4 5; 3 -2] * conj((state.AR[j-1])[1 5; -1])
            end
            U_H_right = Tensor(ones, space(H_list[j])[4])
            # @tensor G[-3 -4 -6 -8] := -1im*(Vₗ[9 -6; 7 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -8]
            # @tensor G[-3 -4 -5 -6] := -1im*(Vₗ[9 -5; -6 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -7]
            G[j] += -1im*(@tensor Vₗ[9 7; 4 1] * state.AR[j][1 2; 10] * H_list[j][7 5; 2 3] * conj(U_H_right[3]) * O₂[4 8; 5 6] * conj(U_space₂[6]) * conj(state.AR[j][9 8; 10]))
        end
    end
    
    return G
end

function correlator_wrong(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle_o, left::MPOTensor, right::MPOTensor, i::Int, N::Int)
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)
    
    U_space₁ = Tensor(ones, space(O₁)[1])
    U_space₂ = Tensor(ones, space(O₂)[4])
    
    unit_left = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], space(left)[1] ⊗ pspace, pspace ⊗ space(left)[1])
    unit_right = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], conj(space(right)[4]) ⊗ pspace, pspace ⊗ conj(space(right)[4]))
    
    G = similar(1:N, scalartype(state))
    
    for op_index = 1:N-1 # divide by N-1?
    
        println("op_index = $(op_index) of N = $(N)")
        H_list = fill(unit_left, N)
    
        H_list[op_index] = left
        H_list[op_index+1] = right
    
        U_H_left = Tensor(ones, space(H_list[i])[1])
        U_H_right = Tensor(ones, space(H_list[i])[4])
    
        for i₂ = op_index+2:N
            H_list[i₂] = unit_right
        end
    
        @tensor Vₗ[-1 -2; -3 -4] := state.AC[i][1 2; -4] * conj(U_space₁[3]) * O₁[3 4; 2 -3] * H_list[i][5 6; 4 -2] * conj(U_H_left[5]) * conj(state.AC[i][1 6; -1])
        @tensor Vᵣ[-1 -2; -3 -4] := state.AC[i][-1 2; 1] * H_list[i][-2 4; 2 3] * conj(U_H_right[3]) * O₂[-3 6; 4 5] * conj(U_space₂[5]) * conj(state.AC[i][-4 6; 1])
    
        for j = i-1:-1:1 # j < i ==> factor -i
            # global Vᵣ
            if j < i-1
                @tensor Vᵣ[-1 -2; -3 -4] := (-2im) * Vᵣ[1 4; -3 6] * (state.AL[j+1])[-1 2; 1] * middle[3; 2] * H_list[j+1][-2 5; 3 4] * conj((state.AL[j+1])[-4 5; 6])
            end
            if j <= op_index < i
                U_H_left = Tensor(ones, space(H_list[j])[1])
                final_value = 1im*(@tensor Vᵣ[4 8; 5 10] * state.AL[j][1 2; 4] * O₁[3 6; 2 5] * conj(U_space₁[3]) * H_list[j][7 9; 6 8] * conj(U_H_left[7]) * conj(state.AL[j][1 9; 10]))
                G[j] += final_value/(i-j)
            end
        end
        for j = i+1:N # j > i ==> factor i and conjugate
            # global Vₗ
            # global U_H_right
            if j > i+1
                @tensor Vₗ[-1 -2; -3 -4] := (2im) * Vₗ[1 4; -3 1] * (state.AR[j-1])[1 2; -4] * middle[3; 2] * H_list[j-1][4 5; 3 -2] * conj((state.AR[j-1])[1 5; -1])
            end
            if i <= op_index < j
                U_H_right = Tensor(ones, space(H_list[j])[4])
                # @tensor G[-3 -4 -6 -8] := -1im*(Vₗ[9 -6; 7 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -8]
                # @tensor G[-3 -4 -5 -6] := -1im*(Vₗ[9 -5; -6 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -7]
                final_value = -1im*(@tensor Vₗ[9 7; 4 1] * state.AR[j][1 2; 10] * H_list[j][7 5; 2 3] * conj(U_H_right[3]) * O₂[4 8; 5 6] * conj(U_space₂[6]) * conj(state.AR[j][9 8; 10]))
                G[j] += final_value/(j-i)
            end
        end
    end
    U_H_left = Tensor(ones, space(left)[1])
    U_H_right = Tensor(ones, space(right)[4])
    G[i] += @tensor (state.AC[i])[1 2; 10] * O₂[5 4; 2 3] * conj(U_space₂[3]) * left[6 7; 4 12] * conj(U_H_left[6]) * O₁[8 9; 7 5] * conj(U_space₁[8]) * conj((state.AC[i])[1 9; 15]) * (state.AR[i+1])[10 11; 16] * right[12 14; 11 13] * conj(U_H_right[13]) * conj((state.AR[i+1])[15 14; 16])
    return G
end

function transfer_matrix_loop(right_env, O₂, state, H, middle_o, i, sites, j, k)
    if sites == 0
        U_space₂ = Tensor(ones, space(O₂)[4])
        @tensor return_value[-1 -2; -3 -4] := state.AC[i][-1 1; 7] * H[i][j,k][-2 3; 1 2] * O₂[-3 5; 3 4] * conj(U_space₂[4]) * conj(state.AC[i][-4 5; 6]) * right_env[k][7 2; 6]
        return return_value
        # return transfer_right(right_env[k], H[i][j,k], state.AC[i], state.AC[i])
    else 
        elements = []
        for (a,b) in keys(H[i])
            if a == j
                @tensor above[-1 -2; -3] := (2im) * state.AL[i][-1 1; -3] * middle_o[-2; 1]
                new_element = transfer_right(transfer_matrix_loop(right_env, O₂, state, H, middle_o, i+1, sites-1, b, k), H[i][a,b], above, state.AL[i])
                push!(elements, new_element)
            end
        end
        total = elements[1]
        for i = 2:length(elements)
            total += elements[i]
        end
        return total
    end
end

function correlator_OLD2(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle_o, H::MPOHamiltonian, i::Int, N::Int)
    # spin = 1//2
    # pspace = U1Space(i => 1 for i in (-spin):spin)
    
    U_space₁ = Tensor(ones, space(O₁)[1])
    U_space₂ = Tensor(ones, space(O₂)[4])
    
    # unit_left = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], space(left)[1] ⊗ pspace, pspace ⊗ space(left)[1])
    # unit_right = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], conj(space(right)[4]) ⊗ pspace, pspace ⊗ conj(space(right)[4]))
    
    G = similar(1:N, scalartype(state))
    
    # for (j,k) in keys(H) # divide by N-1?
    
    #     println("op_index = $(op_index) of N = $(N)")
    #     H_list = fill(unit_left, N)
    
    #     H_list[op_index] = left
    #     H_list[op_index+1] = right
    
    #     U_H_left = Tensor(ones, space(H_list[i])[1])
    #     U_H_right = Tensor(ones, space(H_list[i])[4])
    
    #     for i₂ = op_index+2:N
    #         H_list[i₂] = unit_right
    #     end
    
    #     @tensor Vₗ[-1 -2; -3 -4] := state.AC[i][1 2; -4] * conj(U_space₁[3]) * O₁[3 4; 2 -3] * H_list[i][5 6; 4 -2] * conj(U_H_left[5]) * conj(state.AC[i][1 6; -1])
    #     @tensor Vᵣ[-1 -2; -3 -4] := state.AC[i][-1 2; 1] * H_list[i][-2 4; 2 3] * conj(U_H_right[3]) * O₂[-3 6; 4 5] * conj(U_space₂[5]) * conj(state.AC[i][-4 6; 1])
    
    #     for j = i-1:-1:1 # j < i ==> factor -i
    #         # global Vᵣ
    #         if j < i-1
    #             @tensor Vᵣ[-1 -2; -3 -4] := (-2im) * Vᵣ[1 4; -3 6] * (state.AL[j+1])[-1 2; 1] * middle_o[3; 2] * H_list[j+1][-2 5; 3 4] * conj((state.AL[j+1])[-4 5; 6])
    #         end
    #         if j <= op_index < i
    #             U_H_left = Tensor(ones, space(H_list[j])[1])
    #             final_value = 1im*(@tensor Vᵣ[4 8; 5 10] * state.AL[j][1 2; 4] * O₁[3 6; 2 5] * conj(U_space₁[3]) * H_list[j][7 9; 6 8] * conj(U_H_left[7]) * conj(state.AL[j][1 9; 10]))
    #             G[j] += final_value/(i-j)
    #         end
    #     end
    #     for j = i+1:N # j > i ==> factor i and conjugate
    #         # global Vₗ
    #         # global U_H_right
    #         if j > i+1
    #             @tensor Vₗ[-1 -2; -3 -4] := (2im) * Vₗ[1 4; -3 1] * (state.AR[j-1])[1 2; -4] * middle_o[3; 2] * H_list[j-1][4 5; 3 -2] * conj((state.AR[j-1])[1 5; -1])
    #         end
    #         if i <= op_index < j
    #             U_H_right = Tensor(ones, space(H_list[j])[4])
    #             # @tensor G[-3 -4 -6 -8] := -1im*(Vₗ[9 -6; 7 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -8]
    #             # @tensor G[-3 -4 -5 -6] := -1im*(Vₗ[9 -5; -6 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -7]
    #             final_value = -1im*(@tensor Vₗ[9 7; 4 1] * state.AR[j][1 2; 10] * H_list[j][7 5; 2 3] * conj(U_H_right[3]) * O₂[4 8; 5 6] * conj(U_space₂[6]) * conj(state.AR[j][9 8; 10]))
    #             G[j] += final_value/(j-i)
    #         end
    #     end
    # end
    # U_H_left = Tensor(ones, space(left)[1])
    # U_H_right = Tensor(ones, space(right)[4])

    U_space₁ = Tensor(ones, space(O₁)[1])
    U_space₂ = Tensor(ones, space(O₂)[4])

    envs = environments(state, H)
    for i₂ = i+1:N # j > i ==> factor i and conjugate
        sites = i₂-i-1
        for (j₁,k₁) in keys(H[i])
            for (j₂,k₂) in keys(H[i₂])
                elem = transfer_matrix_loop(rightenv(envs, i₂, state), O₂, state, H, middle_o, i+1, sites, k₁, k₂)
                element_final = @tensor leftenv(envs, i, state)[j₁][9 10; 11] * state.AL[i][11 1; 2] * O₁[3 5; 1 4] * conj(U_space₁[3]) * H[i][j₁,k₁][10 7; 5 6] * conj(state.AL[i][9 7; 8]) * elem[2 6; 4 8]
                G[i₂] += (-1im) * element_final / (2*(i₂ - i + 1))
            end
        end
    end
    # envs = environments(state, H)
    left_env = leftenv(envs, i, state)
    right_env = rightenv(envs, i, state)

    for (j,k) in keys(H[i])
        @tensor element_final = left_env[j][10 6; 1] * state.AC[i][1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[i][j,k][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj(state.AC[i][10 9; 13]) * right_env[k][11 12; 13]
        G[i] += element_final
    end

    for i₂ = i-1:-1:1 # j < i ==> factor -i
        sites = i-i₂-1
        for (j₁,k₁) in keys(H[i₂])
            for (j₂,k₂) in keys(H[i])
                elem = transfer_matrix_loop(rightenv(envs, i, state), O₂, state, H, middle_o, i₂+1, sites, k₁, k₂)
                element_final = @tensor leftenv(envs, i₂, state)[j₁][9 10; 11] * state.AL[i₂][11 1; 2] * O₁[3 5; 1 4] * conj(U_space₁[3]) * H[i₂][j₁,k₁][10 7; 5 6] * conj(state.AL[i₂][9 7; 8]) * elem[2 6; 4 8]
                G[i₂] += (1im) * element_final / (2*(i - i₂ + 1))
            end
        end
    end
    # # println(isa(rightenv(envs,i,state)[k], MPSTensor))
            # elem = transfer_matrix_loop(rightenv(envs, i+sites, state), O₂, state, H, middle_o, i, i₂-i, j, k)
            # element_final = @tensor leftenv(envs, i, state)[j][9 10; 11] * state.AC[i][11 1; 2] * O₁[3 5; 1 4] * conj(U_space₁[3]) * H[j,k]     elem[3 2; 1]
            # G[i₂] += element_final
            # # # G_new = @tensor leftenv(envs, i, state)[j][1 2; 3] * transfer_matrix_loop(rightenv(envs, i, state), state, H, i, j, k) [3 2; 1]
            # # transfer_right(rightenv(envs, i, state)[k], H[i][j,k], state.AC[i], state.AC[i])
            # # G[i] = @tensor (state.AC[i])[1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[j,k][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj((state.AC[i])[10 9; 13]) * leftenv(envs, length(state), state)[10 6; 1] * rightenv(envs, length(state), state)[11 12; 13]

            # value = @tensor (state.AC[i])[1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[i][j,k][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj((state.AC[i])[10 9; 13]) * leftenv(envs, i, state)[j][10 6; 1] * rightenv(envs, i, state)[k][11 12; 13]
        # for i₂ = i+1:N # j > i ==> factor i and conjugate
        #     for (j,k) in keys(H[i])
        #         # println(isa(rightenv(envs,i,state)[k], MPSTensor))
        #         elem = transfer_matrix_loop(rightenv(envs, i+sites, state), state, H, middle_o, i, i₂-i, j, k)
        #         element_final = @tensor leftenv(envs, i, state)[j][1 2; 3] * elem[3 2; 1]
        #         G[i₂] += element_final
        #         # # G_new = @tensor leftenv(envs, i, state)[j][1 2; 3] * transfer_matrix_loop(rightenv(envs, i, state), state, H, i, j, k) [3 2; 1]
        #         # transfer_right(rightenv(envs, i, state)[k], H[i][j,k], state.AC[i], state.AC[i])
        #         # G[i] = @tensor (state.AC[i])[1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[j,k][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj((state.AC[i])[10 9; 13]) * leftenv(envs, length(state), state)[10 6; 1] * rightenv(envs, length(state), state)[11 12; 13]
    
        #         value = @tensor (state.AC[i])[1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[i][j,k][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj((state.AC[i])[10 9; 13]) * leftenv(envs, i, state)[j][10 6; 1] * rightenv(envs, i, state)[k][11 12; 13]
        #     end
        # # value = @plansor (state.AC[i])[1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[i][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj((state.AC[i])[10 9; 13]) * leftenv(envs, i, state)[10 6; 1] * rightenv(envs, i, state)[11 12; 13]
        # end
    # G[i] += @tensor (state.AC[i])[1 2; 10] * O₂[5 4; 2 3] * conj(U_space₂[3]) * left[6 7; 4 12] * conj(U_H_left[6]) * O₁[8 9; 7 5] * conj(U_space₁[8]) * conj((state.AC[i])[1 9; 15]) * (state.AR[i+1])[10 11; 16] * right[12 14; 11 13] * conj(U_H_right[13]) * conj((state.AR[i+1])[15 14; 16])
    return G
end


function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle_o, H::MPOHamiltonian, N::Int)
    corr = zeros(ComplexF64, N, N)
    envs = environments(state, H)

    U_space₁ = Tensor(ones, space(O₁)[1])
    U_space₂ = Tensor(ones, space(O₂)[4])
    
    for i = 1:N
        println("i = $(i)")
        left_env = leftenv(envs, i, state)
        right_env = rightenv(envs, i, state)
    
        for (j,k) in keys(H[i])
            element_final = @tensor left_env[j][10 6; 1] * state.AC[i][1 2; 11] * O₂[4 5; 2 3] * conj(U_space₂[3]) * H[i][j,k][6 7; 5 12] * O₁[8 9; 7 4] * conj(U_space₁[8]) * conj(state.AC[i][10 9; 13]) * right_env[k][11 12; 13]
            corr[i,i] += element_final
        end

        @tensor above[-1 -2; -3] := (2im) * state.AR[i+1][-1 1; -3] * middle_o[-2; 1]
        transfer = TransferMatrix(above, H[i+1], state.AR[i+1])
        for sites = 2:N-i
            new_value = 0.0 
            for (j₁, k₁) in keys(H[i])
                for (j₂, k₂) in keys(transfer.middle)
                    if (k₁ == j₂)
                        for (j₃, k₃) in keys(H[i+sites])
                            if (k₂ == j₃)
                                new_value += @tensor left_env[j₁][6 5; 1] * state.AC[i][1 2; 8] * O₁[3 4; 2 18] * conj(U_space₁[3]) * H[i][j₁,k₁][5 7; 4 10] * conj(state.AC[i][6 7; 11]) * transfer.above[8 9; 13] * transfer.middle[j₂,k₂][10 12; 9 15] * conj(transfer.below[11 12; 19]) * state.AR[i+sites][13 14; 23] * H[i+sites][j₃,k₃][15 16; 14 22] * O₂[18 20; 16 17] * conj(U_space₂[17]) * conj(state.AR[i+sites][19 20; 21]) * right_env[k₃][23 22; 21]
                            end
                        end
                    end
                end
            end
            corr[i,i+sites] = (-1im) * new_value
            corr[i+sites,i] = (1im) * conj(new_value)

            @tensor above[-1 -2; -3] := (2im) * state.AR[i+sites][-1 1; -3] * middle_o[-2; 1]
            tra_new = TransferMatrix(above, H[i+sites], state.AR[i+sites])
            transfer = transfer * tra_new
        end

        if i != N
            new_value = 0.0
            for (j₁, k₁) in keys(H[i])
                for (j₂, k₂) in keys(H[i+1])
                    if (k₁ == j₂)
                        new_value += @tensor left_env[j₁][6 5; 1] * state.AC[i][1 2; 8] * O₁[3 4; 2 13] * conj(U_space₁[3]) * H[i][j₁,k₁][5 7; 4 10] * conj(state.AC[i][6 7; 14]) * state.AR[i+1][8 9; 18] * H[i+1][j₂,k₂][10 11; 9 17] * O₂[13 15; 11 12] * conj(U_space₂[12]) * conj(state.AR[i+1][14 15; 16]) * right_env[k₂][18 17; 16]
                    end
                end
            end
            corr[i,i+1] = new_value
            corr[i+1,i] = conj(new_value)
        end
    end
end

function correlator(state::AbstractMPS, O₁::MPOTensor, O₂::MPOTensor, middle, N::Int)
    # TO DO: implement function above for a range of i-values more efficiently
end