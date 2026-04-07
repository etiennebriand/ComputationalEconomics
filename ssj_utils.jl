# =============================================================================
#  ssj_utils.jl  —  Shared SSJ Utilities
# =============================================================================
#
#  This file is `include`d by both KS_ssj and hank_ssj notebooks.
#  It contains four functions that are identical (or functionally identical)
#  across the two models:
#
#    build_Λ(a_choice, np)      — sparse (asset, skill) transition matrix
#    inv_dist(Π)                — stationary distribution of a stochastic matrix
#    get_E(Λ, y_ss, T)          — forward expectation iteration (fake-news input)
#    build_J(Y, D, E, T)        — fake-news matrix + Sequence Space Jacobian
#
#  Both models use NumericalParameters structs with fields (na, ns, a_grid, Ps),
#  so build_Λ accepts np::NumericalParameters from either.  get_E and build_J
#  take the horizon T::Int directly (not np) so they work with both models'
#  NumericalParameters without requiring the same struct type.
#
#  Reference: Auclert, Bardóczy, Rognlie, Straub (2021, Econometrica)
#   "Using the Sequence-Space Jacobian to Solve and Estimate
#    Heterogeneous-Agent Models"
# =============================================================================


"""
    build_Λ(a_choice, np) → Λ  [na·ns × na·ns sparse]

Build the sparse transition matrix mapping the joint distribution over
(asset, skill) forward one period, given savings policy `a_choice` [na × ns].

State ordering: [a₁s₁, a₂s₁, …, aₙₐs₁, a₁s₂, …, aₙₐsₙₛ].

Uses linear interpolation weights between adjacent asset grid points.
Boundary cases (a_choice above/below grid) assign all mass to the
nearest grid point.
"""
function build_Λ(a_choice::Matrix, np)
    @unpack na, ns, a_grid, Ps = np

    weights_R = zeros(eltype(a_choice), na * ns, ns)
    weights_L = zeros(eltype(a_choice), na * ns, ns)
    IDX_col_R = zeros(Int64, na * ns, ns)   # column indices for the right weight

    @views for ss in 1:ns
        ss_shift = (ss - 1) * na
        for aa in 1:na
            al_idx = searchsortedlast(a_grid, a_choice[aa, ss])

            if al_idx == na
                # savings at/above grid top → all mass to the top node
                for sss in 1:ns
                    weights_R[ss_shift + aa, sss] += Ps[ss, sss]
                    IDX_col_R[ss_shift + aa, sss]  = al_idx + (sss - 1) * na
                end

            elseif al_idx == 0
                # savings below grid bottom → all mass to the lowest node
                # (left weight = 1; IDX_col_R = 2 so that IDX_col_R − 1 = 1)
                for sss in 1:ns
                    weights_L[ss_shift + aa, sss] += Ps[ss, sss]
                    IDX_col_R[ss_shift + aa, sss]  = (sss - 1) * na + 2
                end

            else
                # regular: linearly interpolate between al_idx and al_idx+1
                wr = (a_choice[aa, ss] - a_grid[al_idx]) /
                     (a_grid[al_idx + 1] - a_grid[al_idx])
                wl = 1.0 - wr
                for sss in 1:ns
                    weights_R[ss_shift + aa, sss] += Ps[ss, sss] * wr
                    weights_L[ss_shift + aa, sss] += Ps[ss, sss] * wl
                    IDX_col_R[ss_shift + aa, sss]  = (sss - 1) * na + al_idx + 1
                end
            end
        end
    end

    IDX_from = repeat(1:(na * ns); outer=2 * ns)
    weights  = vcat(vec(weights_R), vec(weights_L))
    IDX_to   = vcat(vec(IDX_col_R), vec(IDX_col_R) .- 1)

    return sparse(IDX_from, IDX_to, weights, na * ns, na * ns)
end


"""
    inv_dist(Π) → x  [na·ns vector]

Stationary distribution of stochastic matrix Π via the linear system
(I − Π')x = 0 with normalisation x₁ = 1.

Source: https://discourse.julialang.org/t/stationary-distribution-with-sparse-transition-matrix/40301/3
"""
function inv_dist(Π::AbstractArray)
    x = [1; (I - Π'[2:end, 2:end]) \ Vector(Π'[2:end, 1])]
    x = x ./ sum(x)
    @assert all(≥(-1e-10), x) "inv_dist: negative entries — check transition matrix."
    return max.(x, 0.0)
end


"""
    get_E(Λ, y_ss, T) → E  [length(y_ss) × (T+1)]

Forward-iterate the expectation operator over `T` periods:
  E[:,1]   = vec(y_ss)         (steady-state object, flattened)
  E[:,t+1] = Λ · E[:,t]

E[:,s] gives the expected value of `y_ss` exactly s−1 periods ahead,
starting from today's state.  Used to construct the fake-news matrix.

Arguments:
  Λ     : transition matrix [na·ns × na·ns]
  y_ss  : steady-state quantity of interest (array, flattened to a vector)
  T     : sequence-space horizon (integer)
"""
function get_E(Λ, y_ss, T::Int)
    E       = zeros(length(y_ss), T + 1)
    E[:, 1] = vec(y_ss)
    for tt in 2:(T + 1)
        E[:, tt] = Λ * E[:, tt-1]
    end
    return E
end


"""
    build_J(Y, D, E, T) → (J, F)

Assemble the T×T Sequence Space Jacobian J and fake-news matrix F.

Fake-news matrix (Auclert et al. 2021, eq. 9):
  F[1, s]   = Y[s]                          (direct effect at date of shock)
  F[t, s]   = E[:,t-1]' * D[:,s]  (t ≥ 2)  (lagged-expectation effect)

Jacobian from fake-news (eq. 12):
  J[:, 1]   = F[:, 1]
  J[1, t]   = F[1, t]             (t ≥ 2)
  J[2:, t]  = F[2:, t] + J[1:end-1, t-1]

Arguments:
  Y  : aggregate impulse vector       [T]
  D  : distribution impulse matrix    [na·ns × T]
  E  : expectation matrix             [na·ns × (T+1)]
  T  : sequence-space horizon (integer)
"""
function build_J(Y, D, E, T::Int)
    F         = zeros(T, T)
    F[1, :]   = Y
    F[2:end, :] .= E[:, 1:(T-1)]' * D

    J           = zeros(T, T)
    J[:, 1]     = F[:, 1]
    J[1, 2:end] = F[1, 2:end]
    for tt in 2:T
        @views J[2:end, tt] .= F[2:end, tt] .+ J[1:end-1, tt-1]
    end

    return J, F
end
