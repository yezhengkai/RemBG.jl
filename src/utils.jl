# https://medium.com/@bowlescompling/m2-1-softmax-in-julia-1498901f741c
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html#scipy.special.log_softmax
"""
    log_softmax(x)
    log_softmax(x, dims)

Compute the logarithm of the softmax function.

In principle::

    log_softmax(x) = log(softmax(x))

but using a more accurate implementation.

# Arguments
- `x`: Input array
- `dims`: Axis to compute values along

# Returns
- `Array or Number`: An array with the same shape as `x`. Exponential of the result will
sum to 1 along the specified axis. If `x` is a scalar, a scalar is
returned.

# Examples
```julia
julia> x = [1000.0 1.0];
julia> y = log_softmax(x)
2-element Vector{Float64}:
    0.0
 -999.0

julia> x = [1000.0 1.0; 1000.0 1.0];
julia> y = log_softmax(x, 2)
2×2 Matrix{Float64}:
 0.0  -999.0
 0.0  -999.0
```
"""
function log_softmax(x)
    x_max = maximum(x)

    if ndims(x_max) > 0
        x_max[.!isfinite.(x_max)] .= 0
    elseif !isfinite(x_max)
        x_max = 0
    end

    tmp = x .- x_max
    exp_tmp = exp.(tmp)

    s = sum(exp_tmp)
    out = log(s)

    out = tmp .- out
    return out
end
function log_softmax(x, dims)
    x_max = maximum(x; dims=dims)

    if ndims(x_max) > 0
        x_max[.!isfinite.(x_max)] .= 0
    elseif !isfinite(x_max)
        x_max = 0
    end

    tmp = x .- x_max
    exp_tmp = exp.(tmp)

    s = sum(exp_tmp; dims=dims)
    out = log.(s)

    out = tmp .- out
    return out
end
