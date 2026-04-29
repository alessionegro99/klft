# PLAN.md — First Wilson-Loop Kernel Fusion Improvement

## Goal

Improve the existing temporal Wilson-loop measurement by replacing the pattern

```cpp
for (int r = 1; r <= Rmax; ++r)
  for (int t = 1; t <= Tmax; ++t)
    Kokkos::parallel_reduce(...);
```

with a simpler first optimization:

```text
one Kokkos kernel per temporal length t,
fused over all r, all spatial directions, and all lattice sites
```

This keeps the implementation simple while reducing the number of measurement kernels from `Rmax * Tmax` to `Tmax`.

## Target Observable

Measure temporal Wilson loops for the static potential:

\[
W(r,t)=\frac{1}{3V}\sum_{i=1}^{3}\sum_x
\frac{1}{N_c}\mathrm{ReTr}
\left[
S_i(x,r)
T(x+r\hat i,t)
S_i(x+t\hat 0,r)^\dagger
T(x,t)^\dagger
\right].
\]

Here:

- `0` is the temporal direction, adapt this to the code convetnions.
- `i = 1,2,3` are spatial directions;
- `S_i(x,r)` is the spatial transporter of length `r`;
- `T(x,t)` is the temporal transporter of length `t`;
- all shifts use periodic boundary conditions.

## Implementation Strategy

### 1. Keep or add spatial transporter precomputation

Precompute spatial lines once per gauge configuration:

```cpp
S(i, 1, x) = U(i, x);

S(i, r + 1, x) =
    S(i, r, x) * U(i, shift(x, i, r));
```

for:

```text
i = 1,2,3
r = 1,...,Rmax
x = all lattice sites
```

Store these in a device `Kokkos::View`.

Suggested logical layout:

```cpp
S(idir, r, site)
```

where `idir = 0,1,2` corresponds to spatial directions `mu = 1,2,3`.

### 2. Use rolling temporal transporters

Avoid storing all `T(t,x)` for now.

Use one device array:

```cpp
Tcurr(site)
```

Initialize before the `t` loop:

```cpp
Tcurr(site) = identity;
```

Then for each temporal length:

```cpp
for (int t = 1; t <= Tmax; ++t) {
    update_Tcurr_to_length_t(t);
    measure_all_r_for_fixed_t(t);
}
```

The update should implement:

```cpp
Tcurr(x) = Tcurr(x) * U(0, shift(x, 0, t - 1));
```

This must be done in a `Kokkos::parallel_for` over all sites.

### 3. Replace many reductions with one fused kernel per t

For fixed `t`, measure all `r`, all spatial directions, and all sites in one `Kokkos::parallel_for`.

Before measuring, make sure the output accumulator is zeroed:

```cpp
Kokkos::deep_copy(W_accum, 0.0);
```

Use an accumulator view:

```cpp
Kokkos::View<double**> W_accum("W_accum", Rmax + 1, Tmax + 1);
```

Then for each `t`:

```cpp
Kokkos::parallel_for(
    "measure_temporal_wilson_loops_fixed_t",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {1, 0, 0},
        {Rmax + 1, 3, Volume}
    ),
    KOKKOS_LAMBDA(const int r, const int idir, const int site) {
        const int mu = idir + 1;

        const int x   = site;
        const int x_r = shift(site, mu, r);
        const int x_t = shift(site, 0,  t);

        Matrix loop =
            S(idir, r, x)
          * Tcurr(x_r)
          * dagger(S(idir, r, x_t))
          * dagger(Tcurr(x));

        const double w = real_trace(loop) / Nc;

        Kokkos::atomic_add(&W_accum(r, t), w);
    }
);
```

This is the main requested improvement.

### 4. Normalize after all measurements

After the `t` loop, normalize on device:

```cpp
Kokkos::parallel_for(
    "normalize_temporal_wilson_loops",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {1, 1},
        {Rmax + 1, Tmax + 1}
    ),
    KOKKOS_LAMBDA(const int r, const int t) {
        W_accum(r, t) /= (3.0 * Volume);
    }
);
```

Then copy `W_accum` to host once and write the output.

## Expected Benefit

Current structure:

```text
Rmax * Tmax measurement reductions
```

First optimized structure:

```text
Tmax measurement kernels
```

For example, if `Rmax = 9` and `Tmax = 9`, this changes the measurement from 81 kernels to 9 kernels.

This should improve GPU utilization if the previous implementation was dominated by many small reductions and synchronizations.

## Output

Keep the existing output format if possible.

Write only after copying the final normalized `W_accum` to host.

Avoid copying or writing after each individual `(r,t)`.

## Correctness Checks

### 1. Plaquette check

For:

```text
r = 1
t = 1
```

the result should match the average temporal plaquette in the `(i,0)` planes, averaged over `i = 1,2,3`.

### 2. Naive comparison

Keep a temporary naive implementation for a small lattice.

Compare:

```text
precomputed fused result
vs
link-by-link Wilson loop result
```

for several `(r,t)` values.

Agreement should be at floating-point roundoff level.

### 3. Boundary check

Test loops starting near the spatial and temporal boundaries.

The result must be correct with periodic boundary conditions.

### 4. Gauge invariance check

Apply a random gauge transformation to a saved configuration and verify that `W(r,t)` is unchanged within numerical precision.

## Notes for Codex

Do not implement temporal gauge fixing.

Do not implement the more complex two-stage block reduction yet.

Do not precompute all temporal transporters `T(t,x)` yet.

The goal of this change is only the easiest first improvement:

```text
rolling Tcurr
+
one fused Kokkos::parallel_for per t
+
Kokkos::atomic_add into W_accum(r,t)
```

Keep the code readable and consistent with the existing Kokkos style in the project.
