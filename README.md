# cmaes

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) in one C++17 header, built on Eigen.

CMA-ES minimizes a function f(x) of n real numbers without gradients. Each generation it samples a
population of points from a multivariate normal distribution, ranks them by f, and moves and
reshapes the distribution toward the better ones. It only uses the ranking of f, so it copes with
objectives that are noisy, non-smooth, badly scaled or built from a simulation, such as tuning the
constants of a robot or a vehicle in a Monte Carlo simulator.

The equations and default parameters follow N. Hansen, [The CMA Evolution Strategy: A
Tutorial](https://arxiv.org/abs/1604.00772), with positive recombination weights only.

## Use it

Copy `include/cmaes/cmaes.hpp` into your project (or add `include/` to the include path) and make
sure Eigen 3.3 or newer is available. With CMake you can `add_subdirectory` this repo and link
`cmaes::cmaes`.

```cpp
#include <cmaes/cmaes.hpp>

double f(const Eigen::VectorXd& x) { return x.squaredNorm(); }

int main() {
  cmaes::Options opt;            // defaults are fine for most problems
  cmaes::Result r = cmaes::minimize(f, Eigen::VectorXd::Constant(10, 3.0), /*sigma0=*/1.0, opt);
  // r.x, r.f: best point found; r.mean: final mean; r.stop: why it stopped
}
```

`x0` is the starting point and `sigma0` the initial step size. A good `sigma0` is about a third of
the range where you expect the optimum, in every coordinate. If the coordinates have very different
ranges, search over the unit box and scale each one to its real range inside f (see
`examples/ask_tell.cpp`).

### Ask and tell

To evaluate in parallel, or run your own loop, drive the optimizer yourself:

```cpp
cmaes::CMAES es(x0, sigma0, opt);
while (!es.done()) {
  const std::vector<Eigen::VectorXd>& X = es.ask();   // es.lambda() points to evaluate
  std::vector<double> fx(X.size());
  for (size_t k = 0; k < X.size(); ++k) fx[k] = f(X[k]);   // in any order, on any thread
  es.tell(fx);                                         // same order as X
}
Eigen::VectorXd answer = es.best_x();   // or es.mean() when f is noisy
```

f may return NaN for a point it cannot score; NaN is ranked like +inf, after every finite value.
Once `done()` is true, `ask()` throws: start a new `CMAES` to search on.

`minimize_batch(fb, x0, sigma0, opt)` does the same with a function that scores a whole
generation at once: `std::vector<double> fb(const std::vector<Eigen::VectorXd>& X)`.

### Noisy objectives

When f comes from a random simulation:

- Use `es.mean()` / `r.mean` as the answer. `best_x()` is the luckiest sample, not the best point.
- Run every candidate of a generation on the same random trials (common random numbers), so the
  ranking compares candidates and not their luck, and draw new trials each generation so the
  search cannot overfit one set. `minimize_batch` and ask/tell make this easy.
- Noise keeps sigma from shrinking to zero, so the tolerance stops rarely trigger: set `max_iter`
  or `max_evals`. A larger `lambda` averages out more noise per generation.
- Score the result again on trials the search never saw before you trust its value.

### Bounds

Set `opt.lower` and `opt.upper` (both finite, same size as `x0`, with `x0` inside). Every point
handed to f is then inside the box.

- `BoundHandling::Mirror` (default): the search runs unbounded and each sample that lands outside
  the box is reflected back into it before it is evaluated. Points inside are not changed. Needs
  no tuning.
- `BoundHandling::ClampPenalty`: each sample is clamped into the box, and for ranking only
  `penalty_weight * sum(((x - clamped) / (upper - lower))^2)` is added to its f. The penalty must be
  large next to the differences in f near the bound.

## Options

| option | default | meaning |
|---|---|---|
| `lambda` | 4 + floor(3 ln n) | points per generation |
| `mu` | lambda / 2 | best points recombined into the new mean |
| `seed` | 1 | the same seed and the same f give the same run on the same machine, compiler and compiler flags (flags such as `-march=native` change the floating-point results) |
| `max_iter` | 100 + 150 (n+3)² / sqrt(lambda) | generations per run |
| `max_evals` | no limit | evaluations, shared by all restarts; checked after each generation |
| `f_target` | -inf | stop once f ≤ f_target |
| `tol_fun` | 1e-12 | stop when the best f of the last 10 + ceil(30n/lambda) generations, and every f of the latest one, span less than this |
| `tol_x` | 1e-12 · sigma0 | stop when every coordinate's step and spread are below this |
| `tol_up_sigma` | 1e20 | stop when sigma/sigma0 exceeds this times the largest axis of C: sigma0 too small, or f has no minimum |
| `max_condition` | 1e14 | stop when the condition number of C exceeds this |
| `sigma_max` | inf | cap on the step size |
| `lower`, `upper` | none | box bounds |
| `bound_handling` | `Mirror` | `Mirror` or `ClampPenalty` |
| `penalty_weight` | 1 | `ClampPenalty` only |
| `restarts` | 0 | `minimize` only: restarts with a larger population (IPOP-CMA-ES) after any stop other than `TargetReached` or `MaxEvals` |
| `ipop_factor` | 2 | `minimize` only: population growth per restart |

Restarts begin again from `x0` with a new seed. They go on until `f_target` is reached, `max_evals`
is used up or every restart is done, so set `f_target` when you know how good the optimum is.

A run can also stop on `NoEffectAxis` or `NoEffectCoord`, when the step size has become too small
to change the mean in floating point. `cmaes::to_string(stop)` names any stop reason.

## Build the tests and examples

```
make test                 # builds everything and runs the tests; needs a C++17 compiler and Eigen
                          # (found with pkg-config, else /usr/include/eigen3)
make test EIGEN=/opt/homebrew/include/eigen3      # Eigen somewhere else, e.g. Homebrew on macOS
build/rosenbrock          # 10-D Rosenbrock with restarts
build/ask_tell            # tuning a noisy simulation with parallel evaluation
```

or with CMake (any generator, including Visual Studio and Xcode):

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build && ctest -C Release
```

The tests check the default parameters against the tutorial's formulas, convergence on the
sphere, a rotated ellipsoid (including that C learns the inverse Hessian), Rosenbrock and Rastrigin
with restarts, both kinds of bounds (including that mirroring reflects exactly and leaves inside
points alone), every stop reason, NaN and infinite values, noisy and batch objectives, argument
checks, determinism, and that only the ranking of f matters.

## References

- N. Hansen. The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772, 2016.
- N. Hansen and A. Ostermeier. Completely Derandomized Self-Adaptation in Evolution Strategies.
  Evolutionary Computation 9(2), 2001.
- A. Auger and N. Hansen. A Restart CMA Evolution Strategy With Increasing Population Size.
  IEEE Congress on Evolutionary Computation, 2005.
