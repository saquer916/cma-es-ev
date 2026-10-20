// tests for cmaes.hpp, returns 0 if everything passes
#include <cmaes/cmaes.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

using cmaes::VectorXd;
using cmaes::MatrixXd;

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (cond) {                                                       \
      ++g_pass;                                                       \
    } else {                                                          \
      ++g_fail;                                                       \
      std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #cond);    \
      std::printf(__VA_ARGS__);                                       \
      std::printf("\n");                                              \
    }                                                                 \
  } while (0)

static bool close_rel(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }

template <class E>
static bool throws(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const E&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// ---- test functions ----

static double sphere(const VectorXd& x) { return x.squaredNorm(); }

static double rosenbrock(const VectorXd& x) {
  double s = 0;
  for (int i = 0; i + 1 < x.size(); ++i) s += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1 - x[i], 2);
  return s;
}

static double rastrigin(const VectorXd& x) {
  double s = 10.0 * x.size();
  for (int i = 0; i < x.size(); ++i) s += x[i] * x[i] - 10.0 * std::cos(2.0 * cmaes::detail::kPi * x[i]);
  return s;
}

// random rotation via QR of a gaussian matrix
static MatrixXd random_rotation(int n, std::uint64_t seed) {
  cmaes::detail::Rng rng(seed);
  MatrixXd A(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) A(i, j) = rng.normal();
  Eigen::HouseholderQR<MatrixXd> qr(A);
  return qr.householderQ() * MatrixXd::Identity(n, n);
}

// ---- tests ----

// expected values computed separately in python from the tutorial formulas
// (n = 3 and 40 have odd lambda)
static void test_default_parameters() {
  struct Row { int n, lambda, mu; double w1, mueff, cc, cs, c1, cmu, damps, chiN; };
  const Row rows[] = {
      {1, 4, 2, 0.8041628599327295, 1.4597898888525862, 0.6894039888609658, 0.46379186818956736, 0.2963055195692892,
       0.027690800889192065, 1.4637918681895674, 0.7976190476190477},
      {2, 6, 3, 0.6370425712412168, 2.0286114646100617, 0.6245545390268264, 0.44620498737831715, 0.1548153998964136,
       0.057859085071916304, 1.4462049873783172, 1.254272742818995},
      {3, 7, 3, 0.5856451065097651, 2.2548150822016044, 0.5588013228860189, 0.4149090010980616, 0.09640963257927214,
       0.05124308701358615, 1.4149090010980616, 1.5968775302586076},
      {10, 10, 5, 0.45627264690340597, 3.1672992814107017, 0.29499038303562225, 0.28442858794636744,
       0.015283824524751714, 0.02015428276120837, 1.2844285879463675, 3.0847265651690123},
      {40, 15, 7, 0.34479619859202054, 4.5409152090759575, 0.09300921663424917, 0.1320305687020827,
       0.0011694327252618397, 0.0031225007109974075, 1.1320305687020826, 6.28521508039776},
  };
  for (const Row& r : rows) {
    cmaes::CMAES es(VectorXd::Zero(r.n), 1.0);
    CHECK(es.lambda() == r.lambda, "n=%d lambda %d", r.n, es.lambda());
    CHECK(es.mu() == r.mu, "n=%d mu %d", r.n, es.mu());
    CHECK(close_rel(es.weights()[0], r.w1, 1e-13), "n=%d w1 %.17g", r.n, es.weights()[0]);
    CHECK(close_rel(es.weights().sum(), 1.0, 1e-14), "n=%d sum w %.17g", r.n, es.weights().sum());
    CHECK(close_rel(es.mueff(), r.mueff, 1e-13), "n=%d mueff %.17g", r.n, es.mueff());
    CHECK(close_rel(es.cc(), r.cc, 1e-13), "n=%d cc %.17g", r.n, es.cc());
    CHECK(close_rel(es.cs(), r.cs, 1e-13), "n=%d cs %.17g", r.n, es.cs());
    CHECK(close_rel(es.c1(), r.c1, 1e-13), "n=%d c1 %.17g", r.n, es.c1());
    CHECK(close_rel(es.cmu(), r.cmu, 1e-13), "n=%d cmu %.17g", r.n, es.cmu());
    CHECK(close_rel(es.damps(), r.damps, 1e-13), "n=%d damps %.17g", r.n, es.damps());
    CHECK(close_rel(es.chiN(), r.chiN, 1e-13), "n=%d chiN %.17g", r.n, es.chiN());
  }
}

static void test_rng() {
  cmaes::detail::Rng rng(42);
  const int N = 200000;
  double s = 0, s2 = 0, s4 = 0;
  for (int i = 0; i < N; ++i) {
    const double z = rng.normal();
    s += z, s2 += z * z, s4 += z * z * z * z;
  }
  const double mean = s / N, var = s2 / N - mean * mean, kurt = s4 / N;
  CHECK(std::fabs(mean) < 0.01, "mean %.4f", mean);
  CHECK(std::fabs(var - 1.0) < 0.01, "variance %.4f", var);
  CHECK(std::fabs(kurt - 3.0) < 0.06, "fourth moment %.4f", kurt);
  cmaes::detail::Rng a(7), b(7);
  bool same = true;
  for (int i = 0; i < 1000; ++i) same = same && a.normal() == b.normal();
  CHECK(same, "same seed gives the same samples");
}

static void test_sphere() {
  for (int n : {1, 2, 10, 30}) {
    const cmaes::Result r = cmaes::minimize(sphere, VectorXd::Constant(n, 3.0), 1.0);
    CHECK(r.f < 1e-10, "n=%d f %.3g stop %s", n, r.f, cmaes::to_string(r.stop));
    CHECK(r.stop == cmaes::Stop::TolFun || r.stop == cmaes::Stop::TolX, "n=%d stop %s", n, cmaes::to_string(r.stop));
    CHECK(r.mean.norm() < 1e-4, "n=%d |mean| %.3g", n, r.mean.norm());
  }
}

// rotated ellipsoid, cond 1e6. C should end up ~ inverse Hessian
static void test_rotated_ellipsoid() {
  const int n = 10;
  const MatrixXd Q = random_rotation(n, 3);
  VectorXd scale(n);
  for (int i = 0; i < n; ++i) scale[i] = std::pow(1e6, i / (n - 1.0));
  auto f = [&](const VectorXd& x) { const VectorXd y = Q * x; return (scale.array() * y.array().square()).sum(); };
  cmaes::Options opt;
  opt.seed = 5;
  cmaes::CMAES es(VectorXd::Constant(n, 1.0), 1.0, opt);
  while (!es.done()) {
    const auto& X = es.ask();
    std::vector<double> v;
    for (const auto& x : X) v.push_back(f(x));
    es.tell(v);
  }
  CHECK(es.best_f() < 1e-8, "f %.3g stop %s", es.best_f(), cmaes::to_string(es.stop()));
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(es.cov());
  const double cond = eig.eigenvalues().maxCoeff() / eig.eigenvalues().minCoeff();
  CHECK(cond > 1e5 && cond < 1e7, "condition of C %.3g (Hessian 1e6)", cond);
  // if C ~ H^-1 then H^(1/2) C H^(1/2) ~ c*I (starts at cond 1e6)
  const MatrixXd Hhalf = Q.transpose() * scale.cwiseSqrt().asDiagonal() * Q;
  Eigen::SelfAdjointEigenSolver<MatrixXd> fit(Hhalf * es.cov() * Hhalf);
  const double mismatch = fit.eigenvalues().maxCoeff() / fit.eigenvalues().minCoeff();
  CHECK(mismatch < 10.0, "condition of H^(1/2) C H^(1/2) %.3g (1e6 at the start)", mismatch);
}

static void test_rosenbrock() {
  for (int n : {2, 10}) {
    cmaes::Options opt;
    opt.restarts = 3;
    const cmaes::Result r = cmaes::minimize(rosenbrock, VectorXd::Zero(n), 0.5, opt);
    CHECK(r.f < 1e-10, "n=%d f %.3g after %d restarts", n, r.f, r.restarts);
    CHECK((r.x - VectorXd::Ones(n)).norm() < 1e-4, "n=%d |x - 1| %.3g", n, (r.x - VectorXd::Ones(n)).norm());
  }
}

static void test_rastrigin_ipop() {
  const int n = 5;
  cmaes::Options opt;
  opt.restarts = 9;
  opt.f_target = 1e-8;
  opt.lower = VectorXd::Constant(n, -5.12);
  opt.upper = VectorXd::Constant(n, 5.12);
  opt.seed = 11;
  const cmaes::Result r = cmaes::minimize(rastrigin, VectorXd::Constant(n, 3.0), 2.0, opt);
  CHECK(r.f <= 1e-8, "f %.3g after %d restarts, last lambda %d, stop %s", r.f, r.restarts, r.lambda, cmaes::to_string(r.stop));
  CHECK(r.stop == cmaes::Stop::TargetReached, "stop %s", cmaes::to_string(r.stop));
  CHECK(r.restarts > 0, "a 5-D Rastrigin should need at least one restart (%d)", r.restarts);
}

static void test_determinism() {
  cmaes::Options opt;
  opt.seed = 99;
  const cmaes::Result a = cmaes::minimize(rosenbrock, VectorXd::Zero(6), 0.5, opt);
  const cmaes::Result b = cmaes::minimize(rosenbrock, VectorXd::Zero(6), 0.5, opt);
  CHECK(a.x == b.x && a.f == b.f && a.evaluations == b.evaluations, "same seed, same run");
  opt.seed = 100;
  const cmaes::Result c = cmaes::minimize(rosenbrock, VectorXd::Zero(6), 0.5, opt);
  CHECK(!(a.x == c.x), "a different seed gives a different run");
}

// only ranks matter, so a monotone transform of f should give the exact same run
static void test_rank_invariance() {
  const int n = 8;
  cmaes::Options opt;
  opt.seed = 4;
  cmaes::CMAES a(VectorXd::Constant(n, 2.0), 0.7, opt), b(VectorXd::Constant(n, 2.0), 0.7, opt);
  for (int g = 0; g < 60; ++g) {
    const auto& Xa = a.ask();
    const auto& Xb = b.ask();
    std::vector<double> fa, fb;
    for (const auto& x : Xa) fa.push_back(rosenbrock(x));
    for (const auto& x : Xb) fb.push_back(std::exp(std::sqrt(rosenbrock(x))) - 5.0);
    a.tell(fa);
    b.tell(fb);
  }
  CHECK(a.mean() == b.mean() && a.sigma() == b.sigma() && a.cov() == b.cov(), "f and exp(sqrt(f)) - 5 give the same run");
}

// optimum at x = 2 is outside [-1, 1]^n so the answer should be the corner x = 1
static void test_bounds() {
  const int n = 6;
  for (auto mode : {cmaes::BoundHandling::Mirror, cmaes::BoundHandling::ClampPenalty}) {
    const char* name = mode == cmaes::BoundHandling::Mirror ? "mirror" : "clamp+penalty";
    cmaes::Options opt;
    opt.lower = VectorXd::Constant(n, -1.0);
    opt.upper = VectorXd::Constant(n, 1.0);
    opt.bound_handling = mode;
    bool inside = true;
    long long evals = 0;
    auto f = [&](const VectorXd& x) {
      ++evals;
      inside = inside && (x.array() >= -1.0).all() && (x.array() <= 1.0).all();
      return (x.array() - 2.0).square().sum();
    };
    const cmaes::Result r = cmaes::minimize(f, VectorXd::Zero(n), 0.5, opt);
    CHECK(inside, "%s: every evaluated point inside the box", name);
    CHECK((r.x - VectorXd::Ones(n)).norm() < 1e-6, "%s: |x - 1| %.3g", name, (r.x - VectorXd::Ones(n)).norm());
    CHECK((r.mean - VectorXd::Ones(n)).norm() < 1e-6, "%s: |mean - 1| %.3g", name, (r.mean - VectorXd::Ones(n)).norm());
    CHECK(evals == r.evaluations, "%s: evaluation count %lld vs %lld", name, evals, r.evaluations);
  }
  // optimum inside the box
  cmaes::Options opt;
  opt.lower = VectorXd::Constant(4, -10.0);
  opt.upper = VectorXd::Constant(4, 10.0);
  const cmaes::Result r = cmaes::minimize(sphere, VectorXd::Constant(4, 3.0), 2.0, opt);
  CHECK(r.f < 1e-10, "interior optimum with bounds: f %.3g", r.f);
}

// mirror shouldn't touch points already inside, and should reflect the rest correctly.
// compare against an unbounded twin with the same seed that gets told the same f
static double reflect(double x, double lo, double up) {
  while (x < lo || x > up) x = x < lo ? 2.0 * lo - x : 2.0 * up - x;
  return x;
}

static void test_mirror() {
  const int n = 4;
  for (double width : {1.0, 2e8}) {
    cmaes::Options ob;
    ob.lower = VectorXd::Constant(n, -width / 2);
    ob.upper = VectorXd::Constant(n, width / 2);
    const double sigma0 = width == 1.0 ? 3.0 : 1e-3;  // way outside the box / never near the edge
    const VectorXd x0 = VectorXd::Constant(n, 0.123456789012345);
    cmaes::CMAES bounded(x0, sigma0, ob), free(x0, sigma0);
    double worst = 0;
    bool interior_exact = true, inside = true;
    for (int g = 0; g < 30; ++g) {
      const auto& Xb = bounded.ask();
      const auto& Xf = free.ask();
      std::vector<double> f;
      for (size_t k = 0; k < Xb.size(); ++k) {
        for (int i = 0; i < n; ++i) {
          const double raw = Xf[k][i], got = Xb[k][i];
          inside = inside && got >= -width / 2 && got <= width / 2;
          if (raw >= -width / 2 && raw <= width / 2) interior_exact = interior_exact && got == raw;
          else worst = std::max(worst, std::fabs(got - reflect(raw, -width / 2, width / 2)));
        }
        f.push_back((Xb[k].array() - 0.3).square().sum());
      }
      bounded.tell(f);
      free.tell(f);
    }
    CHECK(inside, "width %g: every point inside the box", width);
    CHECK(interior_exact, "width %g: points inside the box are not moved", width);
    CHECK(worst < 1e-12, "width %g: reflection error %.3g", width, worst);
  }
}

static void test_stops() {
  cmaes::Options opt;
  opt.max_evals = 500;
  cmaes::Result r = cmaes::minimize(rosenbrock, VectorXd::Zero(10), 0.5, opt);
  CHECK(r.stop == cmaes::Stop::MaxEvals, "stop %s", cmaes::to_string(r.stop));
  CHECK(r.evaluations >= 500 && r.evaluations < 500 + 10, "evaluations %lld", r.evaluations);

  opt = cmaes::Options();
  opt.f_target = 1e-3;
  r = cmaes::minimize(sphere, VectorXd::Constant(5, 3.0), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::TargetReached && r.f <= 1e-3, "stop %s f %.3g", cmaes::to_string(r.stop), r.f);
  CHECK(r.f > 1e-6, "stopped at the target, not long after (f %.3g)", r.f);

  opt = cmaes::Options();
  opt.max_iter = 7;
  r = cmaes::minimize(sphere, VectorXd::Constant(5, 3.0), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::MaxIter && r.iterations == 7, "stop %s after %d", cmaes::to_string(r.stop), r.iterations);

  // linear f has no min, sigma keeps growing
  opt = cmaes::Options();
  opt.tol_up_sigma = 1e6;
  r = cmaes::minimize([](const VectorXd& x) { return x.sum(); }, VectorXd::Zero(3), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::TolUpSigma, "linear f: stop %s", cmaes::to_string(r.stop));

  // flat f
  r = cmaes::minimize([](const VectorXd&) { return 1.0; }, VectorXd::Zero(3), 1.0);
  CHECK(r.stop == cmaes::Stop::TolFun, "flat f: stop %s after %d", cmaes::to_string(r.stop), r.iterations);

  // all inf / all NaN should count as flat too
  r = cmaes::minimize([](const VectorXd&) { return std::numeric_limits<double>::infinity(); }, VectorXd::Zero(3), 1.0);
  CHECK(r.stop == cmaes::Stop::TolFun && r.iterations < 100, "+inf f: stop %s after %d", cmaes::to_string(r.stop), r.iterations);
  r = cmaes::minimize([](const VectorXd&) { return std::nan(""); }, VectorXd::Zero(3), 1.0);
  CHECK(r.stop == cmaes::Stop::TolFun && r.iterations < 100, "NaN f: stop %s after %d", cmaes::to_string(r.stop), r.iterations);
  CHECK(r.x == VectorXd::Zero(3) && std::isinf(r.f), "NaN f: best is x0 with f = inf");

  // tol_fun off -> tolx
  opt = cmaes::Options();
  opt.tol_fun = 0;
  r = cmaes::minimize(sphere, VectorXd::Constant(5, 1.0), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::TolX, "tol_fun 0: stop %s", cmaes::to_string(r.stop));

  // cond 1e6 problem vs max_condition 1e4
  opt = cmaes::Options();
  opt.max_condition = 1e4;
  r = cmaes::minimize([](const VectorXd& x) { return x[0] * x[0] + 1e6 * x[1] * x[1] + x[2] * x[2]; }, VectorXd::Constant(3, 1.0), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::ConditionCov, "ill-conditioned: stop %s", cmaes::to_string(r.stop));

  // tol_fun and tol_x off: sigma shrinks until it stops moving the mean
  opt = cmaes::Options();
  opt.tol_fun = 0;
  opt.tol_x = 1e-300;
  r = cmaes::minimize([](const VectorXd& x) { return (x.array() - 1.0).square().sum(); }, VectorXd::Zero(3), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::NoEffectAxis, "stop %s (expected no effect axis)", cmaes::to_string(r.stop));
  r = cmaes::minimize([](const VectorXd& x) { return (x[0] - 1e6) * (x[0] - 1e6) + 1e12 * x[1] * x[1] + x[2] * x[2]; },
                      VectorXd::Zero(3), 1.0, opt);
  CHECK(r.stop == cmaes::Stop::NoEffectCoord, "stop %s (expected no effect coordinate)", cmaes::to_string(r.stop));

  // max_evals budget is shared across restarts
  opt = cmaes::Options();
  opt.restarts = 50;
  opt.max_evals = 3000;
  r = cmaes::minimize(rastrigin, VectorXd::Constant(10, 3.0), 2.0, opt);
  CHECK(r.evaluations >= 3000 && r.evaluations < 3000 + r.lambda, "evaluations %lld over %d restarts", r.evaluations, r.restarts);

  // sigma_max
  opt = cmaes::Options();
  opt.sigma_max = 0.3;
  opt.max_iter = 50;
  cmaes::CMAES es(VectorXd::Zero(3), 0.3, opt);
  double smax = 0;
  while (!es.done()) {
    const auto& X = es.ask();
    std::vector<double> v;
    for (const auto& x : X) v.push_back(x.sum());
    es.tell(v);
    smax = std::max(smax, es.sigma());
  }
  CHECK(smax <= 0.3, "sigma %.3g with sigma_max 0.3", smax);
}

// half the space returns NaN, should still converge on the other half
static void test_nan() {
  auto f = [](const VectorXd& x) { return x[0] < 0 ? std::nan("") : (x.array() - 1.0).square().sum(); };
  const cmaes::Result r = cmaes::minimize(f, VectorXd::Constant(4, 0.5), 0.5);
  CHECK(r.f < 1e-10, "f %.3g", r.f);
  CHECK(!std::isnan(r.f), "best f is not NaN");
}

// noisy f (like a monte carlo sim), mean should end up near the optimum
static void test_noisy() {
  const int n = 5;
  cmaes::detail::Rng noise(77);
  cmaes::Options opt;
  opt.max_iter = 400;
  opt.lambda = 20;
  const cmaes::Result r =
      cmaes::minimize([&](const VectorXd& x) { return x.squaredNorm() + 0.01 * noise.normal(); }, VectorXd::Constant(n, 2.0), 1.0, opt);
  CHECK(r.mean.norm() < 0.2, "|mean| %.3g with noise sd 0.01", r.mean.norm());
}

// common random numbers: whole generation gets the same noise
static void test_batch() {
  const int n = 4;
  cmaes::Options opt;
  opt.max_iter = 300;
  int gen = 0;
  const cmaes::Result r = cmaes::minimize_batch(
      [&](const std::vector<VectorXd>& X) {
        cmaes::detail::Rng draw(1000 + gen++);
        const double shift = 0.5 * draw.normal();  // same for everyone so ranking ignores it
        std::vector<double> v;
        for (const auto& x : X) v.push_back(x.squaredNorm() + shift);
        return v;
      },
      VectorXd::Constant(n, 2.0), 1.0, opt);
  CHECK(r.mean.norm() < 1e-6, "|mean| %.3g", r.mean.norm());
}

static void test_arguments() {
  const VectorXd x0 = VectorXd::Zero(3);
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 0.0); }), "sigma0 = 0");
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, std::nan("")); }), "sigma0 = NaN");
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(VectorXd(), 1.0); }), "empty x0");
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(VectorXd::Constant(3, std::nan("")), 1.0); }), "NaN x0");
  cmaes::Options opt;
  opt.lambda = 1;
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "lambda = 1");
  opt = cmaes::Options();
  opt.lambda = 6;
  opt.mu = 7;
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "mu > lambda");
  opt = cmaes::Options();
  opt.lower = VectorXd::Constant(2, -1.0);
  opt.upper = VectorXd::Constant(2, 1.0);
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "bounds of the wrong size");
  opt.lower = VectorXd::Constant(3, 1.0);
  opt.upper = VectorXd::Constant(3, 1.0);
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "lower == upper");
  opt.lower = VectorXd::Constant(3, 0.5);
  opt.upper = VectorXd::Constant(3, 1.0);
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "x0 outside the bounds");
  cmaes::CMAES es(x0, 1.0);
  CHECK(throws<std::logic_error>([&] { es.tell(std::vector<double>(es.lambda(), 0.0)); }), "tell before ask");
  es.ask();
  CHECK(throws<std::invalid_argument>([&] { es.tell(std::vector<double>(es.lambda() + 1, 0.0)); }), "tell with the wrong count");
  opt = cmaes::Options();
  opt.lower = VectorXd::Constant(3, -1.0);
  opt.upper = VectorXd::Constant(3, 1.0);
  opt.bound_handling = cmaes::BoundHandling::ClampPenalty;
  opt.penalty_weight = std::numeric_limits<double>::infinity();
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "penalty_weight = inf");
  opt = cmaes::Options();
  opt.sigma_max = 0;
  CHECK(throws<std::invalid_argument>([&] { cmaes::CMAES es(x0, 1.0, opt); }), "sigma_max = 0");
  opt = cmaes::Options();
  opt.restarts = -1;
  CHECK(throws<std::invalid_argument>([&] { cmaes::minimize(sphere, x0, 1.0, opt); }), "restarts = -1");
  opt = cmaes::Options();
  opt.mu = 50;
  CHECK(throws<std::invalid_argument>([&] { cmaes::minimize(sphere, x0, 1.0, opt); }), "minimize with mu > lambda");
  CHECK(throws<std::invalid_argument>([&] { cmaes::minimize(sphere, VectorXd(), 1.0); }), "minimize with an empty x0");
  // ask() after a stop
  opt = cmaes::Options();
  opt.max_iter = 2;
  cmaes::CMAES es3(x0, 1.0, opt);
  while (!es3.done()) es3.tell(std::vector<double>(es3.ask().size(), 1.0));
  CHECK(throws<std::logic_error>([&] { es3.ask(); }), "ask after the run stopped");
  // all +inf: best_x should still be a point we actually evaluated
  cmaes::CMAES es4(x0, 1.0);
  const std::vector<VectorXd> X4 = es4.ask();
  es4.tell(std::vector<double>(X4.size(), std::numeric_limits<double>::infinity()));
  CHECK(es4.best_x() == X4[0] && std::isinf(es4.best_f()), "all +inf: best_x is the first evaluated point");
  // C should stay exactly symmetric
  cmaes::CMAES es5(VectorXd::Constant(12, 1.0), 1.0);
  bool symmetric = true;
  for (int g = 0; g < 50; ++g) {
    const auto& X = es5.ask();
    std::vector<double> v;
    for (const auto& x : X) v.push_back(rosenbrock(x));
    es5.tell(v);
    symmetric = symmetric && es5.cov() == es5.cov().transpose();
  }
  CHECK(symmetric, "C is exactly symmetric after every generation");
  // custom mu
  opt = cmaes::Options();
  opt.lambda = 12;
  opt.mu = 3;
  cmaes::CMAES es2(x0, 1.0, opt);
  CHECK(es2.mu() == 3 && es2.weights().size() == 3, "custom mu");
}

int main() {
  test_default_parameters();
  test_rng();
  test_sphere();
  test_rotated_ellipsoid();
  test_rosenbrock();
  test_rastrigin_ipop();
  test_determinism();
  test_rank_invariance();
  test_bounds();
  test_mirror();
  test_stops();
  test_nan();
  test_noisy();
  test_batch();
  test_arguments();
  std::printf("%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
