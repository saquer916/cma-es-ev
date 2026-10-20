// CMA-ES in one header (C++17 + Eigen)
// follows Hansen's tutorial: https://arxiv.org/abs/1604.00772 (positive weights only, no active CMA)
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cmaes {

using Eigen::MatrixXd;
using Eigen::VectorXd;

enum class Stop {
  None,
  TargetReached,
  MaxEvals,
  MaxIter,
  TolFun,
  TolX,
  TolUpSigma,  // sigma blew up: sigma0 too small or f has no minimum
  ConditionCov,
  NoEffectAxis,
  NoEffectCoord,
};

inline const char* to_string(Stop s) {
  switch (s) {
    case Stop::None: return "none";
    case Stop::TargetReached: return "target reached";
    case Stop::MaxEvals: return "max evaluations";
    case Stop::MaxIter: return "max iterations";
    case Stop::TolFun: return "tolfun";
    case Stop::TolX: return "tolx";
    case Stop::TolUpSigma: return "tolupsigma";
    case Stop::ConditionCov: return "condition of C";
    case Stop::NoEffectAxis: return "no effect axis";
    case Stop::NoEffectCoord: return "no effect coordinate";
  }
  return "unknown";
}

// either way f only ever sees points inside the box
enum class BoundHandling {
  Mirror,        // reflect samples back in
  ClampPenalty,  // clamp, then add penalty_weight * (scaled distance outside)^2 when ranking
};

struct Options {
  int lambda = 0;                  // 0 = 4 + floor(3 ln n)
  int mu = 0;                      // 0 = lambda / 2
  std::uint64_t seed = 1;
  int max_iter = 0;                // 0 = 100 + 150 (n+3)^2 / sqrt(lambda)
  long long max_evals = 0;         // 0 = no limit
  double f_target = -std::numeric_limits<double>::infinity();
  double tol_fun = 1e-12;
  double tol_x = 0;                // 0 = 1e-12 * sigma0
  double tol_up_sigma = 1e20;
  double max_condition = 1e14;
  double sigma_max = std::numeric_limits<double>::infinity();
  VectorXd lower, upper;           // leave empty for no bounds
  BoundHandling bound_handling = BoundHandling::Mirror;
  double penalty_weight = 1.0;     // ClampPenalty only
  int restarts = 0;                // IPOP restarts, minimize() only
  double ipop_factor = 2.0;
};

namespace detail {

constexpr double kPi = 3.14159265358979323846;

// own Box-Muller so runs don't depend on the std lib's normal_distribution
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : g_(seed) {}
  double uniform() { return (static_cast<double>(g_() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }  // (0, 1)
  double normal() {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double r = std::sqrt(-2.0 * std::log(uniform())), t = 2.0 * kPi * uniform();
    spare_ = r * std::sin(t);
    has_spare_ = true;
    return r * std::cos(t);
  }

 private:
  std::mt19937_64 g_;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

inline int default_lambda(int n) { return 4 + static_cast<int>(std::floor(3.0 * std::log(static_cast<double>(n)))); }

}  // namespace detail

// usage: while (!es.done()) { auto& X = es.ask(); ...score each x...; es.tell(f); }
class CMAES {
 public:
  CMAES(const VectorXd& x0, double sigma0, const Options& opt = Options())
      : opt_(opt), n_(static_cast<int>(x0.size())), sigma0_(sigma0), sigma_(sigma0), rng_(opt.seed) {
    if (n_ < 1) throw std::invalid_argument("cmaes: x0 must have at least one coordinate");
    if (!x0.allFinite()) throw std::invalid_argument("cmaes: x0 must be finite");
    if (!(sigma0 > 0.0) || !std::isfinite(sigma0)) throw std::invalid_argument("cmaes: sigma0 must be positive and finite");
    bounded_ = opt.lower.size() > 0 || opt.upper.size() > 0;
    if (bounded_) {
      if (opt.lower.size() != n_ || opt.upper.size() != n_)
        throw std::invalid_argument("cmaes: lower and upper must both have the size of x0");
      if (!opt.lower.allFinite() || !opt.upper.allFinite() || !(opt.lower.array() < opt.upper.array()).all())
        throw std::invalid_argument("cmaes: bounds must be finite with lower < upper");
      if ((x0.array() < opt.lower.array()).any() || (x0.array() > opt.upper.array()).any())
        throw std::invalid_argument("cmaes: x0 must lie inside the bounds");
      if (opt.bound_handling == BoundHandling::ClampPenalty && !(opt.penalty_weight >= 0.0 && std::isfinite(opt.penalty_weight)))
        throw std::invalid_argument("cmaes: penalty_weight must be finite and >= 0");
    }
    if (!(opt.sigma_max > 0.0)) throw std::invalid_argument("cmaes: sigma_max must be positive");

    lambda_ = opt.lambda > 0 ? opt.lambda : detail::default_lambda(n_);
    mu_ = opt.mu > 0 ? opt.mu : lambda_ / 2;
    if (lambda_ < 2) throw std::invalid_argument("cmaes: lambda must be at least 2");
    if (mu_ < 1 || mu_ > lambda_) throw std::invalid_argument("cmaes: mu must be between 1 and lambda");

    // weights (tutorial eq. 49), ln(mu + 1/2) base if mu was set by hand
    w_.resize(mu_);
    const double wbase = opt.mu > 0 ? mu_ + 0.5 : (lambda_ + 1) / 2.0;
    for (int i = 0; i < mu_; ++i) w_[i] = std::log(wbase) - std::log(i + 1.0);
    w_ /= w_.sum();
    mueff_ = 1.0 / w_.squaredNorm();

    const double n = n_;
    cc_ = (4.0 + mueff_ / n) / (n + 4.0 + 2.0 * mueff_ / n);
    cs_ = (mueff_ + 2.0) / (n + mueff_ + 5.0);
    c1_ = 2.0 / ((n + 1.3) * (n + 1.3) + mueff_);
    cmu_ = std::min(1.0 - c1_, 2.0 * (mueff_ - 2.0 + 1.0 / mueff_) / ((n + 2.0) * (n + 2.0) + mueff_));
    damps_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff_ - 1.0) / (n + 1.0)) - 1.0) + cs_;
    chiN_ = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));

    max_iter_ = opt.max_iter > 0 ? opt.max_iter
                                 : static_cast<int>(std::min<double>(std::numeric_limits<int>::max(),
                                       100.0 + std::floor(150.0 * (n + 3.0) * (n + 3.0) / std::sqrt(static_cast<double>(lambda_)))));
    tol_x_ = opt.tol_x > 0 ? opt.tol_x : 1e-12 * sigma0;
    hist_len_ = 10 + static_cast<int>(std::ceil(30.0 * n / lambda_));
    L_ = static_cast<std::size_t>(lambda_);

    m_ = x0;
    pc_ = VectorXd::Zero(n_);
    ps_ = VectorXd::Zero(n_);
    C_ = MatrixXd::Identity(n_, n_);
    B_ = MatrixXd::Identity(n_, n_);
    D_ = VectorXd::Ones(n_);
    best_x_ = x0;
  }

  const std::vector<VectorXd>& ask() {
    if (done()) throw std::logic_error("cmaes: ask() after the run stopped");
    Y_.assign(L_, VectorXd(n_));
    X_.assign(L_, VectorXd(n_));
    penalty_.assign(L_, 0.0);
    VectorXd z(n_);
    for (std::size_t k = 0; k < L_; ++k) {
      for (int i = 0; i < n_; ++i) z[i] = rng_.normal();
      Y_[k] = B_ * D_.cwiseProduct(z);  // y ~ N(0, C)
      const VectorXd x = m_ + sigma_ * Y_[k];
      X_[k] = to_box(x);
      if (bounded_ && opt_.bound_handling == BoundHandling::ClampPenalty)
        penalty_[k] = opt_.penalty_weight * ((x - X_[k]).array() / (opt_.upper - opt_.lower).array()).square().sum();
    }
    asked_ = true;
    return X_;
  }

  // f in the same order as ask(). NaN counts as +inf
  void tell(const std::vector<double>& f) {
    if (!asked_) throw std::logic_error("cmaes: tell() needs a matching ask()");
    if (f.size() != L_) throw std::invalid_argument("cmaes: tell() needs one f per asked point");
    asked_ = false;
    ++iter_;
    evals_ += lambda_;

    std::vector<double> fr(L_);
    for (std::size_t k = 0; k < L_; ++k) {
      if (!std::isnan(f[k]) && (!has_best_ || f[k] < best_f_)) {
        has_best_ = true;
        best_f_ = f[k];
        best_x_ = X_[k];
      }
      fr[k] = std::isnan(f[k]) ? std::numeric_limits<double>::infinity() : f[k] + penalty_[k];
    }
    std::vector<std::size_t> idx(L_);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return fr[a] < fr[b]; });

    // mean
    VectorXd yw = VectorXd::Zero(n_);
    for (int i = 0; i < mu_; ++i) yw += w_[i] * Y_[idx[static_cast<std::size_t>(i)]];
    m_ += sigma_ * yw;

    // evolution paths
    const VectorXd Cinvhalf_yw = B_ * (B_.transpose() * yw).cwiseQuotient(D_);
    ps_ = (1.0 - cs_) * ps_ + std::sqrt(cs_ * (2.0 - cs_) * mueff_) * Cinvhalf_yw;
    const double ps_norm = ps_.norm();
    const bool hsig = ps_norm / std::sqrt(1.0 - std::pow(1.0 - cs_, 2.0 * iter_)) < (1.4 + 2.0 / (n_ + 1.0)) * chiN_;
    pc_ = (1.0 - cc_) * pc_ + (hsig ? std::sqrt(cc_ * (2.0 - cc_) * mueff_) : 0.0) * yw;

    // covariance: rank-one + rank-mu
    MatrixXd rank_mu = MatrixXd::Zero(n_, n_);
    for (int i = 0; i < mu_; ++i) {
      const VectorXd& y = Y_[idx[static_cast<std::size_t>(i)]];
      rank_mu.noalias() += w_[i] * y * y.transpose();
    }
    const double dh = hsig ? 0.0 : cc_ * (2.0 - cc_);
    C_ = (1.0 - c1_ - cmu_ + c1_ * dh) * C_ + c1_ * pc_ * pc_.transpose() + cmu_ * rank_mu;
    C_ = (0.5 * (C_ + C_.transpose())).eval();  // need eval() here, aliasing

    // step size (CSA)
    sigma_ *= std::exp((cs_ / damps_) * (ps_norm / chiN_ - 1.0));
    sigma_ = std::min(sigma_, opt_.sigma_max);

    // eigendecomposition is O(n^3), only redo it every few gens
    bool degenerate = false;
    if (static_cast<double>(evals_ - eigen_evals_) > lambda_ / (c1_ + cmu_) / n_ / 10.0) {
      eigen_evals_ = evals_;
      Eigen::SelfAdjointEigenSolver<MatrixXd> es(C_);
      const VectorXd ev = es.eigenvalues();
      degenerate = es.info() != Eigen::Success || !ev.allFinite() || !(ev.minCoeff() > 0.0);
      B_ = es.eigenvectors();
      D_ = ev.cwiseMax(1e-300).cwiseSqrt();
    }

    // stop checks (tutorial appendix B.3, tolupsigma is from pycma)
    best_hist_.push_back(fr[idx[0]]);
    if (static_cast<int>(best_hist_.size()) > hist_len_) best_hist_.pop_front();
    const double cond = (D_.maxCoeff() / D_.minCoeff()) * (D_.maxCoeff() / D_.minCoeff());
    if (best_f_ <= opt_.f_target) stop_ = Stop::TargetReached;
    else if (opt_.max_evals > 0 && evals_ >= opt_.max_evals) stop_ = Stop::MaxEvals;
    else if (iter_ >= max_iter_) stop_ = Stop::MaxIter;
    else if (degenerate || !(cond <= opt_.max_condition)) stop_ = Stop::ConditionCov;
    else if (tol_fun_reached(fr)) stop_ = Stop::TolFun;
    else if ((sigma_ * pc_.cwiseAbs().array() < tol_x_).all() && (sigma_ * C_.diagonal().cwiseSqrt().array() < tol_x_).all())
      stop_ = Stop::TolX;
    else if (sigma_ / sigma0_ > opt_.tol_up_sigma * D_.maxCoeff()) stop_ = Stop::TolUpSigma;
    else if (m_ == m_ + 0.1 * sigma_ * D_[iter_ % n_] * B_.col(iter_ % n_)) stop_ = Stop::NoEffectAxis;
    else if (((m_ + 0.2 * sigma_ * C_.diagonal().cwiseSqrt()).array() == m_.array()).any()) stop_ = Stop::NoEffectCoord;
  }

  bool done() const { return stop_ != Stop::None; }
  Stop stop() const { return stop_; }

  // use mean() over best_x() when f is noisy
  VectorXd mean() const { return to_box(m_); }
  const VectorXd& best_x() const { return best_x_; }  // x0 until a non-NaN f comes back
  double best_f() const { return best_f_; }

  int dim() const { return n_; }
  int lambda() const { return lambda_; }
  int mu() const { return mu_; }
  int iterations() const { return iter_; }
  long long evaluations() const { return evals_; }
  int max_iter() const { return max_iter_; }
  double sigma() const { return sigma_; }
  const MatrixXd& cov() const { return C_; }
  const VectorXd& weights() const { return w_; }
  double mueff() const { return mueff_; }
  double cc() const { return cc_; }
  double cs() const { return cs_; }
  double c1() const { return c1_; }
  double cmu() const { return cmu_; }
  double damps() const { return damps_; }
  double chiN() const { return chiN_; }

 private:
  VectorXd to_box(const VectorXd& x) const {
    if (!bounded_) return x;
    if (opt_.bound_handling == BoundHandling::ClampPenalty) return x.cwiseMax(opt_.lower).cwiseMin(opt_.upper);
    VectorXd y = x;
    for (int i = 0; i < n_; ++i) {
      const double lo = opt_.lower[i], up = opt_.upper[i], r = up - lo;
      if (x[i] >= lo && x[i] <= up) continue;  // already inside
      // reflect off the bound it crossed (period 2r)
      const bool below = x[i] < lo;
      const double t = std::fmod(below ? lo - x[i] : x[i] - up, 2.0 * r);
      const double v = t <= r ? (below ? lo + t : up - t) : (below ? up - (t - r) : lo + (t - r));
      y[i] = std::min(up, std::max(lo, v));
    }
    return y;
  }

  bool tol_fun_reached(const std::vector<double>& fr) const {
    if (static_cast<int>(best_hist_.size()) < hist_len_) return false;
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (double v : best_hist_) lo = std::min(lo, v), hi = std::max(hi, v);
    for (double v : fr) lo = std::min(lo, v), hi = std::max(hi, v);
    return (hi == lo ? 0.0 : hi - lo) < opt_.tol_fun;  // hi == lo so all-inf counts as flat
  }

  Options opt_;
  int n_, lambda_ = 0, mu_ = 0, max_iter_ = 0, hist_len_ = 0;
  std::size_t L_ = 0;  // lambda_ as size_t
  bool bounded_ = false, asked_ = false, has_best_ = false;
  double sigma0_, sigma_, tol_x_ = 0;
  double mueff_ = 0, cc_ = 0, cs_ = 0, c1_ = 0, cmu_ = 0, damps_ = 0, chiN_ = 0;
  VectorXd w_, m_, pc_, ps_, D_, best_x_;
  MatrixXd C_, B_;
  std::vector<VectorXd> Y_, X_;
  std::vector<double> penalty_;
  std::deque<double> best_hist_;
  int iter_ = 0;
  long long evals_ = 0, eigen_evals_ = 0;
  double best_f_ = std::numeric_limits<double>::infinity();
  Stop stop_ = Stop::None;
  detail::Rng rng_;
};

struct Result {
  VectorXd x;             // best point found
  double f = std::numeric_limits<double>::infinity();
  VectorXd mean;          // final mean, use this if f is noisy
  Stop stop = Stop::None;
  int iterations = 0;
  long long evaluations = 0;
  int restarts = 0;
  int lambda = 0;         // pop size of the last run
};

// fb scores a whole generation at once (threads, common random numbers, ...)
// restarts > 0 = IPOP: rerun from x0 with a bigger population
template <class BatchF>
Result minimize_batch(BatchF&& fb, const VectorXd& x0, double sigma0, const Options& opt = Options()) {
  if (x0.size() < 1) throw std::invalid_argument("cmaes: x0 must have at least one coordinate");
  if (opt.restarts < 0) throw std::invalid_argument("cmaes: restarts must be >= 0");
  Result r;
  const int lambda0 = opt.lambda > 0 ? opt.lambda : detail::default_lambda(static_cast<int>(x0.size()));
  int lambda = lambda0;
  for (int k = 0; k <= opt.restarts; ++k) {
    Options o = opt;
    o.lambda = lambda;
    // keep a custom mu proportional to lambda
    if (opt.mu > 0) o.mu = std::max(1, static_cast<int>(std::lround(static_cast<double>(opt.mu) * lambda / lambda0)));
    o.seed = opt.seed + std::uint64_t{0x9E3779B97F4A7C15ULL} * static_cast<std::uint64_t>(k);
    if (opt.max_evals > 0) {
      o.max_evals = opt.max_evals - r.evaluations;
      if (o.max_evals <= 0) break;
    }
    CMAES es(x0, sigma0, o);
    while (!es.done()) {
      const std::vector<VectorXd>& X = es.ask();
      const std::vector<double> f = fb(X);
      es.tell(f);
    }
    if (es.best_f() < r.f || r.x.size() == 0) {
      r.f = es.best_f();
      r.x = es.best_x();
    }
    r.mean = es.mean();
    r.stop = es.stop();
    r.iterations += es.iterations();
    r.evaluations += es.evaluations();
    r.restarts = k;
    r.lambda = lambda;
    if (r.stop == Stop::TargetReached || r.stop == Stop::MaxEvals) break;
    if (lambda == std::numeric_limits<int>::max()) break;
    const double grown = std::round(lambda * opt.ipop_factor);
    lambda = grown > lambda ? static_cast<int>(std::min<double>(grown, std::numeric_limits<int>::max())) : lambda + 1;
  }
  return r;
}

// same thing, one point at a time
template <class F>
Result minimize(F&& f, const VectorXd& x0, double sigma0, const Options& opt = Options()) {
  return minimize_batch(
      [&f](const std::vector<VectorXd>& X) {
        std::vector<double> v(X.size());
        for (std::size_t k = 0; k < X.size(); ++k) v[k] = f(X[k]);
        return v;
      },
      x0, sigma0, opt);
}

}  // namespace cmaes
