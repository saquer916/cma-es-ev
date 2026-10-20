// tuning 2 constants of a noisy sim with ask/tell
// toy model: cart drives 8 m, brakes at some distance before the target, slides to a stop.
// floor friction is random per trial. cost = 100 * |miss in m| + time in s
//
// every candidate in a generation runs the same trials (common random numbers) so the ranking
// compares designs and not luck. new trials each generation so it can't overfit one set.
// search is in [0,1]^2 and each coordinate gets scaled to its real range
#include <cmaes/cmaes.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

struct Range { const char* name; double lo, hi; };
static const Range kRanges[] = {{"cruise speed (m/s)", 0.5, 3.0}, {"brake distance (m)", 0.0, 3.0}};

static double decode(const Eigen::VectorXd& x, int i) { return kRanges[i].lo + x[i] * (kRanges[i].hi - kRanges[i].lo); }

static double trial(double speed, double brake_at, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> friction(0.25, 0.35);
  const double decel = friction(g) * 9.81, target = 8.0;
  const double x_brake = std::max(0.0, target - brake_at);
  const double stop = x_brake + speed * speed / (2.0 * decel);
  return 100.0 * std::abs(stop - target) + x_brake / speed + speed / decel;
}

static double mean_cost(const Eigen::VectorXd& x, std::uint64_t first_seed, int trials) {
  double s = 0;
  for (int t = 0; t < trials; ++t) s += trial(decode(x, 0), decode(x, 1), first_seed + t);
  return s / trials;
}

int main() {
  const int trials = 64;
  cmaes::Options opt;
  opt.lower = Eigen::VectorXd::Zero(2);
  opt.upper = Eigen::VectorXd::Ones(2);
  opt.max_iter = 150;
  opt.seed = 2027;
  cmaes::CMAES es(Eigen::VectorXd::Constant(2, 0.5), 0.3, opt);

  while (!es.done()) {
    const std::vector<Eigen::VectorXd>& X = es.ask();
    const std::uint64_t seed = 1000003ULL * static_cast<std::uint64_t>(es.iterations());  // new trials every gen
    std::vector<double> f(X.size());
    std::vector<std::thread> pool;
    for (size_t k = 0; k < X.size(); ++k) pool.emplace_back([&, k] { f[k] = mean_cost(X[k], seed, trials); });
    for (std::thread& t : pool) t.join();
    es.tell(f);
    if (es.iterations() % 25 == 0) {
      const Eigen::VectorXd m = es.mean();
      std::printf("gen %3d  speed %.4f  brake %.4f  sigma %.4f\n", es.iterations(), decode(m, 0), decode(m, 1), es.sigma());
    }
  }

  // noisy cost, so use the mean (best_x is just the luckiest sample) and check it on fresh trials
  const Eigen::VectorXd x = es.mean();
  std::printf("\nstopped on %s after %d generations\n", cmaes::to_string(es.stop()), es.iterations());
  for (int i = 0; i < 2; ++i) std::printf("  %-20s %.4f\n", kRanges[i].name, decode(x, i));
  std::printf("  cost on 10000 fresh trials: %.3f\n", mean_cost(x, 0xC0FFEEULL, 10000));
}
