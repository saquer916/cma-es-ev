// 10-D rosenbrock with restarts. set f_target or it keeps restarting even after it solves it
#include <cmaes/cmaes.hpp>

#include <cmath>
#include <cstdio>

int main() {
  auto rosenbrock = [](const Eigen::VectorXd& x) {
    double s = 0;
    for (int i = 0; i + 1 < x.size(); ++i) s += 100 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1 - x[i], 2);
    return s;
  };
  cmaes::Options opt;
  opt.restarts = 3;
  opt.f_target = 1e-10;
  const cmaes::Result r = cmaes::minimize(rosenbrock, Eigen::VectorXd::Zero(10), 0.5, opt);
  std::printf("f = %.3g after %lld evaluations and %d restarts (stopped on %s)\nx =", r.f, r.evaluations, r.restarts,
              cmaes::to_string(r.stop));
  for (int i = 0; i < r.x.size(); ++i) std::printf(" %.6f", r.x[i]);
  std::printf("\n");
}
