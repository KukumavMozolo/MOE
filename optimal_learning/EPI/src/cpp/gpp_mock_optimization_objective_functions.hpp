// gpp_mock_optimization_objective_functions.hpp
/*
  This file contains mock objective functions *interfaces* for use with optimization routines.  This is solely for unit testing.
  Instead of testing gradient descent against log marginal likelihood with some random set of data (which is an
  integration test), we would instead like to be able to test gradient descent on something easier to understand,
  e.g., z = -x^2 - y^2.  These simpler functions have analytic optima which makes testing optimizers
  (e.g., gradient descent, newton) much easier.

  See the header comments in gpp_optimization.hpp, Section 3a), for further details on what it means for an
  Evaluator class to be "optimizable."

  Since perfomance is irrelevant for these test functions, we will define pure abstract Evaluator classes that can
  be used to test optimizers in gpp_optimization.hpp.  Generally the usage should be to subclass say PolynomialEvaluator
  and override the pure virtuals.  Then we only end up compiling one version of the [templated] optimization code
  for running tests.  See gpp_optimization_test.cpp for examples.

  Following the style laid out in gpp_common.hpp (file comments, item 5), we currently define:
  class PolynomialEvaluator;
  struct PolynomialState;

  PolynomialEvaluator defines a pure abstract base class with interface consistent with the interface that all
  .*Evaluator classes must provide (e.g., ExpectedImprovementEvaluator, LogMarginalLikelihoodEvaluator).

  PolynomialState is simple: it's just a container class that holds a point at which to evaluate the polynomial.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MOCK_OPTIMIZATION_OBJECTIVE_FUNCTIONS_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MOCK_OPTIMIZATION_OBJECTIVE_FUNCTIONS_HPP_

#include <algorithm>
#include <vector>

#include "gpp_common.hpp"

namespace optimal_learning {

struct PolynomialState;

/*
  Class to evaluate the function f(x_1,...,x_{dim}) = -\sum_i (x_i - s_i)^2, i = 1..dim.
  This is a simple quadratic form with maxima at (s_1, ..., s_{dim}).
*/
class PolynomialEvaluator {
 public:
  using StateType = PolynomialState;

  virtual ~PolynomialEvaluator() = default;

  virtual int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*
    Helpful for testing so we know what the optimum is.  This value should be the result of:
      GetOptimumPoint(point);
      state.UpdateCurrentPoint(point);
      optimum_value = ComputeObjectiveFunction(state);

    Then optimum_value == GetOptimumValue().

    RETURNS:
    the optimum value of the polynomial.
  */
  virtual double GetOptimumValue() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*
    Helpful for testing so we know where the optimum (value returned by GetOptimumValue) occurs.

    NOTE: if the optimal point is not unique, this function may return one arbitrarily.

    INPUTS:
    point[dim]: space to write the output
    OUTPUTS:
    point[dim]: the point at which the polynomial obtains its optimum value

  */
  virtual void GetOptimumPoint(double * restrict point) const noexcept OL_NONNULL_POINTERS = 0;

  /*
  */
  virtual double ComputeObjectiveFunction(StateType * quadratic_dummy_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT = 0;

  /*
  */
  virtual void ComputeGradObjectiveFunction(StateType * quadratic_dummy_state, double * restrict grad_polynomial) const OL_NONNULL_POINTERS = 0;

  /*
  */
  virtual void ComputeHessianObjectiveFunction(StateType * quadratic_dummy_state, double * restrict hessian_polynomial) const OL_NONNULL_POINTERS = 0;

  OL_DISALLOW_COPY_AND_ASSIGN(PolynomialEvaluator);

 protected:
  PolynomialEvaluator() = default;
};

struct PolynomialState final {
  using EvaluatorType = PolynomialEvaluator;

  PolynomialState(const EvaluatorType& quadratic_eval, double const * restrict current_point_in)
      : dim(quadratic_eval.dim()),
        current_point(current_point_in, current_point_in + dim) {
  }

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim;
  }

  void GetCurrentPoint(double * restrict current_point_in) const noexcept OL_NONNULL_POINTERS {
    std::copy(current_point.begin(), current_point.end(), current_point_in);
  }

  void UpdateCurrentPoint(const EvaluatorType& OL_UNUSED(ei_eval), double const * restrict current_point_in) noexcept OL_NONNULL_POINTERS {
    std::copy(current_point_in, current_point_in + dim, current_point.begin());
  }

  const int dim;  // spatial dimension (e.g., entries per point of points_sampled)

  // state variables
  std::vector<double> current_point;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PolynomialState);
};

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_MOCK_OPTIMIZATION_OBJECTIVE_FUNCTIONS_HPP_
