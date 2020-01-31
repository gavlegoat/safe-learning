#include <optional>
#include <random>

//#include <glpk.h>
#include <Python.h>

#include "abstract.hpp"

#define MAX_SPLITS 1
#define ABSTRACT_DOMAIN AbstractDomain::ZONOTOPE

static PyObject* DomainError;

struct Interval {
  Eigen::MatrixXd lower;
  Eigen::MatrixXd upper;
};

/**
 * A Space is just a polytope along with its bounding box.
 */
struct Space {
  LinCons space;
  Eigen::VectorXd bb_lower;
  Eigen::VectorXd bb_upper;
};

/**
 * A linear controller.
 */
struct Controller {
  /** The controller itself: the action is u = K x */
  Eigen::MatrixXd k;
  /** A region for which this controller is invariant. */
  LinCons invariant;
  /** The space this controller is intended to cover. */
  Space space;
};

class Environment {
  public:
    /** True if the environment uses continuous semantics. */
    bool continuous;
    /** The time step for continuous environments. */
    double dt;
    /** The safe part of the state space. */
    std::vector<LinCons> unsafe_space;

    Environment(bool c, double d, const std::vector<LinCons>& unsafe):
      continuous(c), dt(d), unsafe_space(unsafe) {}

    /**
     * Take a concrete step in this environment.
     *
     * \param state The current state of the system.
     * \param controller The controller to use for this step.
     * \return The new state of the system.
     */
    virtual Eigen::VectorXd step(const Eigen::VectorXd& state,
        const Eigen::MatrixXd& controller) const = 0;

    /**
     * Take a step using an abstract state and a concrete controller.
     *
     * \param state The (asbstract) state of the system.
     * \param controller The (concrete) controller to use.
     * \return The new (abstract) state of the system.
     */
    virtual std::unique_ptr<AbstractVal> semi_abstract_step(
        const AbstractVal& state, const Eigen::MatrixXd& controller) const = 0;

    /**
     * Take a step using an abstract state and an interval of controllers.
     *
     * \param state The current system state.
     * \param controller The interval of possible controllers.
     * \return The new system state.
     */
    virtual std::unique_ptr<AbstractVal> abstract_step(
        const AbstractVal& state, const Interval& controller) const = 0;

    /**
     * Find the largest invariant for some controller.
     *
     * In general a controller may be safe for a region larger than the cover
     * for which it was computed. Here we want to find the largest region in which
     * the given controller is safe.
     *
     * \param env The environment under control.
     * \param cover The initial space covered by the controller.
     * \param bound The bound on the time horizon.
     * \param other_covers The regions covered by other controllers.
     * \param k The controller.
     * \return The region over which the controller is safe.
     */
    virtual LinCons compute_invariant(const Space& cover, int bound,
        const std::vector<LinCons>& other_covers,
        const Eigen::MatrixXd& k) const = 0;

    virtual ~Environment() = default;
};

/**
 * A linear environment.
 *
 * The environment behavior is defined as follows: if `continuous` is true
 * then \f$\dot{x} = A x + B u\f$ where \f$x\f$ is the state and \f$u\f$ is an
 * action. This continuous environment is discretized with a time step `dt`.
 * If `continuous` is false then \f$x' = A x + B u\f$ where \f$x'\f$ is the
 * state in the next time step and `dt` is not used. In either case,
 * `unsafe_space` defines the unsafe part of the state space.
 */
class LinearEnv: public Environment {
  public:
    /** An environment transition matrix. */
    Eigen::MatrixXd A;
    /** An environment transition matrix. */
    Eigen::MatrixXd B;

    LinearEnv(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, bool c,
        double d, const std::vector<LinCons>& unsafe):
      Environment(c, d, unsafe), A(a), B(b) {}

    Eigen::VectorXd step(const Eigen::VectorXd& state,
        const Eigen::MatrixXd& controller) const override {
      if (continuous) {
        return state + dt * (A * state + B * controller * state);
      } else {
        return A * state + B * controller * state;
      }
    }

    std::unique_ptr<AbstractVal> semi_abstract_step(const AbstractVal& state,
        const Eigen::MatrixXd& controller) const override {
      if (continuous) {
        // x' = x + dt (A x + B K x) = x + dt (A + B K) x
        //    = (I + dt * (A + B K)) x
        Eigen::MatrixXd transition =
          Eigen::MatrixXd::Identity(A.rows(), A.rows()) +
          dt * (A + B * controller);
        return state.scalar_affine(transition,
            Eigen::VectorXd::Zero(A.rows()));
      } else {
        return state.scalar_affine(A + B * controller,
            Eigen::VectorXd::Zero(A.rows()));
      }
    }

    std::unique_ptr<AbstractVal> abstract_step(const AbstractVal& state,
        const Interval& controller) const override {
      // Construct upper and lower bounds on the transition matrix. For a
      // concrete matrix we have W = A + B K for discrete environments or
      // W = I + dt (A + B K) = I + dt A + dt B K for continuous environments.
      Eigen::MatrixXd w_lower, w_upper;
      if (continuous) {
        w_lower = Eigen::MatrixXd::Identity(A.rows(), A.rows()) + dt * A;
        w_upper = Eigen::MatrixXd::Identity(A.rows(), A.rows()) + dt * A;
      } else {
        w_lower = A;
        w_upper = A;
      }
      // At this point w_lower and w_upper are both equal to W - B K.
      // Now we compute B K element-wise so that we can get appropriate bounds
      for (int i = 0; i < A.rows(); i++) {
        for (int j = 0; j < A.rows(); j++) {
          // Find a min of (B K)(i,j) for K in controller.
          double min_bk = 0.0;
          double max_bk = 0.0;
          for (int k = 0; k < B.cols(); k++) {
            if (B(i,k) < 0) {
              min_bk += B(i,k) * controller.upper(k,j);
              max_bk += B(i,k) * controller.lower(k,j);
            } else {
              min_bk += B(i,k) * controller.lower(k,j);
              max_bk += B(i,k) * controller.upper(k,j);
            }
          }
          if (continuous) {
            w_lower(i,j) += dt * min_bk;
            w_upper(i,j) += dt * max_bk;
          } else {
            w_lower(i,j) += min_bk;
            w_upper(i,j) += max_bk;
          }
        }
      }
      Eigen::VectorXd bias = Eigen::VectorXd::Zero(A.rows());
      return state.interval_affine(w_lower, w_upper, bias, bias);
    }

    LinCons compute_invariant(const Space& cover,
        int bound, const std::vector<LinCons>& other_covers,
        const Eigen::MatrixXd& k) const override {
      if (bound <= 0) {
      //if (true) {
        // TODO
        return cover.space;
      }
      // Find an invariant for x' = (A + B K) x or x' = (I + dt * (A + B K)) x
      // For the bounded case, find the maximum space such that `bound` iterations
      // are safe.
      Eigen::MatrixXd transition;
      if (continuous) {
        transition = Eigen::MatrixXd::Identity(A.rows(), A.rows()) +
          dt * (A + B * k);
      } else {
        transition = A + B * k;
      }

      // Find X such that for all x \in X, T * x \in Safe where T is the n-step
      // transition matrix and Safe is the safe region. Then if the _unsafe_ region
      // is defined by A x < b, we need to have
      // A (T x) >= b ==> (A T) x >= b ==> (- A T) x <= - b

      Eigen::MatrixXd n_step = Eigen::MatrixXd::Identity(A.rows(), A.rows());
      std::vector<Eigen::VectorXd> constraints;
      std::vector<double> coeffs;
      for (int i = 0; i < bound; i++) {
        for (const LinCons& lc : unsafe_space) {
          Eigen::MatrixXd m = -lc.weights * n_step;
          for (int i = 0; i < m.rows(); i++) {
            constraints.push_back(m.row(i));
            coeffs.push_back(-lc.biases(i));
          }
        }
        n_step *= transition;
      }

      // FUTURE: Remove redundant constraints.
      // This procedure produces a lot more constraints than necessary, but I
      // think it should be okay since we only use these constraints a few times
      // and the extra constraints don't compound.

      Eigen::MatrixXd ws(constraints.size(), constraints[0].size());
      Eigen::VectorXd bs(constraints.size());
      for (int i = 0; i < ws.rows(); i++) {
        ws.row(i) = constraints[i];
        bs(i) = coeffs[i];
      }

      return LinCons(ws, bs);
    }
};

class NonlinearEnv: public Environment {
  public:
    std::vector<ArithExpr> update;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
        const Eigen::VectorXd&)> concrete;

    NonlinearEnv(std::function<Eigen::VectorXd(const Eigen::VectorXd&,
          const Eigen::VectorXd&)> con,
        const std::vector<ArithExpr>& up, bool c, double d,
        const std::vector<LinCons>& unsafe):
      Environment(c, d, unsafe), concrete(con), update(up) {}

    Eigen::VectorXd step(const Eigen::VectorXd& state,
        const Eigen::MatrixXd& controller) const override {
      return concrete(state, controller * state);
    }

    std::unique_ptr<AbstractVal> semi_abstract_step(const AbstractVal& state,
        const Eigen::MatrixXd& controller) const override {
      auto action = state.scalar_affine(controller,
          Eigen::VectorXd::Zero(controller.rows()));
      auto combined = state.append(*action);
      //std::cout << "State with action appended" << std::endl;
      //combined->print(stdout);
      auto new_state = combined->arith_computation(update);
      if (continuous) {
        int n = update.size();
        auto both_states = state.append(*new_state);
        Eigen::MatrixXd tr = Eigen::MatrixXd::Identity(n, 2 * n);
        tr.block(0, n, n, n) = dt * Eigen::MatrixXd::Identity(n, n);
        return both_states->scalar_affine(tr, Eigen::VectorXd::Zero(n));
      } else {
        return new_state;
      }
    }

    std::unique_ptr<AbstractVal> abstract_step(const AbstractVal& state,
        const Interval& controller) const override {
      auto action = state.interval_affine(controller.lower, controller.upper,
          Eigen::VectorXd::Zero(state.dims()),
          Eigen::VectorXd::Zero(state.dims()));
      auto combined = state.append(*action);
      auto new_state = combined->arith_computation(update);
      if (continuous) {
        int n = update.size();
        auto both_states = state.append(*new_state);
        Eigen::MatrixXd tr = Eigen::MatrixXd::Identity(n, 2 * n);
        tr.block(0, n, n, n) = dt * Eigen::MatrixXd::Identity(n, n);
        return both_states->scalar_affine(tr, Eigen::VectorXd::Zero(n));
      } else {
        return new_state;
      }
    }

    LinCons compute_invariant(const Space& cover,
        int bound, const std::vector<LinCons>& other_covers,
        const Eigen::MatrixXd& k) const override {
      // `cover` should already be safe because of the properties of the
      // synthesis algorithm. In this method we just need to expand out as
      // much as possible.
      // TODO
      return cover.space;
    }
};

struct PythonCapsule {
  std::vector<ArithExpr> update;
  std::function<Eigen::VectorXd(const Eigen::VectorXd&,
      const Eigen::VectorXd&)> concrete;
};


/**
 * Determine whether an interval of controllers is safe.
 *
 * Given an interval [Kl, Ku], determines whether K is safe for all
 * Kl <= K <= Ku (where the comparisons are element-wise).
 *
 * \param itv The set of controllers to check.
 * \param env The environment under control.
 * \param cover The initial space the controller should cover.
 * \param other_covers Regions covered by other controllers.
 * \param bound The bound on the time horizon.
 * \return True if all of the controllers in `itv` are safe.
 */
bool interval_is_safe(const Interval& itv, const Environment& env,
    const Space& cover, const std::vector<LinCons>& other_covers,
    int bound) {
  auto state = std::make_unique<AbstractVal>(ABSTRACT_DOMAIN,
      cover.bb_lower, cover.bb_upper);
  state = state->meet_linear_constraint(cover.space.weights,
      cover.space.biases);

  //std::cout << "Verifying interval" << std::endl;
  //std::cout << itv.lower.transpose() << std::endl;
  //std::cout << itv.upper.transpose() << std::endl;
  if (bound > 0) {
    for (int i = 0; i < bound; i++) {
      //state->print(stdout);
      auto next = env.abstract_step(*state, itv);
      state = state->join(*next);
    }
  } else {
    while (true) {
      auto next = env.abstract_step(*state, itv);
      auto old_state = state->clone();
      state = state->widen(*state->join(*next));
      if (*old_state == *state) {
        break;
      }
    }
  }
  for (const LinCons& lc : env.unsafe_space) {
    if (!state->meet_linear_constraint(lc.weights, lc.biases)->is_bottom()) {
      //std::cout << "Unsafe" << std::endl;
      //auto t = state->meet_linear_constraint(lc.weights, lc.biases);
      //std::cout << lc.weights << std::endl;
      //std::cout << lc.biases << std::endl;
      //t->print(stdout);
      //for (const LinCons& lc : env.unsafe_space) {
      //  std::cout << lc.weights << std::endl;
      //  std::cout << lc.biases << std::endl;
      //}
      //throw std::runtime_error("");
      return false;
      // FUTURE: Deal with covers
    }
  }
  //std::cout << "Safe" << std::endl;
  return true;
}

/**
 * Determine whether a controller is safe in its region.
 *
 * Assuming safe controllers exist which cover each element of `covers`,
 * this function determines whether the given controller is safe in its
 * declared cover.
 *
 * \param env The environment under control.
 * \param controller The controller to check.
 * \param covers The regions covered by other controllers.
 * \return True if `controller` is safe for the region it covers.
 */
bool controller_is_safe(const Environment& env, const Controller& controller,
    const std::vector<LinCons>& covers, int bound) {
  auto state = std::make_unique<AbstractVal>(ABSTRACT_DOMAIN,
      controller.invariant);
  if (bound > 0) {
    for (int i = 0; i < bound; i++) {
      state = env.semi_abstract_step(*state, controller.k);
    }
  } else {
    while (true) {
      auto next = env.semi_abstract_step(*state, controller.k);
      auto old_state = state->clone();
      state = state->widen(*state->join(*next));
      if (*old_state == *state) {
        break;
      }
    }
  }
  for (const LinCons& lc : env.unsafe_space) {
    if (!state->meet_linear_constraint(lc.weights, lc.biases)->is_bottom()) {
      return false;
      // FUTURE: Deal with covers
    }
  }
  return true;
}

/**
 * Measure the safety of a controller in an environment.
 *
 * We measure safety by sampling initial states, evolving the system for some
 * time, then seeing how close the result is to the unsafe states.
 *
 * \param env The environment under control.
 * \param k The controller.
 * \param initial The states the controller should be safe in.
 * \return A measure of the safety of this controller.
 */
double measure_safety(const Environment& env, const Eigen::MatrixXd& k,
    const Space& initial, int bound) {
  int iters = 50;
  double total = 0.0;
  //Eigen::MatrixXd update = env.A + env.B * k;
  //Eigen::MatrixXd n_step = Eigen::MatrixXd::Identity(
  //    update.rows(), update.rows());
  //for (int j = 0; j < bound; j++) {
  //  n_step *= update;
  //}
  for (int i = 0; i < iters; i++) {
    // Sample x from the initial space.
    Eigen::VectorXd x;
    while (true) {
      x = initial.bb_lower + Eigen::VectorXd::Random(
          initial.bb_lower.size()).cwiseProduct(
          initial.bb_upper - initial.bb_lower);
      Eigen::VectorXd ax = initial.space.weights * x;
      bool inside = true;
      for (int j = 0; j < ax.size(); j++) {
        if (ax(j) > initial.space.biases(j)) {
          inside = false;
          break;
        }
      }
      if (inside) {
        break;
      }
    }
    int is = bound > 0 ? bound : 20;
    // See how safe x is.
    for (int j = 0; j < is; j++) {
      x = env.step(x, k);
    }
    //x.applyOnTheLeft(n_step);
    double min = std::numeric_limits<double>::max();
    for (const LinCons& lc : env.unsafe_space) {
      if (lc.distance_from(x) < min) {
        min = lc.distance_from(x);
      }
    }
    total += min;
  }
  return total / iters;
}

/**
 * Find an unsafe controller in a given interval.
 *
 * We do this with the same approximated gradient descent used for the overall
 * shield synthesis.
 *
 * \param env The environment under control.
 * \param cover The initial space in which we need to be safe.
 * \param other_covers Spaces where other controllers exist.
 * \param itv The interval in which to search.
 * \param The bound on the time horizon.
 * \return An unsafe controller if one can be found.
 */
std::optional<Eigen::MatrixXd> find_counterexample(const Environment& env,
    const Space& cover, const std::vector<LinCons>& other_covers,
    const Interval& itv, int bound) {
  // Start from the center of the given space.
  Eigen::MatrixXd k = (itv.lower + itv.upper) / 2;
  Controller contr = {
    .k = k,
    .invariant = cover.space,
    .space = cover
  };
  double lr = 0.05;  // originally 0.005
  double v = 0.08;   // oroginally 0.04
  for (int i = 0; i < 30; i++) {   // originally 200
    if (!controller_is_safe(env, contr, other_covers, bound)) {
      // If the controller is unsafe then we've found a counterexample.
      return k;
    }
    Eigen::MatrixXd delta = Eigen::MatrixXd::Random(k.rows(), k.cols());
    double sim_plus = measure_safety(env, k + v * delta, cover, bound);
    double sim_minus = measure_safety(env, k - v * delta, cover, bound);
    if (sim_plus <= 0.0) {
      // k + v * delta has a non-positive safety score, so it is unsafe.
      return k + v * delta;
    } else if (sim_minus <= 0.0) {
      return k - v * delta;
    }
    Eigen::MatrixXd grad = (sim_plus - sim_minus) / v * delta;
    k -= lr * grad;
    k = k.cwiseMax(itv.lower).cwiseMin(itv.upper);
  }
  return {};
}

/**
 * Find a safe region in the parameter space around `k`.
 *
 * Given an environment and a current (safe) controller `k`, find an interval
 * of safe parameters. This first tries to verify an interval of size
 * `step_size` around the initial controller. If it is unable to do so, then
 * it starts shrinking the space until it finds an interval that can be
 * verified.
 *
 * \param env The environment under control.
 * \param cover The initial space where we need to be safe.
 * \param other_covers Spaces where other controllers exist.
 * \param k The controller to use as a starting point.
 * \param step_size The maximum step size for gradient descent.
 * \param bound The bound on the time horizon.
 * \return A region of the controller space which is safe on `cover`.
 */
std::optional<Interval> compute_safe_space(
    const Environment& env, const Space& cover,
    const std::vector<LinCons>& other_covers, const Eigen::MatrixXd& k,
    double step_size, int bound) {
  Interval itv = {
    .lower = k - Eigen::MatrixXd::Constant(k.rows(), k.cols(), step_size),
    .upper = k + Eigen::MatrixXd::Constant(k.rows(), k.cols(), step_size)
  };
  int iters = 0;
  while (!interval_is_safe(itv, env, cover, other_covers, bound)) {
    auto ce = find_counterexample(env, cover, other_covers,
        itv, bound);
    if (!ce) {
      // If we couldn't find a counterexample we just shrink the entire space.
      for (int i = 0; i < itv.lower.rows(); i++) {
        for (int j = 0; j < itv.lower.cols(); j++) {
          double c = k(i,j);
          itv.lower(i,j) = itv.lower(i,j) + (c - itv.lower(i,j)) / 2;
          itv.upper(i,j) = itv.upper(i,j) - (itv.upper(i,j) - c) / 2;
        }
      }
    } else {
      // Cut out the counterexample. We do this by pushing the closest face of
      // the interval inward until the counterexample is excluded.
      Eigen::MatrixXd bad_k = ce.value();
      for (int i = 0; i < bad_k.rows(); i++) {
        for (int j = 0; j < bad_k.cols(); j++) {
          //double c = (itv.lower(i,j) + itv.upper(i,j)) / 2;
          double c = k(i,j);
          //if (itv.upper(i,j) - bad_k(i,j) < bad_k(i,j) - itv.lower(i,j)) {
          if (bad_k(i,j) > c) {
            itv.upper(i,j) = (c + bad_k(i,j)) / 2;
          } else {
            itv.lower(i,j) = (c + bad_k(i,j)) / 2;
          }
        }
      }
    }
    iters++;
    if (iters > 20) {
      return {};
    }
  }

  return itv;
}

// Convert a python list of doubles to an Eigen::VectorXd
static Eigen::VectorXd pylist_to_vector(PyObject* list) {
  Py_ssize_t size = PyList_Size(list);
  Eigen::VectorXd ret(size);
  for (Py_ssize_t i = 0; i < size; i++) {
    PyObject* elem = PyList_GetItem(list, i);
    ret(i) = PyFloat_AsDouble(elem);
  }
  return ret;
}

static Eigen::MatrixXd pylist_to_matrix(PyObject* list) {
  Py_ssize_t rows = PyList_Size(list);
  if (rows == 0) {
    return Eigen::MatrixXd(0, 0);
  }
  PyObject* first = PyList_GetItem(list, 0);
  Py_ssize_t cols = PyList_Size(first);
  Eigen::MatrixXd ret(rows, cols);
  for (Py_ssize_t i = 0; i < rows; i++) {
    Eigen::VectorXd row = pylist_to_vector(PyList_GetItem(list, i));
    ret.row(i) = row;
  }
  return ret;
}

static std::vector<Eigen::MatrixXd> pylist_to_matrix_list(PyObject* list) {
  Py_ssize_t size = PyList_Size(list);
  std::vector<Eigen::MatrixXd> ret;
  for (int i = 0; i < size; i++) {
    ret.push_back(pylist_to_matrix(PyList_GetItem(list, i)));
  }
  return ret;
}

static std::vector<Space> pylist_to_space(PyObject* list) {
  Py_ssize_t len = PyList_Size(list);
  std::vector<Space> ret = {};
  for (Py_ssize_t i = 0; i < len; i++) {
    PyObject* lc = PyList_GetItem(list, i);
    Eigen::MatrixXd a = pylist_to_matrix(PyTuple_GetItem(lc, 0));
    Eigen::VectorXd b = pylist_to_vector(PyTuple_GetItem(lc, 1));
    Eigen::VectorXd l = pylist_to_vector(PyTuple_GetItem(lc, 2));
    Eigen::VectorXd u = pylist_to_vector(PyTuple_GetItem(lc, 3));
    Space tmp = {
      .space = LinCons(a, b),
      .bb_lower = l,
      .bb_upper = u
    };
    ret.push_back(tmp);
  }
  return ret;
}

static std::vector<LinCons> pylist_to_lincons(PyObject* list) {
  Py_ssize_t len = PyList_Size(list);
  std::vector<LinCons> ret = {};
  for (Py_ssize_t i = 0; i < len; i++) {
    PyObject* lc = PyList_GetItem(list, i);
    Eigen::MatrixXd a = pylist_to_matrix(PyTuple_GetItem(lc, 0));
    Eigen::VectorXd b = pylist_to_vector(PyTuple_GetItem(lc, 1));
    ret.push_back(LinCons(a, b));
  }
  return ret;
}

// Convert an Eigen::VectorXd to a python list of doubles
static PyObject* vector_to_pylist(const Eigen::VectorXd& b) {
  PyObject* ret = PyList_New(b.size());
  for (Py_ssize_t i = 0; i < b.size(); i++) {
    PyObject* pi = PyFloat_FromDouble(b(i));
    PyList_SetItem(ret, i, pi);
  }
  return ret;
}

static PyObject* matrix_to_pylist(const Eigen::MatrixXd& m) {
  PyObject* ret = PyList_New(m.rows());
  for (Py_ssize_t i = 0; i < m.rows(); i++) {
    PyObject* pyrow = vector_to_pylist(m.row(i));
    PyList_SetItem(ret, i, pyrow);
  }
  return ret;
}

static PyObject* controller_to_pylist(const std::vector<Controller>& contr) {
  PyObject* ret = PyList_New(contr.size());
  for (Py_ssize_t i = 0; i < contr.size(); i++) {
    PyObject* k = matrix_to_pylist(contr[i].k);
    PyObject* a = matrix_to_pylist(contr[i].invariant.weights);
    PyObject* b = vector_to_pylist(contr[i].invariant.biases);
    PyObject* l = vector_to_pylist(contr[i].space.bb_lower);
    PyObject* u = vector_to_pylist(contr[i].space.bb_upper);
    PyObject* sa = matrix_to_pylist(contr[i].space.space.weights);
    PyObject* sb = vector_to_pylist(contr[i].space.space.biases);
    PyObject* c = Py_BuildValue("N(NN)(NNNN)", k, a, b, sa, sb, l, u);
    PyList_SetItem(ret, i, c);
  }
  return ret;
}

double measure_similarity(const Eigen::MatrixXd& mat, const Space& cover,
    PyObject* measure, PyObject* dataset) {
  if (measure == NULL) {
    return -mat.norm();
  }
  PyObject* K = matrix_to_pylist(mat);
  PyObject* s = Py_BuildValue("NNNN", matrix_to_pylist(cover.space.weights),
      vector_to_pylist(cover.space.biases), vector_to_pylist(cover.bb_lower),
      vector_to_pylist(cover.bb_upper));
  PyObject* args;
  if (dataset == NULL) {
    args = Py_BuildValue("NNO", K, s, Py_None);
  } else {
    args = Py_BuildValue("NNO", K, s, dataset);
  }
  PyObject* res = PyObject_CallObject(measure, args);
  if (PyErr_Occurred()) {
    PyErr_PrintEx(0);
    throw std::runtime_error("Callback failed");
  }
  PyObject* score = PyTuple_GetItem(res, 1);
  Py_XDECREF(dataset);
  dataset = PyTuple_GetItem(res, 2);
  // Dataset here is a borrowed reference but we need it to be separate.
  Py_INCREF(dataset);
  Py_DECREF(args);
  double ret = PyFloat_AsDouble(score);
  Py_DECREF(res);
  return ret;
}

Eigen::MatrixXd get_gradient(const Eigen::MatrixXd& mat, const Space& cover,
    PyObject* measure, PyObject* dataset) {
  PyObject* K = matrix_to_pylist(mat);
  PyObject* s = Py_BuildValue("NNNN", matrix_to_pylist(cover.space.weights),
      vector_to_pylist(cover.space.biases), vector_to_pylist(cover.bb_lower),
      vector_to_pylist(cover.bb_upper));
  PyObject* args;
  if (dataset == NULL) {
    args = Py_BuildValue("NNO", K, s, Py_None);
  } else {
    args = Py_BuildValue("NNO", K, s, dataset);
  }
  PyObject* res = PyObject_CallObject(measure, args);
  if (PyErr_Occurred()) {
    PyErr_PrintEx(0);
    throw std::runtime_error("Callback failed");
  }
  Py_XDECREF(dataset);
  dataset = PyTuple_GetItem(res, 2);
  // Dataset here is a borrowed reference but we need it to be separate.
  Py_INCREF(dataset);
  Py_DECREF(args);
  PyObject* grads = PyTuple_GetItem(res, 0);
  Eigen::MatrixXd ret = pylist_to_matrix(grads);
  Py_DECREF(res);
  return ret;
}

/**
 * Add a bounding box to an array of linear constraints.
 *
 * The input is assumed to be bounded. If this is not the case then this
 * function can throw an error.
 *
 * \param lc The linear constraints to bound.
 * \returns The linear constraint together with a bounding box.
 */
/*
Space lincons_to_space(const LinCons& lc) {
  // At the moment this is just based on a series of linear programming
  // problems with simple objectives.
  Eigen::VectorXd bb_l(lc.weights.cols());
  Eigen::VectorXd bb_u(lc.weights.cols());
  glp_smcp params;
  glp_init_smcp(&params);
  params.msg_lev = GLP_MSG_OFF;
  for (int d = 0; d < lc.weights.cols(); d++) {
    for (auto dir : {GLP_MIN, GLP_MAX}) {
      // Set up a new LP problem
      glp_prob* pr = glp_create_prob();
      glp_set_obj_dir(pr, dir);
      glp_add_rows(pr, lc.weights.rows());
      glp_add_cols(pr, lc.weights.cols());
      // Get the number of matrix elements
      int n = lc.weights.rows() * lc.weights.cols();
      int r = lc.weights.rows();
      // GLPK uses 1-indexing for everything so we need to allocate an extra
      // space
      double ar[n + 1];
      int ia[n + 1];
      int ja[n + 1];
      // Store the matrix
      for (int i = 0; i < lc.weights.rows(); i++) {
        for (int j = 0; j < lc.weights.cols(); j++) {
          ia[j * r + i + 1] = i + 1;
          ja[j * r + i + 1] = j + 1;
          ar[j * r + i + 1] = lc.weights(i, j);
        }
        // Each row is upper-bounded by lc.biases
        glp_set_row_bnds(pr, i + 1, GLP_UP, 0, lc.biases(i));
      }
      for (int j = 0; j < lc.weights.cols(); j++) {
        glp_set_obj_coef(pr, j + 1, 0);
        glp_set_col_bnds(pr, j + 1, GLP_FR, 0, 0);
      }
      glp_load_matrix(pr, n, ia, ja, ar);
      // The objective function is just to optimize x_d
      glp_set_obj_coef(pr, d + 1, 1);
      if (glp_simplex(pr, &params) != 0) {
        throw std::runtime_error("GLPK: Error in glp_simplex");
      }
      auto stat = glp_get_status(pr);
      double opt;
      if (stat == GLP_INFEAS || stat == GLP_NOFEAS) {
        throw std::runtime_error("GLPK: No feasible solution found");
      } else if (stat == GLP_UNBND) {
        // The problem is unbounded so we just use a large bound for x
        opt = 100.0;
      } else if (stat != GLP_OPT) {
        throw std::runtime_error("GLKP: Optimal solution not found");
      } else {
        opt = std::min(100.0, glp_get_obj_val(pr));
      }
      // Store the optimum (which is just the min or max x value) in the
      // appropriate bounds vector.
      if (dir == GLP_MIN) {
        bb_l(d) = opt;
      } else {
        bb_u(d) = opt;
      }

      glp_delete_prob(pr);
    }
  }
  return Space { .space = lc, .bb_lower = bb_l, .bb_upper = bb_u };
}
*/

/**
 * Measure the contribution of one disjunct to the loss function.
 *
 * Within the area covered by the given controller, this measures the
 * controller's similarity to a network within its cover.
 *
 * \param ctrl The controller piece to measure.
 * \param measure A python callback for measuring similarity.
 * \return The similarity of `ctrl` to some network (implicit in `measure`).
 */
double measure_piece(const Controller& ctrl, PyObject* measure) {
  //Space s = lincons_to_space(ctrl.invariant);
  double ret = measure_similarity(ctrl.k, ctrl.space, measure, NULL);
  return ret;
}

/**
 * Measure the similarity of an entire shield to a network.
 *
 * \param shield The shield to measure.
 * \param measure A python callback for measuring similarity.
 * \return The similarity between shield and the network.
 */
double measure_shield(const std::vector<Controller>& shield,
    PyObject* measure) {
  double sum = 0;
  double volume = 0;
  for (const Controller& ctrl : shield) {
    double t = 1;
    for (int i = 0; i < ctrl.space.bb_lower.size(); i++) {
      // If any of our state dimensions have width zero then this volume
      // measure breaks. We can just ignore these dimensions because they
      // don't contribute to any splits anyway.
      if (std::abs(ctrl.space.bb_upper(i) - ctrl.space.bb_lower(i))
          <= 0.00000001) {
        continue;
      }
      t *= ctrl.space.bb_upper(i) - ctrl.space.bb_lower(i);
    }
    volume += t;
    sum += t * measure_piece(ctrl, measure);
  }
  return sum / volume;
}

/**
 * Find a linear controller which is safe in part of the state space.
 *
 * The returned controller is safe starting from any state in `cover`. The
 * controller may assume that states within `other_covers` also have safe
 * controllers. Therefore, the controller doesn't actually need to maintain
 * the invariant that the state is within `cover`. As long as the state can
 * only leave `cover` by moving into some element of `other_cover` it is still
 * jointly safe with the other controllers.
 *
 * The algorithm takes an initial controller to start from. This controller
 * must be safe on `cover` given the assumptions on `other_covers`.
 *
 * \param env An environment to synthesize a controller for.
 * \param cover The area in which this controller is safe.
 * \param other_covers Areas where the system is known to be safe.
 * \param initial A controller which is thought to be safe for `cover`.
 * \return A controller covering `cover` if one exists.
 */
std::optional<Controller> synthesize_linear_controller(
    const Environment& env, const Space& cover, int bound,
    const std::vector<LinCons>& other_covers, const Eigen::MatrixXd& initial,
    PyObject* measure) {
  //std::cout << "synthesize_linear_controller" << std::endl;
  Eigen::MatrixXd k = initial;
  double lr = 0.01;
  double v = 0.1;
  int steps_per_projection = 30;
  PyObject* dataset = NULL;
  for (int i = 0; i < 20; i++) {
    std::optional<Interval> safe = compute_safe_space(
        env, cover, other_covers, k, steps_per_projection * lr / 2, bound);
    if (!safe) {
      // We can't compute a safe space, but we can just return the existing
      // controller because we know it is at least safe.
      //std::cout << "can't find a safe controller" << std::endl;
      return Controller {
        .k = k,
        .invariant = env.compute_invariant(cover, bound, other_covers, k),
        .space = cover
      };
    }

    //double ave_grad_size = 0;
    // Gradient steps
    for (int j = 0; j < steps_per_projection; j++) {
      Eigen::MatrixXd delta = Eigen::MatrixXd::Random(k.rows(), k.cols());
      double sim_plus = measure_similarity(k + v * delta, cover, measure,
          dataset);
      double sim_minus = measure_similarity(k - v * delta, cover, measure,
          dataset);
      Eigen::MatrixXd grad = (sim_plus - sim_minus) / v * delta;
      //Eigen::MatrixXd grad = -get_gradient(k, cover, measure, dataset);
      //ave_grad_size += grad.norm();
      k += lr * grad;
      k = k.cwiseMax(safe.value().lower).cwiseMin(safe.value().upper);
    }
    //ave_grad_size /= steps_per_projection;
    //std::cout << "Average gradient size in batch " << i << ": " << ave_grad_size << std::endl;
  }
  LinCons inv = env.compute_invariant(cover, bound, other_covers, k);
  Py_XDECREF(dataset);
  return Controller {
    .k = k,
    .invariant = inv,
    .space = cover
  };
}

/**
 * Find a set of controllers for a fixed set of disjuncts.
 *
 * \param env The environment under control.
 * \param covers The disjuncts to use.
 * \param inits Initial values for the matrices.
 * \param bound The bound on the time horizon.
 * \param measure A python function for measuring similarity to the network.
 * \return A controller using `covers` as its disjuncts.
 */
std::vector<Controller> synthesize_fixed_covers(const Environment& env,
    const std::vector<Space>& covers, const std::vector<Eigen::MatrixXd>& inits,
    int bound, PyObject* measure) {
  std::vector<Controller> init = {};
  std::vector<LinCons> covered;
  for (const Space& s : covers) {
    covered.push_back(s.space);
  }
  for (int i = 0; i < covers.size(); i++) {
    covered.erase(covered.begin() + i);
    auto res = synthesize_linear_controller(env, covers[i], bound, covered,
        inits[i], measure);
    if (!res) {
      throw std::runtime_error("Unable to synthesize controller");
    }
    covered.insert(covered.begin() + i, res.value().invariant);
    init.push_back(res.value());
  }
  //std::cout << "Shield size: " << init.size() << std::endl;
  //std::cout << "Covers size: " << covers.size() << std::endl;
  return init;
}

/**
 * Split a cover into two pieces.
 *
 * \param s The space to split.
 * \param d The dimension to split in.
 * \param x The value to split at.
 * \return The two resulting spaces.
 */
std::pair<Space, Space> split_cover(const Space& s, int d, double x) {
  Eigen::MatrixXd lw(s.space.weights.rows() + 1, s.space.weights.cols());
  Eigen::MatrixXd uw(s.space.weights.rows() + 1, s.space.weights.cols());
  Eigen::VectorXd lb(s.space.biases.size() + 1);
  Eigen::VectorXd ub(s.space.biases.size() + 1);
  for (int i = 0; i < lw.rows(); i++) {
    if (i < lw.rows() - 1) {
      lw.row(i) = s.space.weights.row(i);
      uw.row(i) = s.space.weights.row(i);
      lb(i) = s.space.biases(i);
      ub(i) = s.space.biases(i);
    } else {
      Eigen::VectorXd new_l = Eigen::VectorXd::Zero(s.space.weights.cols());
      Eigen::VectorXd new_u = Eigen::VectorXd::Zero(s.space.weights.cols());
      new_l(d) = 1;
      new_u(d) = -1;
      lw.row(i) = new_l;
      uw.row(i) = new_u;
      lb(i) = x;
      ub(i) = -x;
    }
  }
  Eigen::VectorXd l_bb_l = s.bb_lower;
  Eigen::VectorXd l_bb_u = s.bb_upper;
  Eigen::VectorXd u_bb_l = s.bb_lower;
  Eigen::VectorXd u_bb_u = s.bb_upper;
  l_bb_u(d) = x;
  u_bb_l(d) = x;
  Space lower = Space { .space = LinCons(lw, lb), .bb_lower = l_bb_l,
    .bb_upper = l_bb_u };
  Space upper = Space { .space = LinCons(uw, ub), .bb_lower = u_bb_l,
    .bb_upper = u_bb_u };
  return std::make_pair(lower, upper);
}

/**
 * Find a set of controllers which are safe and cover the initial space.
 *
 * This function expects that the initial spaces has been partitioned already.
 * A controller is learned for each partition which is safe in that partition.
 *
 * \param env The environment under control.
 * \param covers A partitioning of the initial space.
 * \param bound The bound on the time horizon.
 * \param measure A callback for measuring similarity to a network.
 */
std::vector<Controller> synthesize_shield(const Environment& env,
    std::vector<Space> covers, std::vector<Eigen::MatrixXd> inits,
    int bound, PyObject* measure) {
  // covers and inits are passed by value becuase we need to copy it to make
  // modifications anyway.

  auto init = synthesize_fixed_covers(env, covers, inits, bound, measure);

  // Find the disjunct with the worst similarity to the network.
  std::vector<double> scores;
  for (const Controller& ctrl : init) {
    scores.push_back(measure_piece(ctrl, measure));
  }

  for (int i = 0; i < MAX_SPLITS; i++) {
    std::cout << "Split: " << (i + 1) << " / " << MAX_SPLITS << std::endl;
    // Pick the disjunct with the lowest score
    int to_split = 0;
    int lowest_score = scores[0];
    for (int j = 1; j < scores.size(); j++) {
      if (scores[j] < lowest_score) {
        lowest_score = scores[j];
        to_split = j;
      }
    }

    // Split: try each dimension and split at a set of random points. Then
    // choose whatever split gets the best score.
    double best_score = -std::numeric_limits<double>::max();
    std::vector<Controller> best_controller;
    std::vector<Space> best_covers;
    std::vector<Eigen::MatrixXd> best_inits;
    std::vector<double> best_scores;
    //std::cout << to_split << ": " << covers[to_split].bb_lower.size() << std::endl;
    for (int d = 0; d < covers[to_split].bb_lower.size(); d++) {
      std::cout << "Dimension: " << (d + 1) << " / " << covers[to_split].bb_lower.size() << std::endl;
      // Sample from a truncated uniform distribution. We'll center the
      // distribution at the middle of the bounding box and put two
      // standard deviations at the boundaries of the bounding box.
      double a = covers[to_split].bb_lower(d);
      double b = covers[to_split].bb_upper(d);
      if (std::abs(a - b) <= 0.00000001) {
        // There's no need to consider splits when a = b
        continue;
      }
      double mu = (a + b) / 2.0;
      double sig = (b - a) / 4.0;
      std::default_random_engine generator;
      std::normal_distribution<double> distribution(mu, sig);
      // Try 5 random samples
      for (int j = 0; j < 5; j++) {
        std::cout << "Sample: " << (j + 1) << " / 5" << std::endl;
        // We'll just throw out samples outside our range. About 95% of samples
        // will be within the range so this shouldn't be a performance issue.
        double x = distribution(generator);
        while (x < a || x > b) {
          x = distribution(generator);
        }

        // Split covers
        std::vector<Space> new_covers = covers;
        std::vector<Eigen::MatrixXd> new_inits = inits;
        const Space& s = covers[to_split];
        auto split_space = split_cover(s, d, x);
        new_covers[to_split] = split_space.first;
        new_covers.push_back(split_space.second);
        new_inits.push_back(inits[to_split]);
        auto new_controller = synthesize_fixed_covers(env, new_covers,
            new_inits, bound, measure);
        //std::cout << "New shield size: " << new_controller.size() << std::endl;
        double score = measure_shield(new_controller, measure);
        //std::cout << "score: " << score << " -- best score: " << best_score << std::endl;
        if (score > best_score) {
          //std::cout << "new best x: " << x << std::endl;
          best_score = score;
          best_covers = new_covers;
          best_inits = new_inits;
          best_controller = new_controller;
          best_scores = scores;
          best_scores[to_split] = measure_piece(best_controller[to_split], measure);
          best_scores.push_back(measure_piece(best_controller.back(), measure));
        }
      }
    }
    init = best_controller;
    //std::cout << "Shield size (update " << i << "): " << init.size() << std::endl;
    inits = best_inits;
    scores = best_scores;
    covers = best_covers;
    //std::cout << "Iteration " << i << ": " << scores.size() << std::endl;
    //std::cout << "Best controller:" << std::endl;
    //for (const Controller& c : init) {
    //  std::cout << "Matrix:" << std::endl;
    //  std::cout << c.k << std::endl;
    //  std::cout << "Inv matrix" << std::endl;
    //  std::cout << c.invariant.weights << std::endl;
    //  std::cout << "Inv bias" << std::endl;
    //  std::cout << c.invariant.biases.transpose() << std::endl;
    //}
  }
  //std::cout << "Shield size (end of synthesize_shield): " << init.size() << std::endl;
  return init;
}

LinCons get_cover(const Environment& env, const Eigen::MatrixXd k,
    const Space s, int bound) {
  auto state = std::make_unique<AbstractVal>(ABSTRACT_DOMAIN,
      s.space);
  if (bound > 0) {
    for (int i = 0; i < bound; i++) {
      auto next = env.semi_abstract_step(*state, k);
      //std::cout << "Iteration: " << i << std::endl;
      //next->print(stdout);
      state = state->join(*next);
      //std::cout << "Joined:" << std::endl;
      //state->print(stdout);
    }
  } else {
    while (true) {
      auto next = env.semi_abstract_step(*state, k);
      auto old_state = state->clone();
      state = state->widen(*state->join(*next));
      if (*old_state == *state) {
        break;
      }
    }
  }
  //state->print(stdout);
  return state->get_lincons();
}

static PyObject* py_synthesize_shield(PyObject* self, PyObject* args) {
  PyObject* env_tuple;
  PyObject* covers;
  PyObject* old_shield;
  int bound;
  PyObject* measure;
  if (!PyArg_ParseTuple(args, "OOOiO", &env_tuple, &covers, &old_shield,
        &bound, &measure)) {
    return NULL;
  }
  std::unique_ptr<Environment> env;
  if (PyTuple_Size(env_tuple) == 4) {
    PyObject* env_capsule;
    PyObject* cont_obj;
    double dt;
    PyObject* unsafe;
    if (!PyArg_ParseTuple(env_tuple, "OOdO", &env_capsule, &cont_obj,
          &dt, &unsafe)) {
      return NULL;
    }
    if (!PyBool_Check(cont_obj)) {
      PyErr_SetString(PyExc_RuntimeError, "Malformed environment");
      return NULL;
    }
    PythonCapsule* update = (PythonCapsule*)
        PyCapsule_GetPointer(env_capsule, "synthesis.env_capsule");
    bool continuous = (cont_obj == Py_True);
    std::vector<LinCons> uns = pylist_to_lincons(unsafe);
    env = std::make_unique<NonlinearEnv>(update->concrete, update->update,
        continuous, dt, uns);
  } else {
    PyObject* a_list;
    PyObject* b_list;
    PyObject* cont_obj;
    double dt;
    PyObject* unsafe;
    if (!PyArg_ParseTuple(env_tuple, "OOOdO", &a_list, &b_list, &cont_obj,
          &dt, &unsafe)) {
      return NULL;
    }

    if (!PyBool_Check(cont_obj)) {
      PyErr_SetString(PyExc_RuntimeError, "Malformed environment");
      return NULL;
    }
    Eigen::MatrixXd a = pylist_to_matrix(a_list);
    Eigen::MatrixXd b = pylist_to_matrix(b_list);
    bool continuous = (cont_obj == Py_True);
    std::vector<LinCons> uns = pylist_to_lincons(unsafe);
    env = std::make_unique<LinearEnv>(a, b, continuous, dt, uns);
  }

  std::vector<Eigen::MatrixXd> inits = pylist_to_matrix_list(old_shield);

  auto controller = synthesize_shield(*env, pylist_to_space(covers),
      inits, bound, measure);

  //std::cout << "Shield size (before error): " << controller.size() << std::endl;

  if (controller.empty()) {
    PyErr_SetString(PyExc_RuntimeError, "Unable to synthesize shield");
    return NULL;
  }

  return controller_to_pylist(controller);
}

static PyObject* py_get_covers(PyObject* self, PyObject* args) {
  PyObject* shield;
  PyObject* cover_list;
  PyObject* env_tuple;
  int bound;
  if (!PyArg_ParseTuple(args, "OOOi", &env_tuple, &shield,
        &cover_list, &bound)) {
    return NULL;
  }
  std::unique_ptr<Environment> env;
  if (PyTuple_Size(env_tuple) == 4) {
    PyObject* env_capsule;
    PyObject* cont_obj;
    double dt;
    PyObject* unsafe;
    if (!PyArg_ParseTuple(env_tuple, "OOdO", &env_capsule, &cont_obj,
          &dt, &unsafe)) {
      return NULL;
    }
    if (!PyBool_Check(cont_obj)) {
      PyErr_SetString(PyExc_RuntimeError, "Malformed environment");
      return NULL;
    }
    PythonCapsule* update = (PythonCapsule*)
        PyCapsule_GetPointer(env_capsule, "synthesis.env_capsule");
    bool continuous = (cont_obj == Py_True);
    std::vector<LinCons> uns = pylist_to_lincons(unsafe);
    env = std::make_unique<NonlinearEnv>(update->concrete, update->update,
        continuous, dt, uns);
  } else {
    PyObject* a_list;
    PyObject* b_list;
    PyObject* cont_obj;
    double dt;
    PyObject* unsafe;
    if (!PyArg_ParseTuple(env_tuple, "OOOdO", &a_list, &b_list, &cont_obj,
          &dt, &unsafe)) {
      return NULL;
    }

    if (!PyBool_Check(cont_obj)) {
      PyErr_SetString(PyExc_RuntimeError, "Malformed environment");
      return NULL;
    }
    Eigen::MatrixXd a = pylist_to_matrix(a_list);
    Eigen::MatrixXd b = pylist_to_matrix(b_list);
    bool continuous = (cont_obj == Py_True);
    std::vector<LinCons> uns = pylist_to_lincons(unsafe);
    env = std::make_unique<LinearEnv>(a, b, continuous, dt, uns);
  }
  std::vector<Eigen::MatrixXd> inits = pylist_to_matrix_list(shield);
  std::vector<Space> covers = pylist_to_space(cover_list);
  PyObject* ret = PyList_New(inits.size());
  for (int i = 0; i < inits.size(); i++) {
    LinCons lc = get_cover(*env, inits[i], covers[i], bound);
    PyObject* t = Py_BuildValue("NN", matrix_to_pylist(lc.weights),
        vector_to_pylist(lc.biases));
    PyList_SetItem(ret, i, t);
  }
  return ret;
}

void destroy_capsule(PyObject* capsule) {
  PythonCapsule* update = (PythonCapsule*)
      PyCapsule_GetPointer(capsule, "synthesis.env_capsule");
  delete update;
}

static PyObject* py_get_capsule(PyObject* self, PyObject* args) {
  const char* env_name;
  if (!PyArg_ParseTuple(args, "s", &env_name)) {
    return NULL;
  }
  PythonCapsule* cap = new PythonCapsule;
  // The ArithExpr assumes that the action vars have been appended to the state
  // vars.
  if (strcmp(env_name, "biology")) {
    ArithExpr x1(0);
    ArithExpr x2(1);
    ArithExpr x3(2);
    ArithExpr u1(3);
    ArithExpr u2(4);
    cap->update.push_back(ArithExpr(-0.01) * x1 - x2 *
        (x1 + ArithExpr(4.5)) + u1);
    cap->update.push_back(ArithExpr(-0.025) * x2 + ArithExpr(0.000013) * x3);
    cap->update.push_back(ArithExpr(-0.093) * (x3 + ArithExpr(15.0)) +
        ArithExpr(1.0 / 12.0) * u2);
    cap->concrete = [](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
      Eigen::VectorXd r(x.size());
      r(0) = -0.01 * x(0) - x(1) * (x(0) + 4.5) + u(0);
      r(1) = -0.025 * x(1) + 0.000013 * x(2);
      r(2) = -0.093 * (x(2) + 15.0) + (1.0 / 12.0) * u(1);
      return r;
    };
  } else if (strcmp(env_name, "lanekeeping")) {
    ArithExpr x1(0);
    ArithExpr x2(1);
    ArithExpr x3(2);
    ArithExpr x4(3);
    ArithExpr u1(4);
    ArithExpr u2(5);
    cap->update.push_back(x2 + ArithExpr(27.7) * x3);
    cap->update.push_back(ArithExpr(-1.0 * (133000.0 + 98800.0) /
          (1650.0 * 27.7)) * x2 + ArithExpr((1.59 * 98800.0 - 1.11 * 133000.0)
          / (1650.0 * 27.7) - 27.7) * x4 + ArithExpr(13300.0 / 1650.0) * u1);
    cap->update.push_back(x4 + ArithExpr(-0.035, 0.035));
    cap->update.push_back(ArithExpr((1.59 * 98800.0 - 1.11 * 133000.0) /
        (2315.3 * 27.7)) * x2 + ArithExpr(-1.0 * (1.11 * 1.11 * 133000 +
        1.59 * 1.59 * 98800.0) / (2315.3 * 27.7)) * x4 +
        ArithExpr(1.11 * 133000.0 / 2315.3) * u2);
    cap->concrete = [](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
      Eigen::VectorXd r(x.size());
      r(0) = x(1) + 27.7 * x(2);
      r(1) = -1.0 * (133000.0 + 98800.0) / (1650.0 * 27.7) * x(1) +
          ((1.59 * 98800.0 - 1.11 * 133000.0) / (1650.0 * 27.7) - 27.7) *
          x(3) + (133000.0 / 1650.0) * u(0);
      r(2) = x(3);
      r(3) = ((1.59 * 98800.0 - 1.11 * 133000.0) / (2315.3 * 27.7)) * x(1) +
          (-1.0 * (1.11 * 1.11 * 133000.0 + 1.59 * 1.59 * 98800) /
          (2315.3 + 27.7)) * x(3) + (1.11 * 133000.0 / 2315.3) * u(1);
      return r;
    };
  } else if (strcmp(env_name, "selfdriving")) {
    ArithExpr x1(0);
    ArithExpr x2(1);
    ArithExpr u1(3);
    cap->update.push_back(ArithExpr(-2.0) * (x2 - (x2^3) / ArithExpr(6.0)));
    cap->update.push_back(u1);
    cap->concrete = [](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
      Eigen::VectorXd r(x.size());
      r(0) = -2.0 * (x(1) - std::pow(x(1), 3) / 6.0);
      r(1) = u(0);
      return r;
    };
  } else {
    char exc_str[100];
    exc_str[0] = '\0';
    strcat(exc_str, "Unknown environment: ");
    strcat(exc_str, env_name);
    PyErr_SetString(PyExc_RuntimeError, exc_str);
  }
  PyObject* ret = PyCapsule_New(cap, "synthesis.env_capsule",
      &destroy_capsule);
  return ret;
}

static PyMethodDef SynthesisMethods[] = {
  {"synthesize_shield", py_synthesize_shield, METH_VARARGS,
   "Synthesize a shield for a given environment."},
  {"get_covers", py_get_covers, METH_VARARGS,
   "Get the regions in which a shield should be applied."},
  {"get_env_capsule", py_get_capsule, METH_VARARGS,
   "Get the abstract transformer for an environment by name."}
};

PyMODINIT_FUNC initsynthesis(void) {
  PyObject* m;
  m = Py_InitModule("synthesis", SynthesisMethods);
  if (m == NULL) {
    return;
  }
  char err_name[100];
  strcpy(err_name, "synthesis.domain_error");
  DomainError = PyErr_NewException(err_name, NULL, NULL);
  Py_INCREF(DomainError);
  PyModule_AddObject(m, "domain_error", DomainError);
}

int main() {
  //Eigen::MatrixXd A(2, 2);
  //A << 2, -1,
  //     1,  0;
  //Eigen::MatrixXd B(2, 1);
  //B << 2,
  //     0;
  //Eigen::MatrixXd w1(1, 2);
  //w1 << -1, 0;
  //Eigen::MatrixXd w2(1, 2);
  //w2 << 1, 0;
  //Eigen::MatrixXd w3(1, 2);
  //w3 << 0, -1;
  //Eigen::MatrixXd w4(1, 2);
  //w4 << 0, 1;
  //Eigen::VectorXd b1(1);
  //b1 << -1.5;
  //Eigen::VectorXd b2(1);
  //b2 << -1.5;
  //Eigen::VectorXd b3(1);
  //b3 << -1.5;
  //Eigen::VectorXd b4(1);
  //b4 << -1.5;
  //std::vector<LinCons> unsafe;
  //unsafe.push_back(LinCons(w1, b1));
  //unsafe.push_back(LinCons(w2, b2));
  //unsafe.push_back(LinCons(w3, b3));
  //unsafe.push_back(LinCons(w4, b4));
  //LinearEnv env(A, B, false, 0.01, unsafe);

  //std::vector<Space> covers;
  //Eigen::MatrixXd ws(4, 2);
  //ws <<  0,  1,
  //       0, -1,
  //       1,  0,
  //      -1,  0;
  //Eigen::VectorXd bs(4);
  //bs << 1, 1, 1, 1;
  //Eigen::VectorXd lower(2);
  //lower << -1, -1;
  //Eigen::VectorXd upper(2);
  //upper << 1, 1;
  //covers.push_back(Space { .space = LinCons(ws, bs), .bb_lower = lower,
  //    .bb_upper = upper });

  //Eigen::MatrixXd K(1, 2);
  //K << -1, 0.5;
  //std::vector<Eigen::MatrixXd> inits;
  //inits.push_back(K);

  Eigen::MatrixXd A(15, 15);
  A << 0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,1, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,1, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,1, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,1, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,1, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,1, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,1,
       0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0;

  Eigen::MatrixXd B(15, 8);
  B << 1,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       1, -1,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  1, -1,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  1, -1,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  1, -1,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  1, -1,  0,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  1, -1,  0,
       0,  0,  0,  0,  0,  0,  0,  0,
       0,  0,  0,  0,  0,  0,  1, -1;

  std::vector<double> x_min = {18, 0.1, -1, 0.5, -1, 0.5, -1, 0.5, -1,
    0.5, -1, 0.5, -1, 0.5, -1};
  std::vector<double> x_max = {22, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1,
    1.5, 1, 1.5, 1, 1.5, 1};
  std::vector<double> target = {20, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<LinCons> unsafe;
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(30, 15);
  Eigen::VectorXd b(30);
  Eigen::VectorXd l(15);
  Eigen::VectorXd u(15);
  for (int i = 0; i < 15; i++) {
    Eigen::MatrixXd wp = Eigen::MatrixXd::Zero(1, 15);
    Eigen::MatrixXd wn = Eigen::MatrixXd::Zero(1, 15);
    wp(i) = 1.0;
    wn(i) = -1.0;
    Eigen::VectorXd bp(1);
    Eigen::VectorXd bn(1);
    bp(0) = x_max[i] - target[i];
    bn(0) = x_min[i] - target[i];
    unsafe.push_back(LinCons(wn, bn));
    unsafe.push_back(LinCons(wp, bp));

    w(2 * i, i) = 1.0;
    w(2 * i + 1, i) = -1.0;
    b(2 * i) = x_max[i] - target[i];
    b(2 * i + 1) = x_min[i] - target[i];
    l(i) = x_min[i] - target[i];
    u(i) = x_max[i] - target[i];
  }

  LinearEnv env(A, B, true, 0.01, unsafe);

  std::vector<Space> covers;
  covers.push_back(Space { .space = LinCons(w, b), .bb_lower = l,
      .bb_upper = u });

  Eigen::MatrixXd k(8, 15);
  k << -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
       -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
       -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
       -1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;
  std::vector<Eigen::MatrixXd> inits;
  inits.push_back(k);

  auto res = synthesize_shield(env, covers, inits, 10, NULL);

  for (const Controller& c : res) {
    std::cout << "Matrix:" << std::endl;
    std::cout << c.k << std::endl;
    std::cout << "Invariant matrix:" << std::endl;
    std::cout << c.invariant.weights << std::endl;
    std::cout << "Invariant vector:" << std::endl;
    std::cout << c.invariant.biases.transpose() << std::endl;
  }
}
