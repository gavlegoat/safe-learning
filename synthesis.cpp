#include <optional>

#include <Python.h>

#include "abstract.hpp"

static PyObject* DomainError;

struct Interval {
  Eigen::MatrixXd lower;
  Eigen::MatrixXd upper;
};

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
class Environment {
  public:
    /** An environment transition matrix. */
    Eigen::MatrixXd A;
    /** An environment transition matrix. */
    Eigen::MatrixXd B;
    /** True if the environment uses continuous semantics. */
    bool continuous;
    /** The time step for continuous environments. */
    double dt;
    /** The safe part of the state space. */
    std::vector<LinCons> unsafe_space;

    Environment(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, bool c,
        double d, const std::vector<LinCons>& unsafe):
      A(a), B(b), continuous(c), dt(d), unsafe_space(unsafe) {}

    Eigen::VectorXd step(const Eigen::VectorXd& state,
        const Eigen::MatrixXd& controller) const {
      if (continuous) {
        return state + dt * (A * state + B * controller * state);
      } else {
        return A * state + B * controller * state;
      }
    }

    std::unique_ptr<AbstractVal> semi_abstract_step(const AbstractVal& state,
        const Eigen::MatrixXd& controller) const {
      if (continuous) {
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
        const Interval& controller) const {
      Eigen::MatrixXd w1, w2;
      if (continuous) {
        w1 = Eigen::MatrixXd::Identity(A.rows(), A.rows()) +
          dt * (A + B * controller.lower);
        w2 = Eigen::MatrixXd::Identity(A.rows(), A.rows()) +
          dt * (A + B * controller.upper);
      } else {
        w1 = A + B * controller.lower;
        w2 = A + B * controller.upper;
      }
      Eigen::MatrixXd w_lower = w1.cwiseMin(w2);
      Eigen::MatrixXd w_upper = w1.cwiseMax(w2);
      Eigen::VectorXd bias = Eigen::VectorXd::Zero(A.rows());
      return state.interval_affine(w_lower, w_upper, bias, bias);
    }
};

bool interval_is_safe(const Interval& itv, const Environment& env,
    const Space& cover, const std::vector<LinCons>& other_covers,
    int bound) {
  auto state = std::make_unique<AbstractVal>(AbstractDomain::ZONOTOPE,
      cover.space);
  for (int i = 0; i < bound; i++) {
    state = env.abstract_step(*state, itv);
  }
  for (const LinCons& lc : env.unsafe_space) {
    if (!state->meet_linear_constraint(lc.weights, lc.biases)->is_bottom()) {
      return false;
      // TODO: Deal with covers
    }
  }
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
  auto state = std::make_unique<AbstractVal>(AbstractDomain::ZONOTOPE,
      controller.invariant);
  for (int i = 0; i < bound; i++) {
    state = env.semi_abstract_step(*state, controller.k);
  }
  for (const LinCons& lc : env.unsafe_space) {
    if (!state->meet_linear_constraint(lc.weights, lc.biases)->is_bottom()) {
      return false;
      // TODO: Deal with covers
    }
  }
  return true;
}

/**
 * Measure the safety of a controller in an environment.
 *
 * \param env The environment under control.
 * \param k The controller.
 * \param initial The states the controller should be safe in.
 * \return A measure of the safety of this controller.
 */
double measure_safety(const Environment& env, const Eigen::MatrixXd& k,
    const Space& initial, int bound) {
  int iters = 200;
  double total = 0.0;
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
    // See how safe x is.
    for (int j = 0; j < bound; j++) {
      x = env.A * x + env.B * k * x;
    }
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
  Eigen::MatrixXd k = (itv.lower + itv.upper) / 2;
  Controller contr = {
    .k = k,
    .invariant = cover.space
  };
  double lr = 0.005;
  double v = 0.04;
  for (int i = 0; i < 200; i++) {
    if (!controller_is_safe(env, contr, other_covers, bound)) {
      return k;
    }
    Eigen::MatrixXd delta = Eigen::MatrixXd::Random(k.rows(), k.cols());
    double sim_plus = measure_safety(env, k + v * delta, cover, bound);
    double sim_minus = measure_safety(env, k - v * delta, cover, bound);
    if (sim_plus <= 0.0) {
      return k + v * delta;
    } else if (sim_minus <= 0.0) {
      return k - v * delta;
    }
    Eigen::MatrixXd grad = (sim_plus - sim_minus) / v * delta;
    k += lr * grad;
    k = k.cwiseMax(itv.lower).cwiseMin(itv.upper);
  }
  return {};
}

/**
 * Find a safe region in the parameter space around `k`.
 *
 * Given an environment and a current (safe) controller `k`, find an interval
 * of safe parameters.
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
          double c = (itv.lower(i,j) + itv.upper(i,j)) / 2;
          itv.lower(i,j) = itv.lower(i,j) + (c - itv.lower(i,j)) / 4;
          itv.upper(i,j) = itv.upper(i,j) - (itv.upper(i,j) - c) / 4;
        }
      }
    } else {
      Eigen::MatrixXd bad_k = ce.value();
      for (int i = 0; i < bad_k.rows(); i++) {
        for (int j = 0; j < bad_k.cols(); j++) {
          double c = (itv.lower(i,j) + itv.upper(i,j)) / 2;
          if (itv.upper(i,j) - bad_k(i,j) < bad_k(i,j) - itv.lower(i,j)) {
            itv.upper(i,j) = (c + bad_k(i,j)) / 2;
          } else {
            itv.lower(i,j) = (c + bad_k(i,j)) / 2;
          }
        }
      }
    }
    iters++;
    if (iters > 100) {
      return {};
    }
  }

  std::cout << "Lower:" << std::endl;
  std::cout << itv.lower << std::endl;
  std::cout << "Upper:" << std::endl;
  std::cout << itv.upper << std::endl;
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
    PyObject* c = Py_BuildValue("(O(OO))", k, a, b);
    PyList_SetItem(ret, i, c);
  }
  return ret;
}

double measure_similarity(const Eigen::MatrixXd& mat, PyObject* measure) {
  if (measure == NULL) {
    return 0;
  }
  PyObject* arg = matrix_to_pylist(mat);
  PyObject* res = PyObject_CallObject(measure, arg);
  if (PyErr_Occurred()) {
    PyErr_PrintEx(0);
  }
  return PyFloat_AsDouble(res);
}

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
LinCons compute_invariant(const Environment& env, const Space& cover,
    int bound, const std::vector<LinCons>& other_covers,
    const Eigen::MatrixXd& k) {
  // Find an invariant for x' = (A + B K) x or x' = (I + dt * (A + B K)) x
  // For the bounded case, find the maximum space such that `bound` iterations
  // are safe.
  Eigen::MatrixXd transition;
  if (env.continuous) {
    transition = env.A + env.B * k;
  } else {
    transition = Eigen::MatrixXd::Identity(env.A.rows(), env.A.rows()) +
      env.dt * (env.A + env.B * k);
  }

  Eigen::MatrixXd n_step = transition;
  for (int i = 0; i < bound - 1; i++) {
    n_step *= transition;
  }

  // Find X such that for all x \in X, T * x \in Safe where T is the n-step
  // transition matrix and Safe is the safe region. Then if the _unsafe_ region
  // is defined by A x < b, we need to have
  // A (T x) >= b ==> (A T) x >= b ==> (- A T) x <= b

  std::vector<Eigen::VectorXd> constraints;
  std::vector<double> coeffs;
  for (const LinCons& lc : env.unsafe_space) {
    Eigen::MatrixXd m = -lc.weights * n_step;
    for (int i = 0; i < m.rows(); i++) {
      constraints.push_back(m.row(i));
      coeffs.push_back(lc.biases(i));
    }
  }
  Eigen::MatrixXd ws(constraints.size(), constraints[0].size());
  Eigen::VectorXd bs(constraints.size());
  for (int i = 0; i < ws.rows(); i++) {
    ws.row(i) = constraints[i];
    bs(i) = coeffs[i];
  }

  return LinCons(ws, bs);
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
  Eigen::MatrixXd k = initial;
  double lr = 0.005;
  double v = 0.04;
  int steps_per_projection = 10;
  for (int i = 0; i < 200; i++) {
    std::optional<Interval> safe = compute_safe_space(
        env, cover, other_covers, k, steps_per_projection * lr, bound);
    if (!safe) {
      return {};
    }

    // Gradient steps
    for (int j = 0; j < steps_per_projection; j++) {
      Eigen::MatrixXd delta = Eigen::MatrixXd::Random(k.rows(), k.cols());
      double sim_plus = measure_similarity(k + v * delta, measure);
      double sim_minus = measure_similarity(k - v * delta, measure);
      Eigen::MatrixXd grad = (sim_plus - sim_minus) / v * delta;
      k += lr * grad;
      k = k.cwiseMax(safe.value().lower).cwiseMin(safe.value().upper);
    }
  }
  LinCons inv = compute_invariant(env, cover, bound, other_covers, k);
  return Controller {
    .k = k,
    .invariant = inv
  };
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
    const std::vector<Space>& covers,
    const std::vector<Eigen::MatrixXd>& inits, int bound, PyObject* measure) {
  std::vector<Controller> ret = {};
  std::vector<LinCons> covered;
  for (const Space& s : covers) {
    covered.push_back(s.space);
  }
  for (int i = 0; i < covers.size(); i++) {
    covered.erase(covered.begin() + i);
    auto res = synthesize_linear_controller(env, covers[i], bound, covered,
        inits[i], measure);
    if (!res) {
      return {};
    }
    covered.insert(covered.begin() + i, res.value().invariant);
    ret.push_back(res.value());
  }
  return ret;
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
  PyObject* a_list;
  PyObject* b_list;
  bool continuous;
  double dt;
  PyObject* unsafe;
  if (!PyArg_ParseTuple(env_tuple, "OOpdO", &a_list, &b_list, &continuous,
        &dt, &unsafe)) {
    return NULL;
  }
  Environment env(pylist_to_matrix(a_list), pylist_to_vector(b_list),
      continuous, dt, pylist_to_lincons(unsafe));

  std::vector<Eigen::MatrixXd> inits = pylist_to_matrix_list(old_shield);

  auto controller = synthesize_shield(env, pylist_to_space(covers),
      inits, bound, measure);

  if (controller.empty()) {
    PyErr_SetString(PyExc_RuntimeError, "Unable to synthesize shield");
    return NULL;
  }

  return controller_to_pylist(controller);
}

static PyMethodDef SynthesisMethods[] = {
  {"synthesize_shield", py_synthesize_shield, METH_VARARGS,
   "Synthesize a shield for a given environment."},
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
  Eigen::MatrixXd A(2, 2);
  A << 2, -1,
       1,  0;
  Eigen::MatrixXd B(2, 1);
  B << 2,
       0;
  std::vector<LinCons> unsafe;
  Eigen::MatrixXd w1(1, 2);
  w1 << -1, 0;
  Eigen::VectorXd b1(1);
  b1 << -1.5;
  unsafe.push_back(LinCons(w1, b1));
  Eigen::MatrixXd w2(1, 2);
  w2 << 1, 0;
  Eigen::VectorXd b2(1);
  b2 << -1.5;
  unsafe.push_back(LinCons(w2, b2));
  Eigen::MatrixXd w3(1, 2);
  w3 << 0, -1;
  Eigen::VectorXd b3(1);
  b3 << -1.5;
  unsafe.push_back(LinCons(w3, b3));
  Eigen::MatrixXd w4(1, 2);
  w4 << 0, 1;
  Eigen::VectorXd b4(1);
  b4 << -1.5;
  unsafe.push_back(LinCons(w4, b4));
  Environment env(A, B, false, 0.001, unsafe);

  std::vector<Space> covers;
  Eigen::MatrixXd iw(4, 2);
  iw << 1,  0,
       -1,  0,
        0,  1,
        0, -1;
  Eigen::VectorXd ib(4);
  ib << 1, 1, 1, 1;
  Eigen::VectorXd lo(4);
  lo << -1, -1, -1, -1;
  Eigen::VectorXd hi(4);
  hi << 1, 1, 1, 1;
  covers.push_back(Space {
        .space = LinCons(iw, ib),
        .bb_lower = lo,
        .bb_upper = hi
      });

  std::vector<Eigen::MatrixXd> inits;
  Eigen::MatrixXd init(1, 2);
  init << -1, 0.5;
  inits.push_back(init);

  auto contr = synthesize_shield(env, covers, inits, 100, NULL);

  std::cout << contr.size() << std::endl;
  for (const Controller& c : contr) {
    std::cout << "--------------------" << std::endl;
    std::cout << c.k << std::endl;
  }
}
