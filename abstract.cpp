#include "abstract.hpp"

#include "t1p.h"
#include "box.h"

#include <ap_disjunction.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <Eigen/Dense>

//static Managers mans{t1p_manager_alloc(), box_manager_alloc()};

ArithExpr::ArithExpr(): expr{nullptr} {}

ArithExpr::ArithExpr(double constant):
  expr{ap_texpr0_cst_scalar_double(constant)} {}

ArithExpr::ArithExpr(int ind): expr{ap_texpr0_dim(ind)} {}

ArithExpr::ArithExpr(double lower, double upper):
  expr{ap_texpr0_cst_interval_double(lower, upper)} {}

ArithExpr::ArithExpr(ap_texpr0_t* e): expr{e} {}

ArithExpr::ArithExpr(const ArithExpr& other):
  expr{ap_texpr0_copy(other.expr)} {}

ArithExpr::ArithExpr(ArithExpr&& other): expr(other.expr) {
  other.expr = nullptr;
}

ArithExpr::~ArithExpr() {
  if (expr != nullptr) {
    ap_texpr0_free(expr);
  }
}

ArithExpr& ArithExpr::operator=(const ArithExpr& other) {
  if (expr != nullptr) {
    ap_texpr0_free(expr);
  }
  expr = ap_texpr0_copy(other.expr);
  return *this;
}

ArithExpr& ArithExpr::operator=(ArithExpr&& other) {
  if (expr != nullptr) {
    ap_texpr0_free(expr);
  }
  expr = other.expr;
  other.expr = nullptr;
  return *this;
}

ArithExpr ArithExpr::negate() const {
  return ArithExpr{ap_texpr0_unop(AP_TEXPR_NEG, ap_texpr0_copy(expr),
        AP_RTYPE_DOUBLE, AP_RDIR_NEAREST)};
}

ArithExpr ArithExpr::operator+(const ArithExpr& other) const {
  return ArithExpr{ap_texpr0_binop(AP_TEXPR_ADD,
      ap_texpr0_copy(expr),
      ap_texpr0_copy(other.expr),
      AP_RTYPE_DOUBLE, AP_RDIR_NEAREST)};
}

ArithExpr ArithExpr::operator-(const ArithExpr& other) const {
  return ArithExpr{ap_texpr0_binop(AP_TEXPR_SUB,
      ap_texpr0_copy(expr),
      ap_texpr0_copy(other.expr),
      AP_RTYPE_DOUBLE, AP_RDIR_NEAREST)};
}

ArithExpr ArithExpr::operator*(const ArithExpr& other) const {
  return ArithExpr{ap_texpr0_binop(AP_TEXPR_MUL,
      ap_texpr0_copy(expr),
      ap_texpr0_copy(other.expr),
      AP_RTYPE_DOUBLE, AP_RDIR_NEAREST)};
}

ArithExpr ArithExpr::operator/(const ArithExpr& other) const {
  return ArithExpr{ap_texpr0_binop(AP_TEXPR_DIV,
      ap_texpr0_copy(expr),
      ap_texpr0_copy(other.expr),
      AP_RTYPE_DOUBLE, AP_RDIR_NEAREST)};
}

ArithExpr ArithExpr::operator^(int power) const {
  // Apron's t1p domain does not support exponents yet, but since we only
  // allow exponents to integer powers we just expand the exponent here.
  ap_texpr0_t* ret = ap_texpr0_copy(this->expr);
  for (int i = 0; i < power - 1; i++) {
    ret = ap_texpr0_binop(AP_TEXPR_MUL,
        ap_texpr0_copy(this->expr),
        ret,
        AP_RTYPE_DOUBLE, AP_RDIR_NEAREST);
  }
  return ArithExpr{ret};
}

LinCons::LinCons(): weights{Eigen::MatrixXd(0, 0)},
  biases{Eigen::VectorXd(0)} {}

LinCons::LinCons(const Eigen::MatrixXd& ws, const Eigen::VectorXd& bs):
  weights{ws}, biases{bs} {}

double LinCons::distance_from(const Eigen::VectorXd& x) const {
  // A x <= b -> A x - b <= 0
  // distance = max(A x - b)
  // If distance is positive then it is the l_infty distance between x and
  // the constrained space. Otherwise, distance decreases as we move away
  // from the safe region.
  return (this->weights * x - this->biases).maxCoeff();
}

inline ap_manager_t* get_manager_from_domain(AbstractDomain dom, size_t size) {
  ap_manager_t* base;
  switch (dom) {
    case AbstractDomain::ZONOTOPE:
      base = t1p_manager_alloc();
      break;
    case AbstractDomain::INTERVAL:
      base = box_manager_alloc();
      break;
    default:
      throw std::runtime_error("Unrecognized domain in get_manager");
  }
  if (size <= 1) {
    return base;
  }
  return ap_disjunction_manager_alloc(base, NULL);
}

AbstractVal::AbstractVal(): man{nullptr}, value{nullptr} {}

// Note that ap_manager_copy doesn't actually create a copy of m, it only
// increments a reference count, so this copy is not expensive.
AbstractVal::AbstractVal(ap_manager_t* m, ap_abstract0_t* v):
  man{ap_manager_copy(m)}, value{v} {}

AbstractVal::AbstractVal(AbstractDomain dom,
    const std::vector<Eigen::VectorXd>& a,
    const std::vector<double>& b): domain{dom} {
  //if (dom == AbstractDomain::ZONOTOPE) {
  //  man = mans.get_t1p_manager();
  //} else {
  //  man = mans.get_box_manager();
  //}
  man = get_manager_from_domain(dom, 1);
  ap_lincons0_array_t arr = ap_lincons0_array_make(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    ap_linexpr0_t* expr = ap_linexpr0_alloc(AP_LINEXPR_DENSE, a[i].size());
    for (int j = 0; j < a[i].size(); j++) {
      ap_linexpr0_set_coeff_scalar_double(expr, j, -a[i](j));
    }
    ap_linexpr0_set_cst_scalar_double(expr, b[i]);
    ap_lincons0_t cons = ap_lincons0_make(AP_CONS_SUPEQ, expr, NULL);
    arr.p[i] = cons;
  }
  value = ap_abstract0_of_lincons_array(man, 0, a[0].size(), &arr);
  ap_lincons0_array_clear(&arr);
}

AbstractVal::AbstractVal(AbstractDomain dom, const LinCons& lc) {
  man = get_manager_from_domain(dom, 1);
  ap_lincons0_array_t arr = ap_lincons0_array_make(lc.biases.size());
  for (int i = 0; i < lc.biases.size(); i++) {
    ap_linexpr0_t* expr = ap_linexpr0_alloc(AP_LINEXPR_DENSE,
        lc.weights.row(i).size());
    for (int j = 0; j < lc.weights.row(i).size(); j++) {
      ap_linexpr0_set_coeff_scalar_double(expr, j, -lc.weights(i, j));
    }
    ap_linexpr0_set_cst_scalar_double(expr, lc.biases(i));
    ap_lincons0_t cons = ap_lincons0_make(AP_CONS_SUPEQ, expr, NULL);
    arr.p[i] = cons;
  }
  value = ap_abstract0_of_lincons_array(man, 0, lc.weights.row(0).size(), &arr);
  ap_lincons0_array_clear(&arr);
}

AbstractVal::AbstractVal(AbstractDomain dom,
    const Eigen::VectorXd& lowers,
    const Eigen::VectorXd& uppers): domain{dom} {
  //if (dom == AbstractDomain::ZONOTOPE) {
  //  man = mans.get_t1p_manager();
  //} else {
  //  man = mans.get_box_manager();
  //}
  man = get_manager_from_domain(dom, 1);
  ap_interval_t** itv =
    (ap_interval_t**) malloc(lowers.size() * sizeof(ap_interval_t*));
  for (int i = 0; i < lowers.size(); i++) {
    itv[i] = ap_interval_alloc();
    ap_interval_set_double(itv[i], lowers(i), uppers(i));
  }
  value = ap_abstract0_of_box(man, 0, lowers.size(), itv);
  ap_interval_array_free(itv, lowers.size());
}

AbstractVal::AbstractVal(const AbstractVal& other) {
  man = ap_manager_copy(other.man);
  value = ap_abstract0_copy(man, other.value);
}

AbstractVal::AbstractVal(AbstractVal&& other) {
  man = other.man;
  value = other.value;
  other.man = nullptr;
  other.value = nullptr;
}

AbstractVal::~AbstractVal() {
  if (value != nullptr) {
    ap_abstract0_free(man, value);
  }
  if (man != nullptr) {
    // Internally, apron managers are reference counted. If this manager is
    // still used elsewhere (assuming it was copied with ap_manager_copy) the
    // reference count will be decremented but the manager will not be freed.
    ap_manager_free(man);
  }
}

std::unique_ptr<AbstractVal> AbstractVal::add_trailing_dimensions(
    int n) const {
  ap_dimchange_t* dimchange = ap_dimchange_alloc(0, n);
  int d = dims();
  for (int i = 0; i < n; i++) {
    dimchange->dim[i] = d;
  }
  ap_abstract0_t* res = ap_abstract0_add_dimensions(
      man, false, value, dimchange, false);
  ap_dimchange_free(dimchange);
  return this->make_new(res);
}

std::unique_ptr<AbstractVal> AbstractVal::add_leading_dimensions(
    int n) const {
  ap_dimchange_t* dimchange = ap_dimchange_alloc(0, n);
  for (int i = 0; i < n; i++) {
    dimchange->dim[i] = 0;
  }
  ap_abstract0_t* res = ap_abstract0_add_dimensions(
      man, false, value, dimchange, false);
  ap_dimchange_free(dimchange);
  return this->make_new(res);
}

std::unique_ptr<AbstractVal> AbstractVal::remove_trailing_dimensions(
    int n) const {
  // NOTE: t1p_remove_dimensions is buggy, but seems to work for removing
  // one dimension. Therefore, we remove one dimension at a time until n
  // dimensions have been removed.
  ap_abstract0_t* res = ap_abstract0_copy(man, value);
  ap_dimchange_t* dimchange = ap_dimchange_alloc(0, 1);
  for (int i = 0; i < n; i++) {
    int d = ap_abstract0_dimension(man, res).realdim;
    dimchange->dim[0] = d - 1;

    //std::cout << "remove_trailing_dimensions before" << std::endl;
    //this->print(stdout);
    res = ap_abstract0_remove_dimensions(man, true, res, dimchange);
    //std::cout << "remove_trailing_dimensions after" << std::endl;
    //ap_abstract0_fprint(stdout, man, res, NULL);
  }
  ap_dimchange_free(dimchange);
  return this->make_new(res);
}

std::unique_ptr<AbstractVal> AbstractVal::meet_linear_constraint(
    const Eigen::MatrixXd& a,
    const Eigen::VectorXd& b) const {
  int size = b.size();
  ap_lincons0_array_t arr = ap_lincons0_array_make(size);

  // a1 x1 + a2 x2 + ... + an xn <= b ==>
  // -a1 x1 - a2 x2 - ... - an xn + b >= 0
  for (int i = 0; i < size; i++) {
    ap_linexpr0_t* expr = ap_linexpr0_alloc(AP_LINEXPR_DENSE, a.cols());
    for (int j = 0; j < a.cols(); j++) {
      ap_linexpr0_set_coeff_scalar_double(expr, j, -a(i,j));
    }
    ap_linexpr0_set_cst_scalar_double(expr, b(i));
    arr.p[i] = ap_lincons0_make(AP_CONS_SUPEQ, expr, NULL);
  }

  ap_abstract0_t* v = ap_abstract0_meet_lincons_array(man, false, value, &arr);
  ap_lincons0_array_clear(&arr);

  return this->make_new(v);
}

std::unique_ptr<AbstractVal> AbstractVal::scalar_affine(
    const Eigen::MatrixXd& w,
    const Eigen::VectorXd& b) const {
  int in_size = w.cols();
  int out_size = w.rows();

  std::unique_ptr<AbstractVal> v{};
  if (out_size > in_size) {
    v = this->add_trailing_dimensions(out_size - in_size);
  } else {
    v = this->clone();
  }

  ap_dim_t* dims = (ap_dim_t*) malloc(out_size * sizeof(ap_dim_t));
  ap_linexpr0_t** update = (ap_linexpr0_t**) malloc(out_size *
      sizeof(ap_linexpr0_t*));
  for (int j = 0; j < out_size; j++) {
    dims[j] = j;
    update[j] = ap_linexpr0_alloc(AP_LINEXPR_DENSE, in_size);
    for (int k = 0; k < in_size; k++) {
      ap_linexpr0_set_coeff_scalar_double(update[j], k, w(j,k));
    }
    ap_linexpr0_set_cst_scalar_double(update[j], b(j));
  }

  ap_abstract0_t* res = ap_abstract0_assign_linexpr_array(
      man, false, v->get_value(), dims, update, out_size, NULL);

  free(dims);
  for (int j = 0; j < out_size; j++) {
    ap_linexpr0_free(update[j]);
  }
  free(update);

  std::unique_ptr<AbstractVal> ret = this->make_new(res);
  if (in_size > out_size) {
    ret = ret->remove_trailing_dimensions(in_size - out_size);
  }
  return ret;
}

std::unique_ptr<AbstractVal> AbstractVal::interval_affine(
    const Eigen::MatrixXd& wl,
    const Eigen::MatrixXd& wu,
    const Eigen::VectorXd& bl,
    const Eigen::VectorXd& bu) const {
  int in_size = wl.cols();
  int out_size = wl.rows();

  std::unique_ptr<AbstractVal> v{};
  if (out_size > in_size) {
    v = this->add_trailing_dimensions(out_size - in_size);
  } else {
    v = this->clone();
  }

  ap_dim_t* dims = (ap_dim_t*) malloc(out_size * sizeof(ap_dim_t));
  ap_linexpr0_t** update = (ap_linexpr0_t**) malloc(out_size *
      sizeof(ap_linexpr0_t*));
  for (int j = 0; j < out_size; j++) {
    dims[j] = j;
    update[j] = ap_linexpr0_alloc(AP_LINEXPR_DENSE, in_size);
    for (int k = 0; k < in_size; k++) {
      ap_linexpr0_set_coeff_interval_double(update[j], k, wl(j,k), wu(j, k));
    }
    ap_linexpr0_set_cst_interval_double(update[j], bl(j), bu(j));
  }

  ap_abstract0_t* res = ap_abstract0_assign_linexpr_array(
      man, false, v->get_value(), dims, update, out_size, NULL);

  free(dims);
  for (int j = 0; j < out_size; j++) {
    ap_linexpr0_free(update[j]);
  }
  free(update);

  std::unique_ptr<AbstractVal> ret = this->make_new(res);
  if (in_size > out_size) {
    ret = ret->remove_trailing_dimensions(in_size - out_size);
  }
  return ret;

  /*
  // Create an abstract value for the coefficients.
  ap_interval_t** arr = (ap_interval_t**) malloc((in_size + 1) * out_size *
      sizeof(ap_interval_t*));
  for (int i = 0; i < in_size * out_size; i++) {
    arr[i] = ap_interval_alloc();
    int r = i / in_size;
    int c = i % in_size;
    ap_interval_set_double(arr[i], wl(r, c), wu(r, c));
  }
  for (int i = 0; i < out_size; i++) {
    int ind = i + in_size * out_size;
    arr[ind] = ap_interval_alloc();
    ap_interval_set_double(arr[ind], bl(i), bu(i));
  }
  std::unique_ptr<AbstractVal> coeffs = this->make_new(
      ap_abstract0_of_box(
          this->man, 0, (in_size + 1) * out_size, arr));

  std::unique_ptr<AbstractVal> input = this->append(*coeffs);
  // NOTE: the size of input is in_size * out_size (for the coefficients
  // + out_size (for the biases) + in_size (for the input)

  // Construct an ArithExpr for this assignment
  std::vector<ArithExpr> exprs;
  for (int i = 0; i < out_size; i++) {
    // Start with the bias term
    ArithExpr row(in_size * (out_size + 1) + i);
    for (int j = 0; j < in_size; j++) {
      // r = r + coeff(i, j) * x(j)
      row = row + ArithExpr((i + 1) * in_size + j) * ArithExpr(j);
    }
    exprs.push_back(row);
  }

  // Perform computation
  std::unique_ptr<AbstractVal> output = input->arith_computation(exprs);
  return output->remove_trailing_dimensions(in_size * out_size + in_size);
  */
}

std::unique_ptr<AbstractVal> AbstractVal::relu() const {
  std::unique_ptr<AbstractVal> z = this->clone();
  size_t num_dims = this->dims();
  Eigen::VectorXd b{Eigen::VectorXd::Zero(1)};
  Eigen::MatrixXd relu_w = Eigen::MatrixXd::Identity(num_dims, num_dims);
  Eigen::VectorXd relu_b = Eigen::VectorXd::Zero(num_dims);
  for (size_t i = 0; i < num_dims; i++) {
    Eigen::MatrixXd alt = Eigen::MatrixXd::Zero(1, num_dims);
    Eigen::MatrixXd agt = alt;
    alt(0,i) = 1.0;
    agt(0,i) = -1.0;

    std::unique_ptr<AbstractVal> zlt = z->meet_linear_constraint(alt, b);
    z = z->meet_linear_constraint(agt, b);

    relu_w(i, i) = 0.0;
    zlt = zlt->scalar_affine(relu_w, relu_b);
    relu_w(i, i) = 1.0;

    z = z->join(*zlt);
  }

  return z;
}

// NOTE: by convention, the this and other should point to the same manager.
// The manager of this is used, so if it is not compatible with the manager of
// other I'm not sure what happens.
std::unique_ptr<AbstractVal> AbstractVal::join(const AbstractVal& other) const {
  ap_abstract0_t* res = ap_abstract0_join(man, false, value, other.get_value());
  return this->make_new(res);
}

std::unique_ptr<AbstractVal> AbstractVal::meet(const AbstractVal& other) const {
  ap_abstract0_t* res = ap_abstract0_meet(man, false, value, other.get_value());
  return this->make_new(res);
}

std::unique_ptr<AbstractVal> AbstractVal::widen(
    const AbstractVal& other) const {
  ap_abstract0_t* res = ap_abstract0_widening(man, value, other.get_value());
  return this->make_new(res);
}

std::unique_ptr<AbstractVal> AbstractVal::append(const AbstractVal& b) const {
  int n1 = b.dims();
  int n2 = this->dims();
  std::unique_ptr<AbstractVal> p1 = this->add_trailing_dimensions(n1);
  std::unique_ptr<AbstractVal> p2 = b.add_leading_dimensions(n2);
  // At this point p1 has n1 extra unconstrained dimensions at the end and
  // p2 has n2 extra unconstrained dimensions at the beginning. Meeting
  // these two gives the desired result.

  // Apron seems to have a problem doing this meet when the zonotopes have some
  // unconstrained dimensions.
  // return p1->meet(*p2);
  //
  // Our second attempt was to convert both abstract values to arrays of linear
  // constraints, append the two arrays, then create a new value which
  // satisfies all of the resulting constraints. This runs into a very strange
  // use-after-free bug which I haven't been able to track down
  //
  // The current strategy is to convert only b to an array of linear
  // constraints and to meet this with the resulting array.

  ap_lincons0_array_t lc2 = ap_abstract0_to_lincons_array(
      p2->get_manager(), p2->get_value());

  ap_abstract0_t* v = ap_abstract0_meet_lincons_array(
      man, false, p1->get_value(), &lc2);

  ap_lincons0_array_clear(&lc2);

  return this->make_new(v);
}

std::unique_ptr<AbstractVal> AbstractVal::arith_computation(
    const std::vector<ArithExpr>& exprs) const {
  int in_size = this->dims();
  int out_size = exprs.size();
  std::unique_ptr<AbstractVal> inp;
  if (out_size > in_size) {
    inp = this->add_trailing_dimensions(out_size - in_size);
  } else {
    inp = this->clone();
  }
  ap_texpr0_t** arr =
    (ap_texpr0_t**) malloc(exprs.size() * sizeof(ap_texpr0_t*));
  ap_dim_t* dims = (ap_dim_t*) malloc(exprs.size() * sizeof(dims));
  for (size_t i = 0; i < exprs.size(); i++) {
    arr[i] = ap_texpr0_copy(exprs[i].get_texpr());
    dims[i] = i;
  }
  ap_abstract0_t* v = ap_abstract0_assign_texpr_array(
      man, false, inp->get_value(), dims, arr, exprs.size(), NULL);
  free(dims);
  for (size_t i = 0; i < exprs.size(); i++) {
    ap_texpr0_free(arr[i]);
  }
  free(arr);
  std::unique_ptr<AbstractVal> ret = this->make_new(v);
  if (in_size > out_size) {
    ret = ret->remove_trailing_dimensions(in_size - out_size);
  }
  return ret;
}

bool AbstractVal::contains_point(const Eigen::VectorXd& x) const {
  ap_interval_t** itv = (ap_interval_t**) malloc(
      x.size() * sizeof(ap_interval_t*));
  for (int i = 0; i < x.size(); i++) {
    itv[i] = ap_interval_alloc();
    ap_interval_set_double(itv[i], x(i), x(i));
  }
  ap_abstract0_t* point = ap_abstract0_of_box(man, 0, x.size(), itv);
  ap_interval_array_free(itv, x.size());
  ap_abstract0_t* meet = ap_abstract0_meet(man, false, value, point);
  ap_abstract0_free(man, point);
  bool bottom = ap_abstract0_is_bottom(man, meet);
  ap_abstract0_free(man, meet);
  return !bottom;
}

bool AbstractVal::contains(const AbstractVal& x) const {
  // Convert this to an array of linear constraints
  ap_lincons0_array_t arr = ap_abstract0_to_lincons_array(man, value);

  // Negate each constraint
  for (size_t i = 0; i < arr.size; i++) {
    ap_linexpr0_t* expr = arr.p[i].linexpr0;
    for (size_t j = 0; j < x.dims(); j++) {
      ap_coeff_t* c = ap_linexpr0_coeffref(expr, j);
      ap_coeff_neg(c, c);
    }
    ap_coeff_t* c = ap_linexpr0_cstref(expr);
    ap_coeff_neg(c, c);
  }

  // Meet the new constraints with x
  ap_abstract0_t* v = ap_abstract0_meet_lincons_array(man, false, value, &arr);
  ap_lincons0_array_clear(&arr);

  // Since the new constraints contain everything NOT in this, if the meet is
  // not empty, then x is not entirely contained within this.
  bool bottom = ap_abstract0_is_bottom(man, v);
  ap_abstract0_free(man, v);
  return bottom;
}

Eigen::VectorXd AbstractVal::get_center() const {
  ap_interval_t** bbox = ap_abstract0_to_box(man, value);
  Eigen::VectorXd center(dims());
  for (size_t i = 0; i < dims(); i++) {
    double l, u;
    ap_double_set_scalar(&l, bbox[i]->inf, MPFR_RNDN);
    ap_double_set_scalar(&u, bbox[i]->sup, MPFR_RNDN);
    center(i) = (l + u) / 2.0;
  }
  return center;
}

Eigen::VectorXd AbstractVal::get_contained_point() const {
  ap_interval_t** bbox = ap_abstract0_to_box(man, value);
  Eigen::VectorXd lower(this->dims());
  Eigen::VectorXd upper(this->dims());
  for (size_t i = 0; i < dims(); i++) {
    double l, u;
    ap_double_set_scalar(&l, bbox[i]->inf, MPFR_RNDN);
    ap_double_set_scalar(&u, bbox[i]->sup, MPFR_RNDN);
    lower(i) = l;
    upper(i) = u;
  }
  while (true) {
    // Choose a random point inside the bounding box of this abstract value,
    // then check to see if it is contained in this value. If it is we return
    // it, otherwise try another point.
    Eigen::VectorXd rand = Eigen::VectorXd::Random(dims());
    for (size_t i = 0; i < this->dims(); i++) {
      rand(i) = (1 + rand(i)) * (upper(i) - lower(i)) / 2.0 + lower(i);
    }
    if (this->contains_point(rand)) {
      return rand;
    }
  }
}

std::unique_ptr<AbstractVal> AbstractVal::clone() const {
  return std::make_unique<AbstractVal>(man, ap_abstract0_copy(man, value));
}

std::unique_ptr<AbstractVal> AbstractVal::make_new(ap_abstract0_t* a) const {
  return std::make_unique<AbstractVal>(man, a);
}

//double distance_to_abstract0(ap_manager_t* man, ap_abstract0_t* a,
//    const Eigen::VectorXd& x) {
//  // Convert the abstract value to an array of linear constraints
//  ap_lincons0_array_t arr = ap_abstract0_to_lincons_array(man, a);
//  // Determine whether each constraint is satisfied and find the constraint
//  // which is closest to x
//  bool satisfies_all = true;
//  for (int i = 0; i < arr.size; i++) {
//    // TODO: I'm not sure what the semantics of arr.p[i].scalar are. It seems
//    // to only be used for EQMOD constraints.
//    ap_linexpr0_t* expr = arr.p[i].linexpr0;
//    Eigen::VectorXd coeffs(x.size());
//    for (int j = 0; j < x.size(); j++) {
//      ap_coeff_t* c = ap_linexpr0_coeffref(expr, j);
//      if (c->discr == AP_COEFF_SCALAR) {
//        // TODO
//      } else {
//        // TODO
//      }
//    }
//  }
//  return 0;
//}
//
//double AbstractVal::distance_to_point(const Eigen::VectorXd& x) const {
//  return distance_to_abstract0(this->man, this->value, x);
//}

typedef struct {
  ap_abstract0_t* abs;
  Eigen::VectorXd center;
} abstract_value;

Eigen::VectorXd compute_center(ap_manager_t* man, ap_abstract0_t* a) {
  ap_interval_t** itv = ap_abstract0_to_box(man, a);
  int d = ap_abstract0_dimension(man, a).realdim;
  Eigen::VectorXd center(d);
  for (int i = 0; i < d; i++) {
    double l, u;
    ap_double_set_scalar(&l, itv[i]->inf, MPFR_RNDN);
    ap_double_set_scalar(&u, itv[i]->sup, MPFR_RNDN);
    center(d) = (l + u) / 2.0;
  }
  ap_interval_array_free(itv, d);
  return center;
}

// Given a manager (this is a manager for the disjunction domain) and a
// disjunctive element, reduce the number of disjuncts to size. This is done
// in place so that after this call, the value pointed to by a is a disjunctive
// value with an appropriate number of disjuncts. Note that this function does
// not perform any checks and assumes a is a disjunction. This function
// typically returns a reference to a, but not always.
ap_abstract0_t* merge_disjuncts(ap_manager_t* man, ap_abstract0_t* a,
    size_t size) {
  ap_disjunction_t* ad = (ap_disjunction_t*) a->value;
  size_t s = ad->size;
  // Get the manager for the underlying abstract domain
  ap_disjunction_internal_t* in = (ap_disjunction_internal_t*) man->internal;
  ap_manager_t* under = in->manager;

  // Compute the center of the bounding box of each disjunct
  std::list<abstract_value> vals{};
  bool is_top = false;
  for (size_t i = 0; i < s; i++) {
    abstract_value v;
    ap_abstract0_t* abs = (ap_abstract0_t*) ad->p[i];
    if (ap_abstract0_is_top(under, abs)) {
      is_top = true;
      break;
    } else if (ap_abstract0_is_bottom(under, abs)) {
      // Don't add bottom elements to vals
      continue;
    }
    v.abs = ap_abstract0_copy(under, (ap_abstract0_t*) ad->p[i]);
    v.center = compute_center(under, v.abs);
    vals.push_back(v);
  }
  if (is_top) {
    int d = ap_abstract0_dimension(man, a).realdim;
    ap_abstract0_free(man, a);
    return ap_abstract0_top(man, 0, d);
  } else if (vals.size() == 0) {
    int d = ap_abstract0_dimension(man, a).realdim;
    ap_abstract0_free(man, a);
    return ap_abstract0_bottom(man, 0, d);
  }

  if (s <= size) {
    return a;
  }

  while (vals.size() > size) {
    // Find the two elements whose centers (computed by bounding box) are
    // closest to each other.
    std::list<abstract_value>::iterator it1;
    std::list<abstract_value>::iterator it2;
    double best_dist = std::numeric_limits<double>::max();
    for (auto i = vals.begin(); i != vals.end(); i++) {
      for (auto j = vals.begin(); j != vals.end(); j++) {
        double dist = (i->center - j->center).norm();
        if (dist < best_dist) {
          best_dist = dist;
          it1 = i;
          it2 = j;
        }
      }
    }
    // Join these two elements.
    ap_abstract0_t* a1 = it1->abs;
    ap_abstract0_t* a2 = it2->abs;
    ap_abstract0_t* join = ap_abstract0_join(under, false, a1, a2);
    abstract_value av;
    av.abs = join;
    av.center = compute_center(under, join);
    // Remove the two elements from vals and add the new one.
    vals.erase(it1);
    vals.erase(it2);
    vals.push_back(av);
  }

  // Finally, we replace the values in a with those in vals.
  for (size_t i = 0; i < s; i++) {
    ap_abstract0_free(under, (ap_abstract0_t*) ad->p[i]);
  }
  free(ad->p);
  ad->size = size;
  ad->p = (void**) malloc(size * sizeof(void*));
  int ind = 0;
  for (auto it = vals.begin(); it != vals.end(); it++) {
    ad->p[ind] = it->abs;
    ind++;
  }
  return a;
}

void AbstractVal::print(FILE* out) const {
  ap_abstract0_fprint(out, man, value, NULL);
}

Powerset::Powerset(ap_manager_t* m, ap_abstract0_t* v, size_t s):
  AbstractVal{m, v}, size{s} {}

Powerset::Powerset(const Powerset& p): AbstractVal{p}, size{p.size} {}

Powerset::Powerset(AbstractDomain dom, size_t s,
    const std::vector<Eigen::VectorXd>& a, const std::vector<double>& b) {
  //ap_manager_t* base_manager;
  //if (dom == AbstractDomain::ZONOTOPE) {
  //  base_manager = mans.get_t1p_manager();
  //} else {
  //  base_manager = mans.get_box_manager();
  //}
  //man = mans.get_disj_manager(base_manager);
  man = get_manager_from_domain(dom, s);
  size = s;
  ap_lincons0_array_t arr = ap_lincons0_array_make(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    ap_linexpr0_t* expr = ap_linexpr0_alloc(AP_LINEXPR_DENSE, a[i].size());
    for (int j = 0; j < a[i].size(); j++) {
      ap_linexpr0_set_coeff_scalar_double(expr, j, -a[i](j));
    }
    ap_linexpr0_set_cst_scalar_double(expr, b[i]);
    ap_lincons0_t cons = ap_lincons0_make(AP_CONS_SUPEQ, expr, NULL);
    arr.p[i] = cons;
  }
  value = ap_abstract0_of_lincons_array(man, 0, a[0].size(), &arr);
  ap_lincons0_array_clear(&arr);
}

Powerset::Powerset(AbstractDomain dom, size_t s, const LinCons& lc) {
  man = get_manager_from_domain(dom, s);
  ap_lincons0_array_t arr = ap_lincons0_array_make(lc.biases.size());
  for (int i = 0; i < lc.biases.size(); i++) {
    ap_linexpr0_t* expr = ap_linexpr0_alloc(AP_LINEXPR_DENSE,
        lc.weights.row(i).size());
    for (int j = 0; j < lc.weights.row(i).size(); j++) {
      ap_linexpr0_set_coeff_scalar_double(expr, j, -lc.weights(i, j));
    }
    ap_linexpr0_set_cst_scalar_double(expr, lc.biases(i));
    ap_lincons0_t cons = ap_lincons0_make(AP_CONS_SUPEQ, expr, NULL);
    arr.p[i] = cons;
  }
  value = ap_abstract0_of_lincons_array(man, 0, lc.weights.row(0).size(), &arr);
  ap_lincons0_array_clear(&arr);
}

Powerset::Powerset(AbstractDomain dom, size_t s,
    const Eigen::VectorXd& lowers, const Eigen::VectorXd& uppers) {
  domain = dom;
  //ap_manager_t* base_manager;
  //if (dom == AbstractDomain::ZONOTOPE) {
  //  base_manager = mans.get_t1p_manager();
  //} else {
  //  base_manager = mans.get_box_manager();
  //}
  //man = mans.get_disj_manager(base_manager);
  man = get_manager_from_domain(dom, s);
  ap_interval_t** itv =
    (ap_interval_t**) malloc(lowers.size() * sizeof(ap_interval_t*));
  for (int i = 0; i < lowers.size(); i++) {
    itv[i] = ap_interval_alloc();
    ap_interval_set_double(itv[i], lowers(i), uppers(i));
  }
  value = ap_abstract0_of_box(man, 0, lowers.size(), itv);
  ap_interval_array_free(itv, lowers.size());
}

Powerset& Powerset::operator=(const Powerset& other) {
  size = other.size;
  man = other.man;
  value = ap_abstract0_copy(man, other.value);
  return *this;
}

std::unique_ptr<AbstractVal> Powerset::join(const AbstractVal& other) const {
  std::unique_ptr<AbstractVal> res = this->AbstractVal::join(other);
  ap_manager_t* m = res->get_manager();
  ap_abstract0_t* a = res->get_value();
  a = merge_disjuncts(m, a, size);
  ap_abstract0_t* ra = (ap_abstract0_t*) malloc(sizeof(ap_abstract0_t));
  ra->value = a;
  ra->man = m;
  return this->make_new(ra);
}

std::unique_ptr<AbstractVal> Powerset::meet(const AbstractVal& other) const {
  std::unique_ptr<AbstractVal> res = this->AbstractVal::meet(other);
  ap_manager_t* m = res->get_manager();
  ap_abstract0_t* a = res->get_value();
  a = merge_disjuncts(m, a, size);
  ap_abstract0_t* ra = (ap_abstract0_t*) malloc(sizeof(ap_abstract0_t));
  ra->value = a;
  ra->man = m;
  return this->make_new(ra);
}

std::unique_ptr<AbstractVal> Powerset::arith_computation(
    const std::vector<ArithExpr>& exprs) const {
  std::unique_ptr<AbstractVal> res =
    this->AbstractVal::arith_computation(exprs);
  ap_manager_t* m = res->get_manager();
  ap_abstract0_t* a = res->get_value();
  a = merge_disjuncts(m, a, size);
  ap_abstract0_t* ra = (ap_abstract0_t*) malloc(sizeof(ap_abstract0_t));
  ra->value = a;
  ra->man = m;
  return this->make_new(ra);
}

Eigen::VectorXd Powerset::get_contained_point() const {
  ap_disjunction_t* ad = (ap_disjunction_t*) value->value;
  size_t s = ad->size;
  // Get the manager for the underlying abstract domain
  ap_disjunction_internal_t* in = (ap_disjunction_internal_t*) man->internal;
  ap_manager_t* under = in->manager;
  if (s == 0) {
    return Eigen::VectorXd(0);
  }
  ap_abstract0_t* disjunct = (ap_abstract0_t*) ad->p[0];
  return compute_center(under, disjunct);
}

std::unique_ptr<AbstractVal> Powerset::clone() const {
  return std::make_unique<Powerset>(man, ap_abstract0_copy(man, value), size);
}

std::unique_ptr<AbstractVal> Powerset::make_new(ap_abstract0_t* a) const {
  return std::make_unique<Powerset>(man, a, size);
}

