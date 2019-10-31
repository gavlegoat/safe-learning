/* Greg Anderson
 *
 * Wrapper classes for Apron abstractions.
 */

#ifndef _ABSTRACT_H_
#define _ABSTRACT_H_

#include <ap_abstract0.h>
#include <ap_disjunction.h>
#include <t1p.h>
#include <box.h>

#include <cstdlib>
#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <optional>
#include <Eigen/Dense>

class ArithExpr {
  private:
    ap_texpr0_t* expr;

  public:
    ArithExpr();
    ArithExpr(double constant);
    ArithExpr(int ind);
    ArithExpr(double lower, double upper);
    ArithExpr(ap_texpr0_t* expr);
    ArithExpr(const ArithExpr& other);
    ArithExpr(ArithExpr&& other);
    ~ArithExpr();
    ArithExpr& operator=(const ArithExpr& other);
    ArithExpr& operator=(ArithExpr&& other);
    ArithExpr negate() const;
    ArithExpr operator+(const ArithExpr& other) const;
    ArithExpr operator-(const ArithExpr& other) const;
    ArithExpr operator*(const ArithExpr& other) const;
    ArithExpr operator/(const ArithExpr& other) const;
    ArithExpr operator^(int power) const;

    inline ap_texpr0_t* get_texpr() const {
      return expr;
    }
};

/**
 * A set of linear constraints. A point x satisfies these constraints if
 * `weights * x <= biases`.
 */
class LinCons {
  public:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    LinCons();
    LinCons(const Eigen::MatrixXd& ws, const Eigen::VectorXd& bs);
    double distance_from(const Eigen::VectorXd& x) const;
};

enum class AbstractDomain { ZONOTOPE, INTERVAL };

/**
 * An abstract value over some underlying Apron value. This is just a
 * convenient wrapper class around Apron values.
 */
class AbstractVal {
  protected:
    ap_manager_t* man;
    ap_abstract0_t* value;
    virtual std::unique_ptr<AbstractVal> make_new(ap_abstract0_t* a) const;
    AbstractDomain domain;

  public:
    AbstractVal();
    AbstractVal(ap_manager_t* man, ap_abstract0_t* v);
    // Construct a new abstract value in the given domain subject to the
    // set of linear constraints a x <= b
    AbstractVal(AbstractDomain dom, const std::vector<Eigen::VectorXd>& a,
        const std::vector<double>& b);
    AbstractVal(AbstractDomain dom, const LinCons& lc);

    // Construct a new abstract value in the given domain from the given
    // interval
    AbstractVal(AbstractDomain dom, const Eigen::VectorXd& lowers,
        const Eigen::VectorXd& uppers);
    AbstractVal(const AbstractVal& other);
    AbstractVal(AbstractVal&& other);
    virtual ~AbstractVal();
    AbstractVal& operator=(const AbstractVal& other) = delete;
    AbstractVal& operator=(AbstractVal&& other) = delete;

    inline ap_manager_t* get_manager() const {
      return man;
    }

    inline ap_abstract0_t* get_value() const {
      return value;
    }

    inline AbstractDomain get_domain() const {
      return domain;
    }

    std::unique_ptr<AbstractVal> add_trailing_dimensions(int n) const;
    std::unique_ptr<AbstractVal> add_leading_dimensions(int n) const;
    std::unique_ptr<AbstractVal> remove_trailing_dimensions(int n) const;

    /**
     * Meet this value with the linear constraints a x <= b.
     */
    virtual std::unique_ptr<AbstractVal> meet_linear_constraint(
        const Eigen::MatrixXd& a,
        const Eigen::VectorXd& b) const;

    /**
     * Perform a specific affine transformation.
     */
    virtual std::unique_ptr<AbstractVal> scalar_affine(
        const Eigen::MatrixXd& w,
        const Eigen::VectorXd& b) const;

    /**
     * Perform an abstract transformation where each coefficient is an interval.
     */
    virtual std::unique_ptr<AbstractVal> interval_affine(
        const Eigen::MatrixXd& wl,
        const Eigen::MatrixXd& wu,
        const Eigen::VectorXd& bl,
        const Eigen::VectorXd& bu) const;

    /**
     * A relu is computed as follows: for each dimension i, compute
     * x_l = meet(x, x_i < 0) and x_u = meet(x, x_i >= 0). Compute
     * x'_l = x_l[x_i <- 0]. Let x = join(x'_l, x_u).
     */
    virtual std::unique_ptr<AbstractVal> relu() const;

    virtual std::unique_ptr<AbstractVal> join(const AbstractVal& other) const;

    virtual std::unique_ptr<AbstractVal> meet(const AbstractVal& other) const;

    virtual std::unique_ptr<AbstractVal> widen(const AbstractVal& other) const;

    /**
     * Create an abstract value by adding each dimension of b to this and
     * maintain the relations among variables in b.
     */
    virtual std::unique_ptr<AbstractVal> append(const AbstractVal& b) const;

    virtual std::unique_ptr<AbstractVal> arith_computation(
        const std::vector<ArithExpr>& exprs) const;

    inline bool is_bottom() const {
      return ap_abstract0_is_bottom(man, value);
    }

    bool contains_point(const Eigen::VectorXd& x) const;
    bool contains(const AbstractVal& x) const;
    Eigen::VectorXd get_center() const;
    virtual Eigen::VectorXd get_contained_point() const;

    /**
     * Get the number of dimensions of this abstract value.
     */
    inline size_t dims() const {
      return ap_abstract0_dimension(man, value).realdim;
    }

    virtual std::unique_ptr<AbstractVal> clone() const;

    //virtual double distance_to_point(const Eigen::VectorXd& x) const;
    void print(FILE* out) const;
};

/**
 * Powerset is used for a bounded powerset domain. It is based on Aprons
 * disjunctive domain, but applies a consolidation step after each join or
 * merge.
 */
class Powerset: public AbstractVal {
  private:
    size_t size;

  protected:
    std::unique_ptr<AbstractVal> make_new(ap_abstract0_t* a) const override;

  public:
    Powerset(ap_manager_t* m, ap_abstract0_t* v, size_t s);
    Powerset(const Powerset& p);
    Powerset(AbstractDomain dom, size_t size,
        const std::vector<Eigen::VectorXd>& a,
        const std::vector<double>& b);
    Powerset(AbstractDomain dom, size_t size, const LinCons& lc);
    Powerset(AbstractDomain dom, size_t size,
        const Eigen::VectorXd& lowers,
        const Eigen::VectorXd& uppers);
    Powerset& operator=(const Powerset& other);
    std::unique_ptr<AbstractVal> join(const AbstractVal& other) const override;
    std::unique_ptr<AbstractVal> meet(const AbstractVal& other) const override;
    std::unique_ptr<AbstractVal> arith_computation(
        const std::vector<ArithExpr>& exprs) const override;
    Eigen::VectorXd get_contained_point() const;
    std::unique_ptr<AbstractVal> clone() const override;
    //double distance_to_point(const Eigen::VectorXd& x) const;
};

//class Managers {
//  private:
//    ap_manager_t* t1p_man;
//    ap_manager_t* box_man;
//    std::map<ap_manager_t*, ap_manager_t*> disj_mans;
//
//  public:
//    Managers(ap_manager_t*, ap_manager_t*);
//    ~Managers();
//    inline ap_manager_t* get_t1p_manager() const {
//      return t1p_man;
//    }
//    inline ap_manager_t* get_box_manager() const {
//      return box_man;
//    }
//    inline ap_manager_t* get_disj_manager(ap_manager_t* base) const {
//      return disj_mans.at(base);
//    }
//};

#endif
