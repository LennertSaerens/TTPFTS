#pragma once
#include<cstddef>
#include<iostream>
#include "utils.hpp"
#include "bandits.hpp"
struct policy{
    size_t K;
    size_t dim;
    size_t D;
    double sigma;
    std::vector<size_t> action_space;
    bandit* bandit_ref;
    policy() = default;
    explicit policy( bandit&);
    std::pair<std::pair<size_t, bool>, std::vector<size_t>> loop();
};
struct psi_auer: policy{
    double delta;
    double eps;
    psi_auer()=default;
    explicit psi_auer(bandit&);
    [[nodiscard]] std::pair<std::pair<bool, std::vector<size_t>>, std::vector<size_t>> loop(const size_t&, const double&, const double&, const size_t&);
};

struct psi_ape: policy{
    double delta;
    double eps_1;
    psi_ape()= default;
    explicit psi_ape(bandit&);
    [[nodiscard]] std::pair<std::pair<bool, std::vector<size_t>>, std::vector<size_t>> loop(const size_t&, const double&, const double&, const double&, const size_t&);
    size_t get_ct(const std::vector<std::vector<double>> & means, std::size_t bt, const std::vector<std::vector<double>>& beta) {
        std::vector<double> tmp(K);
        // out les transform reduce inutiles
        std::transform(action_space.begin(), action_space.end(), tmp.begin(),[&](size_t i){
            return minimum_quantity_dom(means[bt], means[i], 0) - beta[bt][i] + INF*(i==bt);
        });
        return get_argmin(tmp, action_space);}
    size_t get_bt(const std::vector<std::vector<double>>& means, const std::vector<size_t>& opt_comp, const std::vector<std::vector<double>>& beta){
        std::vector<double> res;
        res.reserve(opt_comp.size());
        double res_;
        std::transform(opt_comp.begin(), opt_comp.end(), std::back_inserter(res), [&](size_t i){
            res_ = INF;
            for(size_t j:action_space){
                res_ = std::min(res_, minimum_quantity_dom(means[i], means[j], 0.) + beta[j][i] + INF*(i==j));
            }
            return res_;
        });
        return (opt_comp)[std::distance(res.begin(), std::max_element(res.begin(), res.end()))];
        //return res;

    }
};
struct psi_uniform: policy{
    double delta;
    psi_uniform()= default;
    explicit psi_uniform(bandit&);
    [[nodiscard]] std::pair<std::pair<size_t, bool>, std::vector<size_t>> loop(const size_t&, const double&, const double&, const double&, const size_t&);
};