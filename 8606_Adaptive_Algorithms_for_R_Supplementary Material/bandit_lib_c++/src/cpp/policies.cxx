#include <vector>
#include <numeric>
#include "utils.hpp"
#include "bandits.hpp"
#include "policies.hpp"
#include <algorithm>
policy::policy(bandit &bandit_ref) {
    this->bandit_ref = &bandit_ref;
    K = bandit_ref.K;
    D = bandit_ref.D;
    dim = bandit_ref.D;
    sigma = bandit_ref.sigma;
    action_space = bandit_ref.action_space;
}
std::pair<std::pair<size_t, bool>, std::vector<size_t>> policy::loop(void) {
    return {};
}
psi_ape::psi_ape(bandit &bandit_ref): policy(bandit_ref) {
};
std::pair<std::pair<bool, std::vector<size_t>>, std::vector<size_t>>
psi_ape::loop(const size_t& seed, const double& delta, const double& eps_1, const double& eps_2, const size_t& m) {
    this->eps_1 = eps_1;
    this->delta = delta;
    double cg = Cg(this->delta);
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<double> null(K, 0);
    std::vector<size_t> St;
    std::vector<size_t> opt;
    std::vector<bool> opt_mask(K);
    std::vector<bool> St_mask;
    std::vector<size_t> St_comp;
    std::vector<size_t> opt_comp;
    std::vector<size_t> Ts(K, 1);
    std::vector<std::vector<double>> beta (K, std::vector<double>(K, betaij(1, 1, cg, sigma)));
    std::vector<std::vector<double>> means_t(K, std::vector<double>(D));
    double z1_t, z2_t, z4_t;
    size_t z3_t;
// compute the empirical Pareto set St
#define get_St {St_mask  = std::move(pareto_optimal_arms_mask(means_t, null, 0.)); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St), [&St_mask]( size_t i){return St_mask[i];}); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St_comp), [&St_mask]( size_t i){return !St_mask[i];});};
// compute OPT(t)
#define get_opt {std::transform(action_space.begin(), action_space.end(), opt_mask.begin(), [&](size_t i){\
    return get_h(i, means_t, beta, eps_1) > 0;\
});for (size_t  i{0}; i < K; ++i) {\
opt_mask[i]?opt.push_back(i): opt_comp.push_back(i);}}

    size_t at, bt, ct;
// Initial sampling
    for (auto k:action_space){
        means_t[k] = bandit_ref->sample({k})[0];
    }
    get_St
    get_opt
    // check stopping rule
    z1_t = get_z1t(means_t, St, beta,eps_1);
    z2_t = get_z2t(means_t, St_comp, beta, eps_1);
    z4_t = get_z_tilde(means_t,beta,opt,eps_1, eps_2);
    z3_t = std::accumulate(opt_mask.begin(), opt_mask.end(), 0);
    size_t t = K;
    while((z1_t<0 || z2_t <0)&&(z3_t < m) && (z4_t<0)){
        bt = get_bt(means_t, opt_comp, beta);
        ct = get_ct(means_t, bt, beta);
        at = (Ts[bt]>Ts[ct])?ct:bt;
        for (auto k: {at}) {
            std::vector<double> v(std::move(bandit_ref->sample({k})[0])); // to move
            std::transform(means_t[k].begin(), means_t[k].end(), v.begin(), means_t[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
        }
        // update beta
        for (size_t i = 0; i < K; ++i) {
            beta[at][i] = betaij(Ts[i], Ts[at],cg, sigma);
            beta[i][at] = beta[at][i];
        }
        St.clear();
        St_comp.clear();
        opt.clear();
        opt_comp.clear();
        get_St
        get_opt
        z1_t = get_z1t(means_t, St, beta,eps_1);
        z2_t = get_z2t(means_t, St_comp, beta, eps_1);
        z3_t = std::accumulate(opt_mask.begin(), opt_mask.end(), 0);
        z4_t = get_z_tilde(means_t,beta,opt,eps_1, eps_2);
        ++t;
    }
    std::sort(St.begin(), St.end());
    // optimal_arms is always sorted;
    bool is_found = std::equal(St.begin(), St.end(), bandit_ref->optimal_arms.begin(), bandit_ref->optimal_arms.end()); // optimal_set foundp
    return std::pair<std::pair<bool, std::vector<size_t>>, std::vector<size_t>> {std::pair<bool, std::vector<size_t>>{ is_found, Ts}, opt};
}

psi_uniform::psi_uniform(bandit &bandit_ref):policy(bandit_ref) {};

std::pair<std::pair<size_t, bool>, std::vector<size_t>>
psi_uniform::loop(const size_t& seed, const double& delta, const double& eps_1, const double& eps_2, const size_t& m)  {
    this->delta = delta;
    // Initialize the model
    bandit_ref->reset_env(seed);
    double cg = Cg(this->delta);
    std::vector<double> null(K, 0);
    std::vector<size_t> St;
    std::vector<size_t> opt;
    std::vector<bool> opt_mask(K);
    std::vector<bool> St_mask;
    std::vector<size_t> St_comp;
    std::vector<size_t> opt_comp;
    std::vector<size_t> Ts(K, 1);
    std::vector<std::vector<double>> means_t(K, std::vector<double>(D));
    std::vector<std::vector<double>> beta (K, std::vector<double>(K, betaij(1, 1, cg, sigma)));
    std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0,K-1);
    double z1_t, z2_t;
    size_t z3_t;
#define get_St {St_mask  = std::move(pareto_optimal_arms_mask(means_t, null, 0.)); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St), [&St_mask]( size_t i){return St_mask[i];}); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St_comp), [&St_mask]( size_t i){return !St_mask[i];});};
#define get_opt {std::transform(action_space.begin(), action_space.end(), opt_mask.begin(), [&](size_t i){\
    return get_h(i, means_t, beta, eps_1) > 0;\
});for (size_t  i{0}; i < K; ++i) {\
opt_mask[i]?opt.push_back(i): opt_comp.push_back(i);}}

// Initial sampling
    for (auto k:action_space){
        means_t[k] = bandit_ref->sample({k})[0];
    }
    get_St
    get_opt
    // check stopping rule
    z1_t = get_z1t(means_t, St, beta, eps_1);
    z2_t = get_z2t(means_t, St_comp, beta, eps_1);
    z3_t = std::accumulate(opt_mask.begin(), opt_mask.end(), 0);
    size_t t = K;
    while((z1_t<0 || z2_t <0) and z3_t < m){
        size_t kt = distrib(gen);
        for (auto k: action_space) {
            std::vector<double> v(std::move(bandit_ref->sample({k})[0])); // to move
            std::transform(means_t[k].begin(), means_t[k].end(), v.begin(), means_t[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
        }
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < i; ++j) {
                beta[i][j] = betaij(Ts[i], Ts[j], cg, sigma);
                beta[j][i] = beta[i][j];
            }
        }
        St.clear();
        St_comp.clear();
        opt.clear();
        opt_comp.clear();
        get_St
        get_opt
        z1_t = get_z1t(means_t, St, beta, eps_1);
        z2_t = get_z2t(means_t, St_comp, beta, eps_1);
        z3_t = std::accumulate(opt_mask.begin(), opt_mask.end(), 0);
        ++t;
    }
    std::sort(St.begin(), St.end());
    // optimal_arms is always sorted;
    bool is_found = std::equal(St.begin(), St.end(), bandit_ref->optimal_arms.begin(), bandit_ref->optimal_arms.end()); // optimal_set foundp
    return std::pair<std::pair<size_t, bool>, std::vector<size_t>> {std::pair<size_t, bool>{std::accumulate(Ts.begin(), Ts.end(), size_t{0}), is_found}, St};
}
psi_auer::psi_auer(bandit &bandit_ref):policy(bandit_ref){};
std::pair<std::pair<bool, std::vector<size_t>>, std::vector<size_t>> psi_auer::loop(const size_t& seed, const double& delta, const double& eps, const size_t& k) {
    this->eps = eps;
    this->delta = delta;
    double cg = Cg(this->delta);
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<size_t> Ts(K, 0);
    std::vector<size_t> A1_t(action_space); // see paper
    std::vector<size_t> P1_t;
    std::vector<size_t> P2_t;
    std::vector<size_t> optimal_arms;
    std::vector<size_t> A1_t_set_minus_P1_t ;
    std::vector<std::vector<double>> mus_t(K, std::vector<double>(D));
    optimal_arms.reserve(K);
    P1_t.reserve(K);
    P2_t.reserve(K);
    A1_t_set_minus_P1_t.reserve(K);
    while(!A1_t.empty()){
        // sample all active arms
        for (auto a: A1_t){
            std::vector<double> v = std::move(bandit_ref->sample({a})[0]);
            std::transform(mus_t[a].begin(), mus_t[a].end(), v.begin(), mus_t[a].begin(),[&](double mean_t, double xval){
                return (xval + (double)(Ts[a])*mean_t) / ((double)Ts[a] + 1.);
            });
            ++Ts[a];
        }
        // update betas
        // update to remove suboptimal arms
        A1_t.erase(std::remove_if(A1_t.begin(), A1_t.end(), [&A1_t, &mus_t, &Ts, this, &cg](size_t i){
            return std::any_of(A1_t.begin(), A1_t.end(), [&Ts, &mus_t, &i, this, &cg](size_t j){
                return (std::max(minimum_quantity_non_dom(mus_t[i], mus_t[j], 0.),0.)  > (betaij(Ts[i], Ts[j], cg,sigma)));
            });
        }), A1_t.end());
        // compute P1_t
        std::copy_if(A1_t.begin(), A1_t.end(), std::back_inserter(P1_t), [&A1_t, &mus_t, &eps, &Ts, this, &cg](size_t i){
            return std::all_of(A1_t.begin(), A1_t.end(), [&](size_t j){
                return (std::max(minimum_quantity_dom(mus_t[i], mus_t[j], eps),0.) + INF*(i==j)) >(betaij(Ts[i], Ts[j], cg, sigma));
            });
        });
        // recover the set A_1 \P_1
        std::copy_if(A1_t.begin(), A1_t.end(), std::back_inserter(A1_t_set_minus_P1_t), [&P1_t](size_t i){
            return (std::find(P1_t.begin(), P1_t.end(), i) == P1_t.end());
        });
        // compute P2 (see paper)
        if (A1_t_set_minus_P1_t.empty()){
            P2_t = P1_t;
        }
        else {
            std::copy_if(P1_t.begin(), P1_t.end(), std::back_inserter(P2_t), [&mus_t, &eps, &Ts, &A1_t_set_minus_P1_t, this, &cg](size_t i){
                return !any_of(A1_t_set_minus_P1_t.begin(), A1_t_set_minus_P1_t.end(), [&, this](size_t j ){
                    return std::max(minimum_quantity_dom(mus_t[j], mus_t[i], eps),0.) <=(betaij(Ts[i], Ts[j], cg, sigma));
                });
            });
        }
// check if we have at least k optimal arms then break and prepare return
if ((optimal_arms.size() + P1_t.size())>= k){
    if (!P2_t.empty()){
        std::copy(P2_t.begin(), P2_t.end(), std::back_inserter(optimal_arms));}
    break;
}
        if (!P2_t.empty()){
            A1_t.erase(std::remove_if(A1_t.begin(), A1_t.end(), [&P2_t](size_t i){
                return (std::find(P2_t.begin(), P2_t.end(), i)!= P2_t.end());
            }), A1_t.end());
            std::copy(P2_t.begin(), P2_t.end(), std::back_inserter(optimal_arms));}
        // clear all the data
        P1_t.clear();
        P2_t.clear();
        A1_t_set_minus_P1_t.clear();
    }
    std::sort(optimal_arms.begin(), optimal_arms.end());
    // pareto_optimal_arms is always sorted;
    // the boolean value should not be considered when eps\neq 0
    // checking correctness if further done in Python
    bool os_found = std::equal(optimal_arms.begin(), optimal_arms.end(), bandit_ref->optimal_arms.begin(), bandit_ref->optimal_arms.end()); // optimal_set found
    return std::pair<std::pair<bool, std::vector<size_t>>, std::vector<size_t>>{std::pair<bool, std::vector<size_t>>{ os_found, Ts}, optimal_arms};
}
