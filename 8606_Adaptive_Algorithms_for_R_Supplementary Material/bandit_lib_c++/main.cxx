#include <vector>
#include "src/cpp/utils.hpp"
#include "src/cpp/bandits.hpp"
#include "src/cpp/policies.hpp"
#include <iostream>
#include <cstdlib>
int main(int argc, char** argv) {
    std::vector<std::vector<double>> data{{0.75785738, 0.04866446},
    {0.4445091 , 0.19905503},
    {0.45422223, 0.9413393 },
    {0.89876041, 0.05671574},
    {0.80762673, 0.22822175},};
    size_t  K = 5, D=2;
    std::vector<double> arr = {0.001, 0.5};
    size_t i, k{0};
    gaussian gauss(data, arr);
    bernoulli bern(data);
    std::vector<double> mean(2,0);
    for (int i=0; i<5000; ++i){
        mean = sum(gauss.sample({4})[0], mean);
    }
    for (auto i =0; i<2; ++i){
        mean[i] /= 5000;
    }
    psi_auer auer(gauss);
    psi_ape ape(gauss);
    psi_uniform psi_unif(gauss);
    double result{0.};
    for(i=k;i<500; ++i){
        //cout<<auer.loop(i, 0.1, 0).first<<endl;
        auto res = auer.loop(i, 0.1, 0.0,  1);
        std::cout<<"[seed="<<i<<"]: "<<res.first.first<<" "<<std::boolalpha<<res.first.second<<std::endl;
        //auto ret_val = bandit.sample({0,1,2});
        //print_array2d(ret_val);
        result += res.first.first;
    }
    std::cout<<result/500<<std::endl;
    return 0;
}