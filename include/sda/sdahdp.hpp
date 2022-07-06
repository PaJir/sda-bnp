#ifndef __SDAHDP_HPP
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
//#include <boost/filesystem.hpp>
//#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
//#include <random>
#include<thread>
#include<mutex>
#include <sdabnp/infer/dp.hpp>
#include <sdabnp/infer/hdp.hpp>
#include <sdabnp/util/timer.hpp>
#include <sdabnp/util/matching.hpp>
#include <sdabnp/util/trace.hpp>
#include <sdabnp/util/pool.hpp>
#include <cassert>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
typedef Eigen::MatrixXi MXi;
using boost::math::lgamma;

template<class Model>
class SDAHDP{
public:
    SDAHDP(const std::vector< std::vector<VXd> >& test_data, const Model& model,double gam, double alpha, uint32_t Knew,uint32_t Tnew, uint32_t nthr);
    void submitMinibatch(const std::vector< std::vector<VXd> >& train_data);
    void waitUntilDone();
    typename VarHDP<Model>::VarHDPResults getResults();
    MultiTrace getTrace();
private:
    double computeTestLogLikelihood(typename VarHDP<Model>::VarHDPResults dist0);
    typename VarHDP<Model>::VarHDPResults mergeDistributions(typename VarHDP<Model>::VarHDPResults d1, typename VarHDP<Model>::VarHDPResults d2, typename VarHDP<Model>::VarHDPResults d0);

    Timer timer;
    double alpha;
    double gam;
    uint32_t Knew;
    uint32_t T;
    Model model;
    typename VarHDP<Model>::VarHDPResults dist;
    std::mutex distmut;
    MultiTrace mtrace;
    MXd test_mxd;
    std::vector< std::vector<VXd> >& test_data;
    std::vector< typename VarHDP<Model>::VarHDPResults > dists;

    void varHDPJob(const std::vector<VXd>& train_data);
    Pool<std::function<void()> > pool;
};

#include "sdahdp_impl.hpp"


#define __SDAHDP_HPP
#endif /* __SDAHDP_HPP */
