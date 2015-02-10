#ifndef __DP_HPP
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <random>

typedef Eigen::VectorXd VXd;
typedef Eigen::MatrixXd MXd;
typedef boost::math::digamma boost_psi;
typedef boost::math::lgamma boost_lgamma;
double boost_lbeta(double a, double b);

class VarDPResults{
	public:
		MXd zeta;
		MXd a, b, eta;
		std::vector<double> times, objs, testlls;
		void save(std::string filename);
};

template<class Model>
class VarDP{
	public:
		VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t K);
		void run(bool computeTestLL = false, double tol = 1e-6);
		VarDPResults getResults();
	private:
		void initWeightsParams();
		void updateWeightDist();
		void updateLabelDist();
		void updateParamDist();
		double computeObjective();
		double computeTestLogLikelihood();

		std::mt19937 rng;

		double alpha;
		uint32_t K, M, N, Nt; //K is the # components in the model, M is the dimension of the statistic
		Model model;
		MXd zeta, sumzetaT, dlogh_deta, eta, train_stats;
		VXd a, b, psisum, nu, logh, dlogh_dnu, sumzeta;
		std::vector<double> times, objs, testlls;
		std;:vector<VXd> test_data;
};

#include "dp_impl.hpp"

#define __DP_HPP
#endif /* __DP_HPP */
