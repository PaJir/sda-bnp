#ifndef __DP_IMPL_HPP
// todo


template<class Model>
VarDP<Model>::VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t K) : model(model), alpha(alpha), K(K){
	M = this->model.getStatDimension();
	N = train_data.size();
	Nt = test_data.size();
	K0 = 0; // no nonstandard prior components

	//compute exponential family statistics once
	train_stats = MXd::Zero(N, M);
	for (uint32_t i = 0; i < N; i++){
		train_stats.row(i) = this->model.getStat(train_data[i]).transpose();
	}

	test_mxd = MXd::Zero(test_data[0].size(), Nt);
	for (uint32_t i =0; i < Nt; i++){
		test_mxd.col(i) = test_data[i];
	}

	//initialize the random device
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
	a = b = psisum = nu = sumzeta = dlogh_dnu = logh = VXd::Zero(K);
	zeta = MXd::Zero(N, K);
	sumzetaT = dlogh_deta = eta = MXd::Zero(K, M);
}

template<class Model>
VarDP<Model>::VarDP(const std::vector<VXd>& train_data, const std::vector<VXd>& test_data, const Distribution& prior, const Model& model, 
		double alpha, uint32_t K) : model(model), alpha(alpha), K(K){
	M = this->model.getStatDimension();
	N = train_data.size();
	Nt = test_data.size();
	K0 = prior.K;


	//compute exponential family statistics once
	train_stats = MXd::Zero(N, M);
	for (uint32_t i = 0; i < N; i++){
		train_stats.row(i) = this->model.getStat(train_data[i]).transpose();
	}

	test_mxd = MXd::Zero(test_data[0].size(), Nt);
	for (uint32_t i =0; i < Nt; i++){
		test_mxd.col(i) = test_data[i];
	}



	//initialize the random device
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
	a = b = psisum = nu = sumzeta = dlogh_dnu = logh = VXd::Zero(K);
    zeta = MXd::Zero(N, K);
	sumzetaT = dlogh_deta = eta = MXd::Zero(K, M);
	a0 = prior.a;
	b0 = prior.b;
	nu0 = prior.nu;
	eta0 = prior.eta;
	MXd tmp1 = MXd::Zero(eta0.rows(), eta0.cols());
	VXd tmp2 = VXd::Zero(nu0.size());
	this->model.getLogH(eta0, nu0, logh0, tmp1, tmp2);
}

template<class Model>
void VarDP<Model>::run(bool computeTestLL, double tol, uint32_t nItr){
	//clear any previously stored results
	trace.clear();

	//create objective tracking vars
	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();

	//start the timer
	Timer cpuTime;
	cpuTime.start();

	//initialize the variables
	init();

	//loop on variational updates
	uint32_t itr = 0;
	//if nItr > 0, use iteration count
	//otherwise, use convergence in variational objective
	while( (nItr > 0 ? (itr < nItr) : (diff > tol) ) ){
		itr++;
		updateWeightDist();
		updateParamDist();
		updateLabelDist();

		if (nItr == 0){ //only compute the objective if nItr was unspecified
			prevobj = obj;
			//compute the objective
			obj = computeObjective();
			//compute the obj diff
			diff = fabs((obj - prevobj)/obj);
		}

		//stop the clock & store the current time
		trace.times.push_back(cpuTime.stop());
		//save the objective
		trace.objs.push_back(obj);
		//save the current distribution 
		if (computeTestLL){
			trace.testlls.push_back(computeTestLogLikelihood());
		}
		//std::cout << "obj: " << obj << " testll: " << testll << std::endl;
		//restart the clock
		cpuTime.start(); 
	}
	//done!
	return;
}



template<class Model>
void VarDP<Model>::init(){


	//use kmeans++ to break symmetry in the intiialization
	//outputs K-K0 indices for new cluster initialization
	//tries to make them different from the first K0 cluster centers as well
	std::vector<double> maxMinDists;
	std::vector<uint32_t> idces = kmeanspp(train_stats, [this](VXd& x, VXd& y){ return model.naturalParameterDistSquared(x, y); }, K, eta0, K0, rng, maxMinDists);
	for (uint32_t k = 0; k < K; k++){
		//Update the parameters 
		if (k < K0){ //if this is one of the fixed prior clusters
			for (uint32_t j = 0; j < M; j++){
	    		eta(k, j) = eta0(k, j); //+train_stats(idces[k], j);
	    	}
			nu(k) = nu0(k); // + 1.0;
		} else { //otherwise if this is a reasonable sampled cluster, use kmpp
			for (uint32_t j = 0; j < M; j++){
	    		eta(k, j) = model.getEta0()(j)+train_stats(idces[k-K0], j);
	    	}
			nu(k) = model.getNu0() + 1.0;
		}
	}

	//initialize a/b to the prior
	double psibk = 0.0;
	for (uint32_t k = 0; k < K; k++){
		//update weights
		if (k < K0){
			a(k) = a0(k);
			b(k) = b0(k);
		} else {
			a(k) = 1.0;
			b(k) = alpha;
		}
    	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
    	psisum(k) = psiak + psibk;
    	psibk += digamma(b(k)) - digamma(a(k)+b(k));
	}

	//update logh/etc
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);

	updateLabelDist(); //finally make sure labels are updated



	//RANDOM INITIALIZATION (DUMB BUT WORKS REASONABLY WELL)
	//sumzeta = VXd::Zero(K);
	//sumzetaT = MXd::Zero(K, M);

	////fill in K-K0 etas with stats directly from data
	//std::uniform_int_distribution<> uniint(0, N-1);
	//if (K0 > 0){
	//	eta.block(0, 0, K0, M) = eta0;
	//	nu.head(K0) = nu0;
	//}
	//for (uint32_t k = K0; k < K; k++){
	//	eta.row(k) = train_stats.row(uniint(rng));
	//	nu(k) = 1.0;
	//}

	///*compute dist between cluster stats and data stats and use exp(-dist^2) as similarity*/
	//for (uint32_t k = 0; k < K; k++){
	//	zeta.col(k) = (train_stats.rowwise() - eta.row(k)/nu(k)).rowwise().squaredNorm();
	//}
	//zeta.colwise() -= zeta.rowwise().minCoeff().eval();
	//zeta = (-zeta.array()).exp().eval();
	//for (uint32_t i = 0; i < N; i++){
	//	zeta.row(i) /= zeta.row(i).sum();
	//}
	//sumzeta = zeta.colwise().sum().transpose();
	//sumzetaT = zeta.transpose()*train_stats;


	//RANDOM INIT (SAME AS ABOVE, LESS EFFICIENT)
	//for(uint32_t i = 0; i < N; i++){
	//	double rwsum = 0;
    //	double minDistSq = std::numeric_limits<double>::infinity();
	//	for(uint32_t k=0; k< K; k++){
	//		double distsq = (train_stats.row(i) - eta.row(k)/nu(k)).squaredNorm();
	//		zeta(i, k) = distsq;
    //  		minDistSq = minDistSq > distsq ? distsq : minDistSq;
	//	}
	//	for(uint32_t k=0; k < K; k++){
	//		zeta(i, k) = exp(-(zeta(i, k) -minDistSq));
	//		rwsum += zeta(i, k);
	//	}
	//	zeta.row(i) /= rwsum;
	//	sumzeta += zeta.row(i).transpose();
	//	sumzetaT += zeta.row(i).transpose()*train_stats.row(i);
	//}
	


	//std::cout << "INIT" << std::endl;
	//std::cout << "Eta: " << std::endl << eta << std::endl;
	//std::cout << "nu: " << std::endl << nu.transpose() << std::endl;
	//std::cout << "logh: " << std::endl << logh << std::endl;
	//std::cout << "dlogh_deta: " << std::endl << dlogh_deta << std::endl;
	//std::cout << "dlogh_dnu: " << std::endl << dlogh_dnu.transpose() << std::endl;
	//std::cout << "psisum: " << std::endl << psisum.transpose() << std::endl;
	//std::cout << "a: " << std::endl << a.transpose() << std::endl;
	//std::cout << "b: " << std::endl << b.transpose() << std::endl;
	return;
}

template<class Model>
void VarDP<Model>::updateWeightDist(){
	//Update a, b, and psisum
    // 更行模型超参数

	double psibk = 0.0;
	for (uint32_t k = 0; k < K; k++){
		if (k < K0){
			a(k) = a0(k) + sumzeta(k);
			b(k) = b0(k);
		} else {
			a(k) = 1.0 + sumzeta(k);
			b(k) = alpha;
		}
		for (uint32_t j = k+1; j < K; j++){
			b(k) += sumzeta(j);
		}
    	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
    	psisum(k) = psiak + psibk;
    	psibk += digamma(b(k)) - digamma(a(k)+b(k));
	}
	return;
}

template<class Model>
void VarDP<Model>::updateParamDist(){




	//Update the parameters
	eta = sumzetaT;
	nu = sumzeta;
	if (K0 > 0){
		eta.block(0, 0, K0, M) += eta0;
		eta.block(K0, 0, K-K0, M).rowwise() += model.getEta0().transpose();
		nu.head(K0) += nu0;
		nu.tail(K-K0).array() += model.getNu0();
	} else {
		eta.rowwise() += model.getEta0().transpose();
		nu.array() += model.getNu0();
	}


	//for (uint32_t k = 0; k < K; k++){
	//	if (k < K0){
	//    	for (uint32_t j = 0; j < M; j++){
	//    		eta(k, j) = eta0(k, j)+sumzetaT(k, j);
	//    	}
	//    	nu(k) = nu0(k) + sumzeta(k);
	//	} else {
	//		for (uint32_t j = 0; j < M; j++){
	//    		eta(k, j) = model.getEta0()(j)+sumzetaT(k, j);
	//    	}
	//    	nu(k) = model.getNu0() + sumzeta(k);
	//	}
	//}

	//update logh/etc
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);


	return;
}

template<class Model>
void VarDP<Model>::updateLabelDist(){



	//update the label distribution
	zeta.rowwise() = psisum.transpose() - dlogh_dnu.transpose();
	zeta -= train_stats*dlogh_deta.transpose();
	zeta.colwise() -= zeta.rowwise().maxCoeff().eval();
	zeta = zeta.array().exp().eval();
	for (uint32_t i = 0; i < N; i++){
		zeta.row(i) /= zeta.row(i).sum();
	}
	sumzeta = zeta.colwise().sum().transpose();
	sumzetaT = zeta.transpose()*train_stats;

	//sumzeta = VXd::Zero(K);
	//sumzetaT = MXd::Zero(K, M);
	//for (uint32_t i = 0; i < N; i++){
	//	//compute the log of the weights, storing the maximum so far
	//	double logpmax = -std::numeric_limits<double>::infinity();
	//	for (uint32_t k = 0; k < K; k++){
	//		zeta(i, k) = psisum(k) - dlogh_dnu(k);
	//		for (uint32_t j = 0; j < M; j++){
	//			zeta(i, k) -= train_stats(i, j)*dlogh_deta(k, j);
	//		}
	//		logpmax = (zeta(i, k) > logpmax ? zeta(i, k) : logpmax);
	//	}
	//	//make numerically stable by subtracting max, take exp, sum them up
	//	double psum = 0.0;
	//	for (uint32_t k = 0; k < K; k++){
	//		zeta(i, k) -= logpmax;
	//		zeta(i, k) = exp(zeta(i, k));
	//		psum += zeta(i, k);
	//	}
	//	//normalize
	//	for (uint32_t k = 0; k < K; k++){
	//		zeta(i, k) /= psum;
	//	}
	//	//update the sumzeta stats
	//	sumzeta += zeta.row(i).transpose();
	//	for(uint32_t k = 0; k < K; k++){
	//		sumzetaT.row(k) += zeta(i, k)*train_stats.row(i);
	//	}
	//}

	return;
}

template<class Model>
typename VarDP<Model>::Distribution VarDP<Model>::getDistribution(){
	VarDP<Model>::Distribution d;

	d.sumz = (this->zeta.colwise().sum()).transpose(); // 和sumzeta一样的
	d.logp0 = (((1.0-this->zeta.array()).log()).colwise().sum()).transpose();
	for (uint32_t k = 0; k < K; k++){
		if (d.logp0(k) < -800.0){ //stops numerical issues later on -- approximation is good enough, for all intents and purposes exp(-800) = 0
			d.logp0(k) = -800.0;
		}
	}
	d.K = this->K;
	d.a = this->a;
	d.b = this->b;
	d.eta = this->eta;
	d.nu = this->nu;
	return d;
}

template<class Model>
Trace VarDP<Model>::getTrace(){
	return trace;
}

template<class Model>
double VarDP<Model>::computeObjective(){

	//get the label entropy
	MXd mzerotmp = MXd::Zero(zeta.rows(), zeta.cols());
	MXd zlogztmp = zeta.array()*zeta.array().log();
	double labelEntropy = ((zeta.array() > 1.0e-16).select(zlogztmp, mzerotmp)).sum();

	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t k = 0; k < K; k++){
        betaEntropy += -boost_lbeta(a(k), b(k)) + (a(k)-1.0)*digamma(a(k)) +(b(k)-1.0)*digamma(b(k))-(a(k)+b(k)-2.0)*digamma(a(k)+b(k));
	}

	//get the variational exponential family entropy
	double expEntropy = logh.sum() -nu.transpose()*dlogh_dnu - (eta.array()*dlogh_deta.array()).sum();

	//get the likelihood cross entropy
	double likelihoodXEntropy = -sumzeta.transpose()*dlogh_dnu - (sumzetaT.array()*dlogh_deta.array()).sum();

	//get the prior exponential cross entropy
    double priorExpXEntropy = 0; 
    if (K0 > 0){
    	priorExpXEntropy += model.getLogH0()*(K-K0) + logh0.sum() - nu0.transpose()*dlogh_dnu.head(K0) - model.getNu0()*dlogh_dnu.tail(K-K0).sum();
    	priorExpXEntropy -= (dlogh_deta.block(0, 0, K0, M).array()*eta0.array()).sum();
    	for (uint32_t k = K0; k < K; k++){
    		priorExpXEntropy -= dlogh_deta.row(k)*model.getEta0();
		}
	} else {
		priorExpXEntropy += model.getLogH0()*K - model.getNu0()*dlogh_dnu.sum();
		for (uint32_t k = 0; k < K; k++){
			priorExpXEntropy -= dlogh_deta.row(k)*model.getEta0();
		}
	}

	//get the prior label cross entropy
	double priorLabelXEntropy = 0.0;
	double psibktmp = 0.0;
	for (uint32_t k = 0; k < K; k++){
		double psiak = digamma(a(k)) - digamma(a(k)+b(k));
		priorLabelXEntropy += sumzeta(k)*(psiak + psibktmp);
		psibktmp += digamma(b(k)) - digamma(a(k)+b(k));
	}

	//get the prior beta cross entropy
	double priorBetaXEntropy = 0; 
	for (uint32_t k = 0; k < K; k++){
		if (k < K0){
			priorBetaXEntropy += (a0(k)-1.0)*(digamma(a(k)) - digamma(a(k)+b(k))) + (b0(k)-1.0)*(digamma(b(k)) - digamma(a(k)+b(k))) - boost_lbeta(a0(k), b0(k));
		} else {
			priorBetaXEntropy += (alpha-1.0)*(digamma(b(k)) - digamma(a(k)+b(k))) - boost_lbeta(1.0, alpha);
		}
	}

	////get the label entropy
	//MXd mzero = MXd::Zero(zeta.rows(), zeta.cols());
	//MXd zlogz = zeta.array()*zeta.array().log();
	//double labelEntropy = ((zeta.array() > 1.0e-16).select(zlogz, mzero)).sum();
	//
	////get the variational beta entropy
	//double betaEntropy = 0.0;
	//for (uint32_t k = 0; k < K; k++){
    //    betaEntropy += -boost_lbeta(a(k), b(k)) + (a(k)-1.0)*digamma(a(k)) +(b(k)-1.0)*digamma(b(k))-(a(k)+b(k)-2.0)*digamma(a(k)+b(k));
	//}

	////get the variational exponential family entropy
	//double expEntropy = 0.0;
	//for (uint32_t k = 0; k < K; k++){
	//	expEntropy += logh(k) - nu(k)*dlogh_dnu(k);
	//	for (uint32_t j = 0; j < M; j++){
	//		expEntropy -= eta(k, j)*dlogh_deta(k, j);
	//	}
	//}

	////get the likelihood cross entropy
	//double likelihoodXEntropy = 0.0;
	//for (uint32_t k = 0; k < K; k++){
	//	likelihoodXEntropy -= sumzeta(k)*dlogh_dnu(k);
	//	for (uint32_t j = 0; j < M; j++){
	//		likelihoodXEntropy -= sumzetaT(k, j)*dlogh_deta(k, j);
	//	}
	//}

	////get the prior exponential cross entropy
    //double priorExpXEntropy = 0; 
	//for (uint32_t k = 0; k < K; k++){
	//	if (k < K0){
	//		priorExpXEntropy += logh0(k) - nu0(k)*dlogh_dnu(k);
	//		for (uint32_t j=0; j < M; j++){
	//    		priorExpXEntropy -= eta0(k, j)*dlogh_deta(k, j);
	//    	}
	//	} else{
	//		priorExpXEntropy +=  model.getLogH0() - model.getNu0()*dlogh_dnu(k);
	//		for (uint32_t j=0; j < M; j++){
	//    		priorExpXEntropy -= model.getEta0()(j)*dlogh_deta(k, j);
	//    	}
	//	}
	//}

	////get the prior label cross entropy
	//double priorLabelXEntropy = 0.0;
	//double psibk = 0.0;
	//for (uint32_t k = 0; k < K; k++){
	//	double psiak = digamma(a(k)) - digamma(a(k)+b(k));
	//	priorLabelXEntropy += sumzeta(k)*(psiak + psibk);
	//	psibk += digamma(b(k)) - digamma(a(k)+b(k));
	//}

	////get the prior beta cross entropy
	//double priorBetaXEntropy = 0; 
	//for (uint32_t k = 0; k < K; k++){
	//	if (k < K0){
	//		priorBetaXEntropy += (a0(k)-1.0)*(digamma(a(k)) - digamma(a(k)+b(k))) + (b0(k)-1.0)*(digamma(b(k)) - digamma(a(k)+b(k))) - boost_lbeta(a0(k), b0(k));
	//	} else {
	//		priorBetaXEntropy += (alpha-1.0)*(digamma(b(k)) - digamma(a(k)+b(k))) - boost_lbeta(1.0, alpha);
	//	}
	//}

	return labelEntropy 
		 + betaEntropy 
		 + expEntropy
		 - likelihoodXEntropy
		 - priorExpXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
}



template<class Model>
double VarDP<Model>::computeTestLogLikelihood(){

	if (Nt == 0){
		std::cout << "WARNING: Test Log Likelihood = NaN since Nt = 0" << std::endl;
	}

	//first find out how many empty clusters we have
	uint32_t firstEmptyK = K;
	for (uint32_t k = K-1; k >= 0; k--){
		if ( fabs(a(k)-1.0) < 1.0e-3){
			firstEmptyK = k;
		} else {
			break;
		}
	}

	if (firstEmptyK < K){
		//first get average weights -- compress the last empty clusters
		double stick = 1.0;
		VXd weights = VXd::Zero(firstEmptyK+1);
		for(uint32_t k = 0; k < firstEmptyK; k++){
			weights(k) = stick*a(k)/(a(k)+b(k));
			stick *= b(k)/(a(k)+b(k));
		}
		weights(firstEmptyK) = stick;

		MXd logp = model.getLogPosteriorPredictive(test_mxd, eta.block(0, 0, firstEmptyK+1, eta.cols()), nu.head(firstEmptyK+1)).array().rowwise() + (weights.transpose()).array().log();
		VXd llmaxs = logp.rowwise().maxCoeff();
		logp.colwise() -= llmaxs;
		VXd lls = ((logp.array().exp()).rowwise().sum()).log();
		return (lls + llmaxs).sum()/Nt;
	} else {
		//first get average weights -- no compression
		double stick = 1.0;
		VXd weights = VXd::Zero(K);
		for(uint32_t k = 0; k < K-1; k++){
			weights(k) = stick*a(k)/(a(k)+b(k));
			stick *= b(k)/(a(k)+b(k));
		}
		weights(K-1) = stick;

		MXd logp = model.getLogPosteriorPredictive(test_mxd, eta, nu).array().rowwise() + (weights.transpose()).array().log();
		VXd llmaxs = logp.rowwise().maxCoeff(); // 返回矩阵中的最大值
		logp.colwise() -= llmaxs;
		VXd lls = ((logp.array().exp()).rowwise().sum()).log();
		return (lls + llmaxs).sum()/Nt;
	}


	////now loop over all test data
	//double loglike = 0.0;
	//for(uint32_t i = 0; i < Nt; i++){
	//	//get the log likelihoods for all clusters
	//	std::vector<double> loglikes;
	//	for (uint32_t k = 0; k < K; k++){
	//		loglikes.push_back(log(weights(k)) + model.getLogPosteriorPredictive(test_data[i], eta.row(k), nu(k)));
	//	}
	//	//numerically stable sum
	//	//first sort in increasing order
	//	std::sort(loglikes.begin(), loglikes.end());
	//	//then sum in increasing order
	//	double like = 0.0;
	//	for (uint32_t k = 0; k < K; k++){
	//		//subtract off the max first
	//		like += exp(loglikes[k] - loglikes.back());
	//	}
	//	//now multiply by exp(max), take the log, and add to running loglike total
	//	loglike += loglikes.back() + log(like);
	//}
	//return loglike/Nt; //should return NaN if Nt == 0
}


//double boost_lbeta(double a, double b){
//	return lgamma(a)+lgamma(b)-lgamma(a+b);
//}

template<class Model>
void VarDP<Model>::Distribution::save(std::string name){
	std::ofstream out_z(name+"-sumz.log", std::ios_base::trunc);
	out_z << sumz;
	out_z.close();

	std::ofstream out_lp(name+"-logp0.log", std::ios_base::trunc);
	out_lp << logp0;
	out_lp.close();


	std::ofstream out_e(name+"-eta.log", std::ios_base::trunc);
	out_e << eta;
	out_e.close();

	std::ofstream out_n(name+"-nu.log", std::ios_base::trunc);
	out_n << nu.transpose();
	out_n.close();


	std::ofstream out_ab(name+"-ab.log", std::ios_base::trunc);
	out_ab << a.transpose() << std::endl << b.transpose();
	out_ab.close();
}

#define __DP_IMPL_HPP
#endif /* __DP_HPP */
