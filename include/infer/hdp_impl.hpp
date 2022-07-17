#ifndef __HDP_IMPL_HPP

template<class Model>
VarHDP<Model>::VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const Model& model, double gam, double alpha, uint32_t T, uint32_t K) :model(model), test_data(test_data), gam(gam), alpha(alpha), T(T), K(K){
	std::cout<<"init without prior"<<std::endl;
    this->M = this->model.getStatDimension();
	this->N = train_data.size();
	this->Nt = test_data.size();
//    K0 = 0;
    T0 = 0;
    eta0 = MXd::Zero(T, M);

	for (uint32_t i = 0; i < N; i++){
		this->Nl.push_back(train_data[i].size());
		train_stats.push_back(MXd::Zero(Nl.back(), M));
		for (uint32_t j = 0; j < Nl.back(); j++){
			train_stats.back().row(j) = this->model.getStat(train_data[i][j]).transpose();
		}
	}
	for (uint32_t i = 0; i < Nt; i++){
		this->Ntl.push_back(test_data[i].size());
	}

	//seed random gen
	std::random_device rd;
	rng.seed(rd());

	//initialize memory
    // T is global, K is local
	nu = logh = dlogh_dnu = psiuvsum = phizetasum = phisum = VXd::Zero(T); // global part
	eta = dlogh_deta = phizetaTsum = MXd::Zero(T, M); // global part
	u = v = VXd::Zero(T);
	for (uint32_t i = 0; i < N; i++){
        // local part
		a.push_back(VXd::Zero(K));
		b.push_back(VXd::Zero(K));
		psiabsum.push_back(VXd::Zero(K));
		zeta.push_back(MXd::Zero(Nl[i], K));
		zetasum.push_back(VXd::Zero(K));
		phiNsum.push_back(VXd::Zero(K));
		phiEsum.push_back(MXd::Zero(K, M));
		zetaTsum.push_back(MXd::Zero(K, M));
		phi.push_back(MXd::Zero(K, T));
	}
}

template<class Model>
VarHDP<Model>::VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const VarHDPResults& prior,const Model& model, double gam, double alpha, uint32_t T, uint32_t K) : model(model), test_data(test_data), gam(gam), alpha(alpha), T(T), K(K){
    std::cout<<"init with prior"<<std::endl;

    this->M = this->model.getStatDimension();
    this->N = train_data.size();
    this->Nt = test_data.size();
    std::cout<<"done getting M N Nt"<<std::endl;
    T0 = prior.T;
    K0 = prior.K;
    eta0 = prior.eta;
    nu0 = prior.nu;
    u0 = prior.u;
    v0 = prior.v;
    std::cout<<"done getting prior"<<std::endl;

    for (uint32_t i = 0; i < N; i++){
        this->Nl.push_back(train_data[i].size());
        train_stats.push_back(MXd::Zero(Nl.back(), M));
        for (uint32_t j = 0; j < Nl.back(); j++){
            train_stats.back().row(j) = this->model.getStat(train_data[i][j]).transpose();
        }
    }
    std::cout<<"done getting train_stats"<<std::endl;
    for (uint32_t i = 0; i < Nt; i++){
        this->Ntl.push_back(test_data[i].size());
    }
    std::cout<<"done getting test_data"<<std::endl;
    //seed random gen
    std::random_device rd;
    rng.seed(rd());
    std::cout<<"done getting rng params"<<std::endl; //
    //initialize memory
    // T is global, K is local
    nu = logh = dlogh_dnu = psiuvsum = phizetasum = phisum = VXd::Zero(T); // global part
    std::cout<<"done getting global params"<<std::endl; //
    eta = dlogh_deta = phizetaTsum = MXd::Zero(T, M); // global part
    std::cout<<"done getting global params"<<std::endl; //
    u = v = VXd::Zero(T);
    std::cout<<"done getting global params"<<std::endl; //
    for (uint32_t i = 0; i < N; i++){
        // local part
        a.push_back(VXd::Zero(K));
        b.push_back(VXd::Zero(K));
        psiabsum.push_back(VXd::Zero(K));
        zeta.push_back(MXd::Zero(Nl[i], K));
        zetasum.push_back(VXd::Zero(K));
        phiNsum.push_back(VXd::Zero(K));
        phiEsum.push_back(MXd::Zero(K, M));
        zetaTsum.push_back(MXd::Zero(K, M));
        phi.push_back(MXd::Zero(K, T)); // 将K映射到T？
    }
    std::cout<<"done getting local params"<<std::endl;

    MXd tmp1 = MXd::Zero(eta0.rows(), eta0.cols());
    VXd tmp2 = VXd::Zero(nu0.size());
    this->model.getLogH(eta0,nu0,logh0,tmp1,tmp2);
    std::cout<<"done getting logh"<<std::endl;
}

template<class Model>
void VarHDP<Model>::init(){
    std::cout<<"init"<<std::endl;
	//use kmeans++ to break symmetry in the intiialization
	int Nlsum = 0;
	for (uint32_t i = 0; i < N; i++){
		Nlsum += Nl[i];
	}
	MXd tmp_stats = MXd::Zero( std::min(1000, Nlsum), M );
//    MXd tmp_stats = MXd::Zero( Nlsum, M );
	std::uniform_int_distribution<> uni(0, N-1);
	for (uint32_t i = 0; i < tmp_stats.rows(); i++){
		int gid = uni(rng);
		std::uniform_int_distribution<> unil(0, Nl[gid]-1);
		int lid = unil(rng);
		tmp_stats.row(i) = train_stats[gid].row(lid);
	}

    // kmeanspp 仅在init用到
    std::vector<double> maxMinDists;
    // done:kmeaspp部分找到eta0和K0 eta0是有问题的 现在只能保证跑通 and prior hdp中是没有的
//    uint32_t K0 = 0;
    std::cout<<"start kmeanspp"<<std::endl;
    // todo:bug fix 这里设置成T0 = 0 就能跑(就会解锁下面的bug)
    std::vector<uint32_t> idces = kmeanspp(tmp_stats, [this](VXd& x, VXd& y){ return model.naturalParameterDistSquared(x, y); }, T, eta0, T0, rng, maxMinDists);
//    std::vector<uint32_t> idces = kmeanspp(tmp_stats, [this](VXd& x, VXd& y){ return model.naturalParameterDistSquared(x, y); }, T, eta0, T0, rng, maxMinDists);
//	std::vector<uint32_t> idces = kmeanspp(tmp_stats, [this](VXd& x, VXd& y){ return model.naturalParameterDistSquared(x, y); }, T, rng);
    std::cout<<"end kmeanspp"<<std::endl;

    for (uint32_t t = 0; t < T; t++){
		//Update the parameters 
	    for (uint32_t j = 0; j < M; j++){
			if (t < T - T0) {
				eta(t, j) = model.getEta0()(j)+tmp_stats(idces[t], j);
			} else {
				eta(t, j) = model.getEta0()(j);
			}
	    }
		nu(t) = model.getNu0() + 1.0;
	}
    std::cout<<"end getEta0"<<std::endl;

	//update logh/etc
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);
    std::cout<<"end getLogH"<<std::endl;
	//initialize the global topic weights
	u = VXd::Ones(T);
	v = gam*VXd::Ones(T);

	//initial local params
	for(uint32_t i =0; i < N; i++){ // 为什么是N
		//local weights
		a[i] = VXd::Ones(K-1);
		b[i] = alpha*VXd::Ones(K-1);
		//local psiabsum
		psiabsum[i] = VXd::Zero(K);
		double psibk = 0.0;
		for (uint32_t k = 0; k < K-1; k++){
    		double psiak = digamma(a[i](k)) - digamma(a[i](k)+b[i](k));
    		psiabsum[i](k) = psiak + psibk;
    		psibk += digamma(b[i](k)) - digamma(a[i](k)+b[i](k));
		}
		psiabsum[i](K-1) = psibk;
        std::cout<<"end getpsiabsum"<<std::endl;

		//local correspondences
		//go through the data in document i, sum up -stat.T*dloghdeta
		std::vector< std::pair<uint32_t, double> > asgnscores;
		for (uint32_t t = 0; t < T; t++){
			asgnscores.push_back( std::pair<uint32_t, double>(t, 0.0) );
			for (uint32_t j = 0; j < train_stats[i].rows(); j++){
				for (uint32_t m = 0; m < M; m++){
					asgnscores[t].second -= train_stats[i](j, m)*dlogh_deta(t, m);
				}
			}
		}
        std::cout<<"end getasgnscores"<<std::endl;
        // todo:bug fix asgnscores
		//take the top K and weight more heavily for dirichlet
		std::sort(asgnscores.begin(), asgnscores.end(), [] (std::pair<uint32_t, double> s1, std::pair<uint32_t, double> s2){ return s1.second > s2.second;});
		phi[i] = MXd::Ones(K, T);
		for (uint32_t k = 0; k < K; k++){
			phi[i](k, asgnscores[k].first) += 3;
		}
		for (uint32_t k = 0; k < K; k++){
			double csum = 0.0;
			for(uint32_t t = 0; t < T; t++){
				std::gamma_distribution<> gamd(phi[i](k, t), 1.0);
				phi[i](k, t) =  gamd(rng);
				csum += phi[i](k, t);
			}
			for(uint32_t t = 0; t < T; t++){
				phi[i](k, t) /=  csum; 
			}
		}

		for(uint32_t k = 0; k < K; k++){
			for(uint32_t t = 0; t < T; t++){
				phiNsum[i](k) += phi[i](k, t)*dlogh_dnu(t);
				phiEsum[i].row(k) += phi[i](k, t)*dlogh_deta.row(t);
			}
		}
        std::cout<<"end getphi"<<std::endl;
		//everything needed for the first label update is ready
	}
	
}


template<class Model>
void VarHDP<Model>::run(bool computeTestLL, double tol){
	//clear any previously stored results
	times.clear();
	objs.clear();
	testlls.clear();

	//create objective tracking vars
	double diff = 10.0*tol + 1.0;
	double obj = std::numeric_limits<double>::infinity();
	double prevobj = std::numeric_limits<double>::infinity();

	//start the timer
	Timer cpuTime, wallTime;
	cpuTime.start();
	wallTime.start();

	//initialize the variables
	init();

	//loop on variational updates
	while(diff > tol){

		//update the local distributions
		updateLocalDists(tol);
        std::cout<<"done update the local distributions"<<std::endl;
		//update the global distribution
		updateGlobalDist();
        std::cout<<"done update the global distribution"<<std::endl;

		prevobj = obj;
		//store the current time
		times.push_back(cpuTime.get());
		//compute the objective
		obj = computeFullObjective();
        std::cout<<"done update computeFullObjective"<<std::endl;
		//save the objective
		objs.push_back(obj);
		//compute the obj diff
		diff = fabs((obj - prevobj)/obj);
        std::cout<<"done update diff"<<std::endl;
		//if test likelihoods were requested, compute those (but pause the timer first)
		if (computeTestLL){
			cpuTime.stop();
			double testll = computeTestLogLikelihood();
			testlls.push_back(testll);
			cpuTime.start();
			std::cout << "obj: " << obj << " testll: " << testll << std::endl;
		} else {
			std::cout << "obj: " << obj << std::endl;
		}
	}
	//done!
	return;

}

template<class Model>
typename VarHDP<Model>::VarHDPResults VarHDP<Model>::getResults(){
    std::cout<<"getting results"<<std::endl;
    // 相当于distribution
	VarHDP<Model>::VarHDPResults hdpr;


    hdpr.K = this->K;
    hdpr.T = this->T;

	hdpr.eta = this->eta;
	hdpr.nu = this->nu;
	hdpr.u = this->u;
	hdpr.v = this->v;
	hdpr.zeta = this->zeta;
//    for (uint32_t i = 0; i < hdpr.zeta.size(); i++){
//
//        hdpr.sumz.push_back((this->zeta[i].colwise().sum()).transpose()); // 和sumzeta一样的
//        hdpr.logp0.push_back((((1.0-this->zeta[i].array()).log()).colwise().sum()).transpose());
//        for (uint32_t k = 0; k < K; k++){
//            if (d.logp0(k) < -800.0){ //stops numerical issues later on -- approximation is good enough, for all intents and purposes exp(-800) = 0
//                d.logp0(k) = -800.0;
//            }
//        }
//    }



	hdpr.phi = this->phi;
    std::cout<<"done getting params"<<std::endl;

    std::cout<<"phizetaTsum"<<this->phizetaTsum<<std::endl;



//    hdpr.sumz = (this->phizetaTsum.rowwise().sum()).transpose();
    hdpr.sumz = this->phisum;
    std::cout<<"done computing sumz"<<std::endl;

//    hdpr.logp0 = (((1.0-this->phizetaTsum.array()).log()).rowwise().sum()).transpose();
    hdpr.logp0 = (((this->phisum.maxCoeff()+1)-this->phisum.array()).log()).transpose();
    std::cout<<hdpr.logp0.transpose()<<std::endl;
    std::cout<<"done computing logp0"<<std::endl;
//    for (uint32_t k = 0; k < T-1; k++){
//        if (hdpr.logp0(k) < -800.0){ //stops numerical issues later on -- approximation is good enough, for all intents and purposes exp(-800) = 0
//            hdpr.logp0(k) = -800.0;
//        }
//    }
    std::cout<<"done computing logp0"<<std::endl;

    // done: 是否需要push back
//    for (auto vec: this->zeta){
//        tmp_sumz = vec.colwise().sum().transpose();
//        hdpr.sumz.push_back(tmp_sumz);
//    }
//    for (auto vec: this->zeta){
//        tmp_logp0 = (((1.0-vec.array()).log()).colwise().sum()).transpose();
//        hdpr.logp0.push_back(tmp_sumz);
//    }

//    hdpr.sumz = (this->zeta.colwise().sum()).transpose();
//    hdpr.logp0 = (((1.0-this->zeta.array()).log()).colwise().sum()).transpose();

	hdpr.a = this->a;
	hdpr.b = this->b;
	hdpr.times = this->times;
	hdpr.objs = this->objs;
	hdpr.testlls = this->testlls;

//    hdpr.T = this->T;
//    hdpr.K = this->Knew;


	return hdpr;
}

template<class Model>
void VarHDP<Model>::VarHDPResults::save(std::string name){
	for (uint32_t i = 0; i < zeta.size(); i++){
//        std::cout<<zeta.size()<<std::endl;
		std::ostringstream ossz, ossp, ossab;
		ossz << name << "-zeta-" << i << ".log";
		ossp << name << "-phi-" << i << ".log";
		ossab << name << "-ab-" << i << ".log";

		std::ofstream out_z(ossz.str().c_str(), std::ios_base::trunc);
		out_z << zeta[i];
		out_z.close();

		std::ofstream out_p(ossp.str().c_str(), std::ios_base::trunc);
		out_p << phi[i];
		out_p.close();

		std::ofstream out_ab(ossab.str().c_str(), std::ios_base::trunc);
		out_ab << a[i].transpose() << std::endl << b[i].transpose();
		out_ab.close();
	}
	
	std::ofstream out_e(name+"-eta.log", std::ios_base::trunc);
	out_e << eta;
	out_e.close();

	std::ofstream out_n(name+"-nu.log", std::ios_base::trunc);
	out_n << nu.transpose();
	out_n.close();

	std::ofstream out_uv(name+"-uv.log", std::ios_base::trunc);
	out_uv << u.transpose() << std::endl << v.transpose();
	out_uv.close();

    std::ofstream out_z(name+"-sumz.log", std::ios_base::trunc);
    out_z << sumz;
    out_z.close();

    std::ofstream out_lp(name+"-logp0.log", std::ios_base::trunc);
    out_lp << logp0;
    out_lp.close();

	std::ofstream out_trc(name+"-trace.log", std::ios_base::trunc);
	for (uint32_t i = 0; i < times.size(); i++){
		out_trc << times[i] << " " << objs[i];
		if (i < testlls.size()){
			out_trc << " " << testlls[i] << std::endl;
		} else {
			out_trc << std::endl;
		}
	}
	out_trc.close();
}

template<class Model>
void VarHDP<Model>::updateLocalDists(double tol){
	//zero out global stats
	phizetasum = VXd::Zero(T);
	phisum = VXd::Zero(T);
	phizetaTsum = MXd::Zero(T, M);
	//loop over all local obs collections
	for (uint32_t i = 0; i < N; i++){
		//create objective tracking vars
		double diff = 10.0*tol + 1.0;
		double obj = std::numeric_limits<double>::infinity(); // inf
		double prevobj = std::numeric_limits<double>::infinity();// inf

		//run variational updates on the local params
		while(diff > tol){
			updateLocalLabelDist(i);
//            std::cout<<"local part: done updateLocalLabelDist;"<<std::endl;
			updateLocalWeightDist(i);
//            std::cout<<"local part: done updateLocalWeightDist;"<<std::endl;
			updateLocalCorrespondenceDist(i);
//            std::cout<<"local part: done updateCorrespondenceDist;"<<std::endl;
			prevobj = obj;
			obj = computeLocalObjective(i);
//            std::cout<<"local part: done computeLocalObjective;"<<std::endl;
			//compute the obj diff
			diff = fabs((obj - prevobj)/obj);
		}
		//add phi/zeta to global stats
		for(uint32_t t = 0; t < T; t++){
			for(uint32_t k = 0; k < K; k++){
				phizetasum(t) += phi[i](k, t)*zetasum[i](k);
//                std::cout<<"local part: done compute phizetasum;"<<std::endl;
				phisum(t) += phi[i](k, t); // 会被用去更新全局的beta分布参数u v
//                std::cout<<"local part: done compute phisum;"<<std::endl;
				for (uint32_t j = 0; j < M; j++){
					phizetaTsum(t, j) += phi[i](k, t)*zetaTsum[i](k, j);
				}
//                std::cout<<"local part: done compute phizetaTsum;"<<std::endl;
			}
		}
	}
}

template<class Model>
void VarHDP<Model>::updateLocalWeightDist(uint32_t idx){
	//Update a, b, and psisum
    // 跟dp比多了一个纬度

	psiabsum[idx] = VXd::Zero(K);
	double psibk = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
		a[idx](k) = 1.0+zetasum[idx](k);
		b[idx](k) = alpha;
		for (uint32_t j = k+1; j < K; j++){
			b[idx](k) += zetasum[idx](j);
		}
    	double psiak = digamma(a[idx](k)) - digamma(a[idx](k)+b[idx](k));
    	psiabsum[idx](k) = psiak + psibk;
    	psibk += digamma(b[idx](k)) - digamma(a[idx](k)+b[idx](k));
	}
	psiabsum[idx](K-1) = psibk;
}

template<class Model>
void VarHDP<Model>::updateLocalLabelDist(uint32_t idx){
	//update the label distribution
	zetasum[idx] = VXd::Zero(K);
	zetaTsum[idx] = MXd::Zero(K, M);
	for (uint32_t i = 0; i < Nl[idx]; i++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) = psiabsum[idx](k) - phiNsum[idx](k);
			for (uint32_t j = 0; j < M; j++){
				zeta[idx](i, k) -= train_stats[idx](i, j)*phiEsum[idx](k, j);
			}
			logpmax = (zeta[idx](i, k) > logpmax ? zeta[idx](i, k) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) -= logpmax;
			zeta[idx](i, k) = exp(zeta[idx](i, k));
			psum += zeta[idx](i, k);
		}
		//normalize
		for (uint32_t k = 0; k < K; k++){
			zeta[idx](i, k) /= psum;
		}
		//update the zetasum stats
		zetasum[idx] += zeta[idx].row(i).transpose();
		for(uint32_t k = 0; k < K; k++){
			zetaTsum[idx].row(k) += zeta[idx](i, k)*train_stats[idx].row(i);
		}
	}
}

template<class Model>
void VarHDP<Model>::updateLocalCorrespondenceDist(uint32_t idx){
	//update the correspondence distribution

    // local的坐标是idx
	phiNsum[idx] = VXd::Zero(K);
	phiEsum[idx] = MXd::Zero(K, M);
	
	for (uint32_t k = 0; k < K; k++){
		//compute the log of the weights, storing the maximum so far
		double logpmax = -std::numeric_limits<double>::infinity();
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) = psiuvsum(t) - zetasum[idx](k)*dlogh_dnu(t);
			for (uint32_t j = 0; j < M; j++){
				phi[idx](k, t) -= zetaTsum[idx](k, j)*dlogh_deta(t, j);
			}
			logpmax = (phi[idx](k, t) > logpmax ? phi[idx](k, t) : logpmax);
		}
		//make numerically stable by subtracting max, take exp, sum them up
		double psum = 0.0;
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) -= logpmax;
			phi[idx](k, t) = exp(phi[idx](k, t));
			psum += phi[idx](k, t);
		}
		//normalize
		for (uint32_t t = 0; t < T; t++){
			phi[idx](k, t) /= psum;
		}
		//update the phisum stats
		for(uint32_t t = 0; t < T; t++){
			phiNsum[idx](k) += phi[idx](k, t)*dlogh_dnu(t);
			phiEsum[idx].row(k) += phi[idx](k, t)*dlogh_deta.row(t);
		}
	}
}



template<class Model>
void VarHDP<Model>::updateGlobalDist(){
	updateGlobalWeightDist();
	updateGlobalParamDist();
}

template<class Model>
void VarHDP<Model>::updateGlobalWeightDist(){
	//Update u, v, and psisum psiuvsum(t) 权重之和
	psiuvsum = VXd::Zero(T);
	double psivt = 0.0;
	for (uint32_t t = 0; t < T-1; t++){
		u(t) = 1.0+phisum(t);
		v(t) = gam;
		for (uint32_t j = t+1; j < T; j++){
			v(t) += phisum(j);
		}
    	double psiut = digamma(u(t)) - digamma(u(t)+v(t)); // elog betak
    	psiuvsum(t) = psiut + psivt;
    	psivt += digamma(v(t)) - digamma(u(t)+v(t)); // elog (1-betek)
	}
	psiuvsum(T-1) = psivt;
}

template<class Model>
void VarHDP<Model>::updateGlobalParamDist(){
    // 全局参数更新

	for (uint32_t t = 0; t < T; t++){
		nu(t) = model.getNu0() + phizetasum(t);
		for (uint32_t j = 0; j < M; j++){
			eta(t, j) = model.getEta0()(j) + phizetaTsum(t, j);
		}
	}
	model.getLogH(eta, nu, logh, dlogh_deta, dlogh_dnu);
}

template<class Model>
double VarHDP<Model>::computeFullObjective(){


	//reuse the local code for computing each local obj
	double obj = 0;
	for (uint32_t i =0 ; i < N; i++){
		obj += computeLocalObjective(i);
	}

	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t t = 0; t < T-1; t++){
        betaEntropy += -boost_lbeta(u(t), v(t)) + (u(t)-1.0)*digamma(u(t)) +(v(t)-1.0)*digamma(v(t))-(u(t)+v(t)-2.0)*digamma(u(t)+v(t));
	}

	//get the variational exponential family entropy
	double expEntropy = 0.0;
	for (uint32_t t = 0; t < T; t++){
		expEntropy += logh(t) - nu(t)*dlogh_dnu(t);
		for (uint32_t j = 0; j < M; j++){
			expEntropy -= eta(t, j)*dlogh_deta(t, j);
		}
	}

	//prior exp cross entropy
    double priorExpXEntropy = T*model.getLogH0();
	for (uint32_t t = 0; t < T; t++){
		priorExpXEntropy -= model.getNu0()*dlogh_dnu(t);
	    for (uint32_t j=0; j < M; j++){
	    	priorExpXEntropy -= model.getEta0()(j)*dlogh_deta(t, j);
	    }
	}

	//get the prior beta cross entropy
	double priorBetaXEntropy = -T*boost_lbeta(1.0, alpha);
	for (uint32_t t = 0; t < T-1; t++){
		priorBetaXEntropy += (alpha-1.0)*(digamma(v(t)) - digamma(u(t)+v(t)));
	}

	//output
	return obj
		+ betaEntropy
		+ expEntropy
		- priorExpXEntropy
		- priorBetaXEntropy;
}

template<class Model>
double VarHDP<Model>::computeLocalObjective(uint32_t idx){
	//get the label entropy
	MXd mzero = MXd::Zero(zeta[idx].rows(), zeta[idx].cols());
	MXd zlogz = zeta[idx].array()*zeta[idx].array().log();
	double labelEntropy = ((zeta[idx].array() > 1.0e-16).select(zlogz, mzero)).sum();
//    std::cout<<"done labelEntropy"<<std::endl;

	//get the correspondence entropy
	MXd pzero = MXd::Zero(phi[idx].rows(), phi[idx].cols());
	MXd plogp = phi[idx].array()*phi[idx].array().log();
	double corrEntropy = ((phi[idx].array() > 1.0e-16).select(plogp, pzero)).sum();
//    std::cout<<"done corrEntropy"<<std::endl;

	//get the variational beta entropy
	double betaEntropy = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
        betaEntropy += -boost_lbeta(a[idx](k), b[idx](k)) + (a[idx](k)-1.0)*digamma(a[idx](k)) +(b[idx](k)-1.0)*digamma(b[idx](k))-(a[idx](k)+b[idx](k)-2.0)*digamma(a[idx](k)+b[idx](k));
	}
//    std::cout<<"done betaEntropy"<<std::endl;

	//get the likelihood cross entropy
	double likelihoodXEntropy = 0.0;
	for (uint32_t k = 0; k < K; k++){
		likelihoodXEntropy -= zetasum[idx](k)*phiNsum[idx](k);
		for (uint32_t j = 0; j < M; j++){
			likelihoodXEntropy -= zetaTsum[idx](k, j)*phiEsum[idx](k, j);
		}
	}
//    std::cout<<"done likelihoodXEntropy"<<std::endl;

	//get the prior label cross entropy
	double priorLabelXEntropy = 0.0;
	double psibk = 0.0;
	for (uint32_t k = 0; k < K-1; k++){
		double psiak = digamma(a[idx](k)) - digamma(a[idx](k)+b[idx](k));
		priorLabelXEntropy += zetasum[idx](k)*(psiak + psibk);
		psibk += digamma(b[idx](k)) - digamma(a[idx](k)+b[idx](k));
	}
	priorLabelXEntropy += zetasum[idx](K-1)*psibk;
//    std::cout<<"done priorLabelXEntropy"<<std::endl;

	//get the prior correspondence cross entropy
	double priorCorrXEntropy = 0.0;
	double psivt = 0.0;
	for (uint32_t t = 0; t < T-1; t++){
		double psiut = digamma(u(t)) - digamma(u(t)+v(t));
		for (uint32_t k = 0; k < K; k++){
			priorCorrXEntropy += phi[idx](k, t)*(psiut + psivt);
		}
		psivt += digamma(v(t)) - digamma(u(t)+v(t));
	}
	for(uint32_t k = 0; k < K; k++){
		priorCorrXEntropy += phi[idx](k, T-1)*psivt;
	}
//    std::cout<<"done priorCorrXEntropy"<<std::endl;

	//get the prior beta cross entropy
	double priorBetaXEntropy = -K*boost_lbeta(1.0, alpha);
	for (uint32_t k = 0; k < K-1; k++){
		priorBetaXEntropy += (alpha-1.0)*(digamma(b[idx](k)) - digamma(a[idx](k)+b[idx](k)));
	}
//    std::cout<<"done priorBetaXEntropy"<<std::endl;

	return labelEntropy 
		 + corrEntropy
		 + betaEntropy 
		 - likelihoodXEntropy
		 - priorCorrXEntropy
		 - priorLabelXEntropy
		 - priorBetaXEntropy;
}

template<class Model>
double VarHDP<Model>::computeTestLogLikelihood(){

	//TODO: fill in
	//run local variational inference on some % of each test collection
	//compute logposteriorpredictive on the other % using the local variational params
//    test_mxd =



	return 0.0;

}

double boost_lbeta(double a, double b){
	return lgamma(a)+lgamma(b)-lgamma(a+b);
}

#define __HDP_IMPL_HPP
#endif /* __HDP_HPP */
