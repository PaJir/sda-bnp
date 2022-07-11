#ifndef __SDADP_IMPL_HPP

template<class Model>
SDADP<Model>::SDADP(const std::vector<VXd>& test_data, const Model& model, double alpha, uint32_t Knew, uint32_t nthr):
test_data(test_data), model(model), alpha(alpha), Knew(Knew), pool(nthr){ // 线程数
	test_mxd = MXd::Zero(test_data[0].size(), test_data.size());
	for (uint32_t i =0; i < test_data.size(); i++){
		test_mxd.col(i) = test_data[i];
	}
	dist.K = 0;
	timer.start(); //start the clock -- used for tracking performance
}

template<class Model>
void SDADP<Model>::submitMinibatch(const std::vector<VXd>& train_data){
	pool.submit(std::bind(&SDADP<Model>::varDPJob, this, train_data));
}

template<class Model>
void SDADP<Model>::waitUntilDone(){
	pool.wait();
}

template<class Model>
typename VarDP<Model>::Distribution SDADP<Model>::getDistribution(){
	//have to lock/store since the worker pool might be doing stuff with it
	typename VarDP<Model>::Distribution out;
	{
		std::lock_guard<std::mutex> lock(distmut);
		out = dist;
	}
	return out;
}

template<class Model>
MultiTrace SDADP<Model>::getTrace(){
	MultiTrace mt;
	{
		std::lock_guard<std::mutex> lock(distmut);
		mt = mtrace;
	}

	for (uint32_t i = 0; i < dists.size(); i++){
		mt.testlls.push_back(computeTestLogLikelihood(dists[i]));
	}
	return mt;
}

template<class Model>
double SDADP<Model>::computeTestLogLikelihood(typename VarDP<Model>::Distribution dist0){
	//TESTING CODE FOR DEBUGGING -- similar values appear in vb.c test code
	//dist0.eta = MXd::Zero(3, 7);
	//dist0.nu = VXd::Ones(3);
    //dist0.eta << 518081,      -421473,      -421473,       346853,     -15409.5,      12600.5,      467.401,
    //  			   402553,       949145,       949145,  2.24606e+06,      10554.7,      24983.3,          286,
    //  				411719,       388877,       388877,       367628,      8890.26,      8394.19,          200;
	//dist0.nu(0) = 459.401;
	//dist0.nu(1) = 278;
	//dist0.nu(2) = 192;


	//dist0.a = VXd::Ones(3);
	//dist0.b = VXd::Zero(3);
	//dist0.b(0) = alpha+dist0.a(1)+dist0.a(2);
	//dist0.b(1) = alpha+dist0.a(2);
	//dist0.b(2) = alpha;

	//dist0.K = 3;


	uint32_t K = dist0.K;
	uint32_t Nt = test_mxd.cols();

	if (Nt == 0){
 		std::cout << "WARNING: Test Log Likelihood = NaN since Nt = 0" << std::endl;
	}

	//first get average weights -- no compression
	double stick = 1.0;
	VXd weights = VXd::Zero(K);
	for(uint32_t k = 0; k < K-1; k++){
		weights(k) = stick*dist0.a(k)/(dist0.a(k)+dist0.b(k));
		stick *= dist0.b(k)/(dist0.a(k)+dist0.b(k));
	}
	weights(K-1) = stick;

	MXd logp = model.getLogPosteriorPredictive(test_mxd, dist0.eta, dist0.nu).array().rowwise() + (weights.transpose()).array().log();
	VXd llmaxs = logp.rowwise().maxCoeff();
	logp.colwise() -= llmaxs;
	VXd lls = ((logp.array().exp()).rowwise().sum()).log();
	return (lls + llmaxs).sum()/Nt;

	////first get average weights
	//double stick = 1.0;
	//VXd weights = VXd::Zero(K);
	//for(uint32_t k = 0; k < K-1; k++){
	//	weights(k) = stick*dist0.a(k)/(dist0.a(k)+dist0.b(k));
	//	stick *= dist0.b(k)/(dist0.a(k)+dist0.b(k));
	//}
	//weights(K-1) = stick;

	////now loop over all test data and get weighted avg likelihood
	//double loglike = 0.0;
	//for(uint32_t i = 0; i < Nt; i++){
	//	std::vector<double> loglikes;
	//	for (uint32_t k = 0; k < K; k++){
	//		loglikes.push_back(log(weights(k)) + model.getLogPosteriorPredictive(test_data[i], dist0.eta.row(k), dist0.nu(k)));
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
	//return loglike/Nt;
}

template<class Model>
void SDADP<Model>::varDPJob(const std::vector<VXd>& train_data){
	//static int jobNum = 0;
	//int ljn = 0;

	if (train_data.size() == 0){
		return;
	}

	//lock the mutex, get the distribution, unlock
	typename VarDP<Model>::Distribution dist0;
	{
		std::lock_guard<std::mutex> lock(distmut);
	//	ljn = jobNum++;
	//	std::cout << "Starting job " << ljn << std::endl;
		dist0 = dist; // todo:dist是什么
	} //release the lock

	//std::ostringstream oss;
	//oss << "dist0-" << ljn;
	//dist0.save(oss.str().c_str());
	//oss.str(""); oss.clear();


	//do minibatch inference
	//std::cout << "Inference " << ljn << std::endl;
	typename VarDP<Model>::Distribution dist1;
	double starttime = timer.get();
	if(dist0.K == 0){ //if there is no prior to work off of
		VarDP<Model> vdp(train_data, test_data, model, alpha, Knew);
        // 调用了单独的的dp模型
		vdp.run(false);
	 	dist1 = vdp.getDistribution();
	} else { //if there is a prior
		VarDP<Model> vdp(train_data, test_data, dist0, model, alpha, dist0.K+Knew); // 中心结点作为先验
		vdp.run(false);
	 	dist1 = vdp.getDistribution();
	}
	//std::cout << "Done Inference " << ljn << std::endl;
	//oss << "dist1-" << ljn;
	//dist1.save(oss.str().c_str());
	//oss.str(""); oss.clear();



	//remove empty clusters
	//std::cout << "Remove empties " << ljn << std::endl;
	for (uint32_t k = dist0.K; k < dist1.K; k++){
		if (dist1.sumz(k) < 1.0 && k < dist1.K-1){
			dist1.sumz.segment(k, dist1.K-(k+1)) = (dist1.sumz.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.sumz.conservativeResize(dist1.K-1);
			dist1.logp0.segment(k, dist1.K-(k+1)) = (dist1.logp0.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.logp0.conservativeResize(dist1.K-1);
			dist1.nu.segment(k, dist1.K-(k+1)) = (dist1.nu.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.nu.conservativeResize(dist1.K-1);
			dist1.a.segment(k, dist1.K-(k+1)) = (dist1.a.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.a.conservativeResize(dist1.K-1);
			dist1.b.segment(k, dist1.K-(k+1)) = (dist1.b.segment(k+1, dist1.K-(k+1))).eval(); //eval stops aliasing
			dist1.b.conservativeResize(dist1.K-1);
			dist1.eta.block(k, 0, dist1.K-(k+1), dist1.eta.cols()) = (dist1.eta.block(k+1, 0, dist1.K-(k+1), dist1.eta.cols())).eval();
			dist1.eta.conservativeResize(dist1.K-1, Eigen::NoChange);
			dist1.K--;
			k--;
		} else if (dist1.sumz(k) < 1.0){ //just knock off the end
			dist1.sumz.conservativeResize(dist1.K-1);
			dist1.logp0.conservativeResize(dist1.K-1);
			dist1.nu.conservativeResize(dist1.K-1);
			dist1.a.conservativeResize(dist1.K-1);
			dist1.b.conservativeResize(dist1.K-1);
			dist1.eta.conservativeResize(dist1.K-1, Eigen::NoChange);
			dist1.K--;
			k--;
		}
	}
	if(dist1.K == 0){//if removing empty clusters destroyed all of them, just quit
		return;
	}
	//std::cout << "Done Remove empties " << ljn << std::endl;

	//oss << "dist1r-" << ljn;
	//dist1.save(oss.str().c_str());
	//oss.str(""); oss.clear();


	//lock mutex, store the local trace, merge the minibatch distribution, unlock
	//std::cout << "Merging " << ljn << std::endl;
	double mergetime;
	typename VarDP<Model>::Distribution dist2; //dist2 is used to check if a matching was solved later
	{
		std::lock_guard<std::mutex> lock(distmut);
		dist2 = dist;

		//oss << "dist2-" << ljn;
		//dist2.save(oss.str().c_str());
		//oss.str(""); oss.clear();

		//merge
		double t0 = timer.get(); //reuse t0 -- already stored it above
		//std::cout << "Job " << ljn << std::endl;
		dist = mergeDistributions(dist1, dist, dist0); // merge
		mergetime = timer.get()- t0;
	//	std::cout << "Done Merging, saving dist " << ljn << std::endl;
		t0 = timer.get();
		dists.push_back(dist);
		mtrace.starttimes.push_back(starttime);
		mtrace.mergetimes.push_back(mergetime);
		mtrace.times.push_back(t0);
		//mtrace.testlls.push_back(testll);
		//mtrace.clusters.push_back(dist.eta.rows());
		mtrace.clusters.push_back(dist.K);
		if (mtrace.matchings.size() == 0){
			mtrace.matchings.push_back(0); // the first merge never needs to do a matching since all components are new
		} else {
			uint32_t nm = mtrace.matchings.back();
			if (dist1.K > dist0.K && dist2.K > dist0.K){
				mtrace.matchings.push_back(nm+1);
			} else {
				mtrace.matchings.push_back(nm);
			}
		}

	//	std::cout << "Done saving dist " << ljn << std::endl;
		//oss << "distf-" << ljn;
		//dist.save(oss.str().c_str());
		//oss.str(""); oss.clear();
	} //release the lock

	//} //release the lock

	//double t0 = timer.get();
	//double testll = computeTestLogLikelihood(distf);
	//{
	//	std::lock_guard<std::mutex> lock(distmut);
	//	mtrace.starttimes.push_back(starttime);
	//	mtrace.mergetimes.push_back(mergetime);
	//	mtrace.times.push_back(t0);
	//	mtrace.testlls.push_back(testll);
	//	mtrace.clusters.push_back(dist.eta.rows());
	//	if (mtrace.matchings.size() == 0){
	//		mtrace.matchings.push_back(0); // the first merge never needs to do a matching since all components are new
	//	} else {
	//		uint32_t nm = mtrace.matchings.back();
	//		if (dist1.K > dist0.K && dist2.K > dist0.K){
	//			mtrace.matchings.push_back(nm+1);
	//		} else {
	//			mtrace.matchings.push_back(nm);
	//		}
	//	}
	//} //release the lock

	//std::cout << "DONE " << ljn << std::endl;
	//done!
	return;
}


template<class Model>
typename VarDP<Model>::Distribution SDADP<Model>::mergeDistributions(typename VarDP<Model>::Distribution src, typename VarDP<Model>::Distribution dest, typename VarDP<Model>::Distribution prior){
	// 将三个部分merge到一起
    uint32_t Kp = prior.K;
	uint32_t Ks = src.K;
	uint32_t Kd = dest.K;
	uint32_t M = dest.eta.cols();
	typename VarDP<Model>::Distribution out;
	assert(Kd >= Kp && Ks >= Kp);

    // 判断聚类簇数是否相同，若相同直接合并
	if (Ks == Kp){ // dist1 dist0 minibatch
		//no new components created; just do the merge directly
		//match the first Ks elements (one for each src component) to the dest
		out = dest;
		out.eta.block(0, 0, Ks, M) += src.eta - prior.eta;
		out.nu.head(Ks) += src.nu - prior.nu;
		out.sumz.head(Ks) += src.sumz;
		out.logp0.head(Ks) += src.logp0;
		for (uint32_t k = 0; k < Kd; k++){ // 和dest合并
			out.a(k) = 1.0 + out.sumz(k);
			out.b(k) = alpha;
			for (uint32_t j = k+1; j < Kd; j++){
				out.b(k) += out.sumz(j);
			}
		}
	} else if (Kd == Kp) { // dist dist0
		//new components were created in src dist1, but dest is still the same size as prior
		//just do the merge directly from dest into src
		out = src; //dist1 dist
		if (Kp > 0){
			out.eta.block(0, 0, Kd, M) += dest.eta - prior.eta;
			out.nu.head(Kd) += dest.nu - prior.nu;
			out.sumz.head(Kd) += dest.sumz;
			out.logp0.head(Kd) += dest.logp0;
		}
		for (uint32_t k = 0; k < Ks; k++){ // 和src合并
			out.a(k) = 1.0 + out.sumz(k);
			out.b(k) = alpha;
			for (uint32_t j = k+1; j < Ks; j++){
				out.b(k) += out.sumz(j);
			}
		}
	}

    //

    else {
		uint32_t Ksp = Ks-Kp;  // 计算类别数量差 src prior
		uint32_t Kdp = Kd-Kp; // dest prior
		//new components were created in both dest and src -- need to solve a matching
		MXd costs = MXd::Zero(Ksp+Kdp, Ksp+Kdp);
		MXi costsi = MXi::Zero(Ksp+Kdp, Ksp+Kdp); // 维度

		//get logp0 and Enk for d1 and d2
		VXd logp0s = src.logp0.tail(Ksp);
		VXd logp0d = dest.logp0.tail(Kdp);
		VXd Enks = src.sumz.tail(Ksp);
		VXd Enkd = dest.sumz.tail(Kdp);

		//compute costs
		MXd etam = MXd::Zero(1, model.getEta0().size());
		VXd num = VXd::Zero(1);
		VXd loghm = num;
		VXd dlogh_dnum = num;
		MXd dlogh_detam = etam;
		for (uint32_t i = 0; i < Ksp; i++){ //(0,Ksp)(Kdp,Ksp+Kdp)
			//compute costs in the 1-2 block and fill in the 1-0 block
			for (uint32_t j = 0; j < Kdp; j++){
				etam = src.eta.row(Kp+i) + dest.eta.row(Kp+j) - model.getEta0().transpose();
				num(0) = src.nu(Kp+i) + dest.nu(Kp+j) - model.getNu0();
				model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum); // 会改变传进去的值
				costs(i, j) = loghm(0) - log(alpha)*(1.0-exp(logp0s(i)+logp0d(j))) - lgamma(Enks(i)+Enkd(j)); // src dest
			}
			//compute costs in the 1-0 block  src prior
			etam = src.eta.row(Kp+i);
			num(0) = src.nu(Kp+i); // unused in Dir model
			model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
			double c10 = loghm(0) - log(alpha)*(1.0-exp(logp0s(i))) - lgamma(Enks(i)); //src
			for (uint32_t j = Kdp; j < Ksp+Kdp; j++){
				costs(i, j) = c10;
			}
		}

		//compute costs in the 2-0 block  dist2- dist0
		for (uint32_t j = 0; j < Kdp; j++){ //(0,Kdp)(Ksp,Ksp+Kdp)
			etam = dest.eta.row(Kp+j);
			num(0) = dest.nu(Kp+j);
			model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
			double c20 = loghm(0) - log(alpha)*(1.0-exp(logp0d(j))) - lgamma(Enkd(j)); //dest
			for (uint32_t i = Ksp; i < Ksp+Kdp; i++){
				costs(i, j) = c20;
			}
		}

		//the 0-0 block is a constant
		for (uint32_t i = Ksp; i < Ksp+Kdp; i++){   //(Ksp,Ksp+Kdp)(Kdp,Ksp+Kdp)
			for (uint32_t j = Kdp; j < Ksp+Kdp; j++){
				costs(i, j) = model.getLogH0();
			}
		}

		//std::cout << "Cost matrix: " << std::endl << costs << std::endl;

		//now all costs have been computed, and max/min are known
		//subtract off the minimum from everything and remap to integers between 0 and INT_MAX/1000 
		double mincost = costs.minCoeff();
		double maxcost = costs.maxCoeff();
		maxcost -= mincost;
		double fctr = ((double)INT_MAX/1000.0)/maxcost;
		for (uint32_t i = 0; i < Ksp+Kdp; i++){
			for (uint32_t j = 0; j < Ksp+Kdp; j++){
				costsi(i, j) = (int)(fctr*(costs(i, j) - mincost));
			}
		}

		//std::cout << "costsi: " << std::endl << costsi << std::endl;

		std::vector<int> matchings;

        // 计算cost 并使用hungarian算法进行匹配
        // 如文章中调用hungarian计算
		int cost = hungarian(costsi, matchings);


		//std::cout << "matchings: " << std::endl;
		//for (uint32_t i = 0; i < matchings.size(); i++){
		//	std::cout << i << "->" << matchings[i] << std::endl;
		//}

		out = dest;
		//merge the first Kp elements directly (no matchings)
		if (Kp > 0){ //if dest + src both have elements, but their common prior is empty this can happen
			out.eta.block(0, 0, Kp, M) += src.eta.block(0, 0, Kp, M) - prior.eta;
			out.nu.head(Kp) += src.nu.head(Kp) - prior.nu;
			out.sumz.head(Kp) += src.sumz.head(Kp);
			out.logp0.head(Kp) += src.logp0.head(Kp);
		}

		//merge the last Ksp elements using the matchings
        // 合并进原来的K0个

		for (uint32_t i = Kp; i < Ks; i++){
			uint32_t toIdx = Kp+matchings[i-Kp];
			if (toIdx < Kd){
				out.eta.row(toIdx) += src.eta.row(i) - model.getEta0().transpose();
				out.nu(toIdx) += src.nu(i) - model.getNu0();
				out.sumz(toIdx) += src.sumz(i);
				out.logp0(toIdx) += src.logp0(i);
			} else {
				out.eta.conservativeResize(out.K+1, Eigen::NoChange);
				out.nu.conservativeResize(out.K+1);
				out.sumz.conservativeResize(out.K+1);
				out.logp0.conservativeResize(out.K+1);
				out.K++;
				out.eta.row(out.K-1) = src.eta.row(i);
				out.nu(out.K-1) = src.nu(i);
				out.sumz(out.K-1) = src.sumz(i);
				out.logp0(out.K-1) = src.logp0(i);
			}
		}
		out.a.resize(out.K);
		out.b.resize(out.K);

        // 最终再根据一堆超参数更新beta分布的参数
		for (uint32_t k = 0; k < out.K; k++){
			out.a(k) = 1.0 + out.sumz(k);
			out.b(k) = alpha;
			for (uint32_t j = k+1; j < out.K; j++){
				out.b(k) += out.sumz(j);
			}
		}
	}
	return out;
}

#define __SDADP_IMPL_HPP
#endif /* __SDADP_IMPL_HPP */
