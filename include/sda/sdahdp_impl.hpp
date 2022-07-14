#ifndef __SDAHDP_IMPL_HPP


template<class Model>
// done: 增加初始化参数 alpha T
SDAHDP<Model>::SDAHDP(const std::vector< std::vector<VXd> >& test_data, const Model& model,double gam, double alpha, uint32_t Knew, uint32_t T, uint32_t nthr):
        test_data(test_data), model(model), gam(gam), alpha(alpha), Knew(Knew), T(T), pool(nthr){
//    test_mxd = MXd::Zero(test_data[0].size(), test_data.size());
//    //需要在计算testlikelihood的时候使用
//    for (uint32_t i =0; i < test_data.size(); i++){
//        for (uint32_t j = 0; j < test_data[i].size(); j++)
//            test_mxd.col(i) = test_data[i][j];
//    }
    dist.T = 0;
    timer.start(); //start the clock -- used for tracking performance
}

template<class Model>
void SDAHDP<Model>::submitMinibatch(const std::vector< std::vector<VXd> >& train_data){
    pool.submit(std::bind(&SDAHDP<Model>::varHDPJob, this, train_data));
}

template<class Model>
void SDAHDP<Model>::waitUntilDone(){
    pool.wait();
}

template<class Model>
typename VarHDP<Model>::VarHDPResults SDAHDP<Model>::getResults(){
    //have to lock/store since the worker pool might be doing stuff with it
    typename VarHDP<Model>::VarHDPResults out;
    {
        std::lock_guard<std::mutex> lock(distmut);
        out = dist;
    }
    return out;
}

template<class Model>
MultiTrace SDAHDP<Model>::getTrace(){
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
double SDAHDP<Model>::computeTestLogLikelihood(typename VarHDP<Model>::VarHDPResults dist0){
    // todo: perplexity的部分

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


    uint32_t T = dist0.T;
    uint32_t Nt = test_mxd.cols();

    if (Nt == 0){
        std::cout << "WARNING: Test Log Likelihood = NaN since Nt = 0" << std::endl;
    }

    //first get average weights -- no compression
    // 需要eta和u v(权重的)
    // todo 是否需要根据每一层去计算后验概率 再加总
    double stick = 1.0;
    VXd weights = VXd::Zero(T);
    for(uint32_t k = 0; k < T-1; k++){
        weights(k) = stick*dist0.u(k)/(dist0.u(k)+dist0.v(k));
        stick *= dist0.v(k)/(dist0.u(k)+dist0.v(k)); // done： a 和 b 的维度要改变
    }
    weights(T-1) = stick;

    // 传入 test_MX eta nu

    MXd logp = model.getLogPosteriorPredictive(test_mxd, dist0.eta, dist0.nu).array().rowwise() + (weights.transpose()).array().log();
    VXd llmaxs = logp.rowwise().maxCoeff();
    logp.colwise() -= llmaxs;
    VXd lls = ((logp.array().exp()).rowwise().sum()).log();
    return (lls + llmaxs).sum()/Nt;
}

template<class Model>
void SDAHDP<Model>::varHDPJob(const std::vector< std::vector<VXd> >& train_data){

    //static int jobNum = 0;
    //int ljn = 0;

    if (train_data.size() == 0){
        return;
    }

    //lock the mutex, get the distribution, unlock
    typename VarHDP<Model>::VarHDPResults dist0;
    {
        std::cout<<"success getting results"<<std::endl;
        std::lock_guard<std::mutex> lock(distmut);
        //	ljn = jobNum++;
        //	std::cout << "Starting job " << ljn << std::endl;
        dist0 = dist;
    } //release the lock

    //std::ostringstream oss;
    //oss << "dist0-" << ljn;
    //dist0.save(oss.str().c_str());
    //oss.str(""); oss.clear();


    //do minibatch inference
    //std::cout << "Inference " << ljn << std::endl;
    typename VarHDP<Model>::VarHDPResults dist1;
    double starttime = timer.get();

    if(dist0.T == 0){ //if there is no prior to work off of
        //VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const Model& model, double gam, double alpha, uint32_t T, uint32_t K);
        VarHDP<Model> vhdp(train_data, test_data, model, gam, alpha, T, Knew);
        vhdp.run(false);
        std::cout<<"success run"<<std::endl;
        dist1 = vhdp.getResults();
        std::cout<<"success getting results without prior"<<std::endl;
    } else { //if there is a prior
        //VarHDP(const std::vector< std::vector<VXd> >& train_data, const std::vector< std::vector<VXd> >& test_data, const Model& model, double gam, double alpha, uint32_t T, uint32_t K);
//        VarHDP<Model> vhdp(train_data, test_data, dist0, model, alpha, dist0.K+Knew);
        VarHDP<Model> vhdp(train_data, test_data, dist0, model, gam, alpha, dist0.T+T, dist0.K+Knew);
        vhdp.run(false);
        std::cout<<"success run"<<std::endl;
        dist1 = vhdp.getResults();
        std::cout<<"success getting results with prior"<<std::endl;
    }
    //std::cout << "Done Inference " << ljn << std::endl;
    //oss << "dist1-" << ljn;
    //dist1.save(oss.str().c_str());
    //oss.str(""); oss.clear();



    //remove empty clusters
//    std::cout << "Remove empties " << ljn << std::endl;
    for (uint32_t k = dist0.T; k < dist1.T; k++){ // 先验和src的差，也就是说判断src多出来的有那个是
//        std::cout<<dist0.T<<dist1.T<<std::endl;
//        std::cout<<dist1.sumz(k)<<std::endl;
//        for (uint32_t i = dist0.K)


        if (dist1.sumz(k) < 1.0 && k < dist1.T-1){ // 判断local的 remove 将第k个元素被取代
            // done: 是否需要加参数 这边是在去除空的cluster 要根据hdp的参参数调用
            // eta nu u v zeta phi a b times objs testlls
            dist1.eta.block(k, 0, dist1.T-(k+1), dist1.eta.cols()) = (dist1.eta.block(k+1, 0, dist1.T-(k+1), dist1.eta.cols())).eval();
            dist1.eta.conservativeResize(dist1.T-1, Eigen::NoChange);
            dist1.nu.segment(k, dist1.T-(k+1)) = (dist1.nu.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
            dist1.nu.conservativeResize(dist1.T-1);

//            dist1.a.segment(k, dist1.T-(k+1)) = (dist1.a.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.a.conservativeResize(dist1.T-1);
//            dist1.b.segment(k, dist1.T-(k+1)) = (dist1.b.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.b.conservativeResize(dist1.T-1);
            // todo bug 对每个餐馆内都要做操作
            dist1.u.segment(k, dist1.T-(k+1)) = (dist1.u.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
            dist1.u.conservativeResize(dist1.T-1);
            dist1.v.segment(k, dist1.T-(k+1)) = (dist1.v.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
            dist1.v.conservativeResize(dist1.T-1);

            dist1.sumz.segment(k, dist1.T-(k+1)) = (dist1.sumz.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
            dist1.sumz.conservativeResize(dist1.T-1);
            dist1.logp0.segment(k, dist1.T-(k+1)) = (dist1.logp0.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
            dist1.logp0.conservativeResize(dist1.T-1);


//            dist1.zeta.segment(k, dist1.T-(k+1)) = (dist1.zeta.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.zeta.conservativeResize(dist1.T-1);
//            dist1.phi.segment(k, dist1.T-(k+1)) = (dist1.phi.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.phi.conservativeResize(dist1.T-1);


            // 怎么去除vector中的不要的元素
//            dist1.times.segment(k, dist1.T-(k+1)) = (dist1.times.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.times.conservativeResize(dist1.K-1);
//            dist1.objs.segment(k, dist1.T-(k+1)) = (dist1.objs.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.objs.conservativeResize(dist1.T-1);
//            dist1.testlls.segment(k, dist1.T-(k+1)) = (dist1.testlls.segment(k+1, dist1.T-(k+1))).eval(); //eval stops aliasing
//            dist1.testlls.conservativeResize(dist1.T-1);

            dist1.T--;
            k--;
        } else if (dist1.sumz(k) < 1.0){ //just knock off the end
            // eta nu u v zeta phi a b times objs testlls
            dist1.sumz.conservativeResize(dist1.T-1);
            dist1.logp0.conservativeResize(dist1.T-1);
            // todo bug
            dist1.eta.conservativeResize(dist1.T-1, Eigen::NoChange);
            dist1.nu.conservativeResize(dist1.T-1);
//            dist1.a.conservativeResize(dist1.T-1);
//            dist1.b.conservativeResize(dist1.T-1);
            dist1.u.conservativeResize(dist1.T-1);
            dist1.v.conservativeResize(dist1.T-1);
//            dist1.zeta.conservativeResize(dist1.T-1);
//            dist1.phi.conservativeResize(dist1.T-1);
//            dist1.times.conservativeResize(dist1.T-1);
//            dist1.objs.conservativeResize(dist1.T-1);
//            dist1.testlls.conservativeResize(dist1.T-1);

            dist1.T--;
            k--;
        }
    }
    if(dist1.T == 0){//if removing empty clusters destroyed all of them, just quit
        return;
    }
//    std::cout << "Done Remove empties " << ljn << std::endl;

    //oss << "dist1r-" << ljn;
    //dist1.save(oss.str().c_str());
    //oss.str(""); oss.clear();


    //lock mutex, store the local trace, merge the minibatch distribution, unlock
    //std::cout << "Merging " << ljn << std::endl;
    double mergetime;
    typename VarHDP<Model>::VarHDPResults dist2; //dist2 is used to check if a matching was solved later
    {
        std::lock_guard<std::mutex> lock(distmut);
        dist2 = dist;

        //oss << "dist2-" << ljn;
        //dist2.save(oss.str().c_str());
        //oss.str(""); oss.clear();

        //merge
        double t0 = timer.get(); //reuse t0 -- already stored it above
        //std::cout << "Job " << ljn << std::endl;
        dist = mergeDistributions(dist1, dist, dist0);
        std::cout<<"complete merge"<<std::endl;
        std::cout<<dist.sumz.transpose()<<std::endl;
        std::cout<<dist.logp0.transpose()<<std::endl;
        mergetime = timer.get()- t0;
        //	std::cout << "Done Merging, saving dist " << ljn << std::endl;
        t0 = timer.get();
        dists.push_back(dist);
        mtrace.starttimes.push_back(starttime);
        mtrace.mergetimes.push_back(mergetime);
        mtrace.times.push_back(t0);
        //mtrace.testlls.push_back(testll);
        //mtrace.clusters.push_back(dist.eta.rows());
        mtrace.clusters.push_back(dist.T);
        if (mtrace.matchings.size() == 0){
            mtrace.matchings.push_back(0); // the first merge never needs to do a matching since all components are new
        } else {
            uint32_t nm = mtrace.matchings.back();
            if (dist1.T > dist0.T && dist2.T > dist0.T){
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
//    double testll = computeTestLogLikelihood(dist);
//    std::cout<<testll<<std::endl;
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
typename VarHDP<Model>::VarHDPResults SDAHDP<Model>::mergeDistributions(typename VarHDP<Model>::VarHDPResults src, typename VarHDP<Model>::VarHDPResults dest, typename VarHDP<Model>::VarHDPResults prior){
    // 将三个部分merge到一起
    // todo: 计算cost的部分需要根据HDP的cost的改动

    uint32_t Kp = prior.T;
    uint32_t Ks = src.T;
    uint32_t Kd = dest.T;
    uint32_t M = dest.eta.cols();
    typename VarHDP<Model>::VarHDPResults out;
    assert(Kd >= Kp && Ks >= Kp);

    // 判断聚类簇数是否相同，若相同直接合并
    // todo: 维度需要改变
    if (Ks == Kp){
        //no new components created; just do the merge directly
        //match the first Ks elements (one for each src component) to the dest
        out = dest;
        out.eta.block(0, 0, Ks, M) += src.eta - prior.eta;
        out.nu.head(Ks) += src.nu - prior.nu;

        // todo 找到全局的sumz和logp0
        out.sumz.head(Ks) += src.sumz;
        out.logp0.head(Ks) += src.logp0;
        for (uint32_t k = 0; k < Kd; k++){
            out.u(k) = 1.0 + out.sumz(k);
            out.v(k) = alpha;
            for (uint32_t j = k+1; j < Kd; j++){
                out.v(k) += out.sumz(j);
            }
        }
    } else if (Kd == Kp) {
        //new components were created in src, but dest is still the same size as prior
        //just do the merge directly from dest into src
        out = src;
        if (Kp > 0){
            out.eta.block(0, 0, Kd, M) += dest.eta - prior.eta;
            out.nu.head(Kd) += dest.nu - prior.nu;
            out.sumz.head(Kd) += dest.sumz;
            out.logp0.head(Kd) += dest.logp0;
        }
        for (uint32_t k = 0; k < Ks; k++){
            out.u(k) = 1.0 + out.sumz(k);
            out.u(k) = alpha;
            for (uint32_t j = k+1; j < Ks; j++){
                out.u(k) += out.sumz(j);
            }
        }
    } else {
        uint32_t Ksp = Ks-Kp;  // 计算类别数量差
        uint32_t Kdp = Kd-Kp;
        //new components were created in both dest and src -- need to solve a matching
        MXd costs = MXd::Zero(Ksp+Kdp, Ksp+Kdp);
        MXi costsi = MXi::Zero(Ksp+Kdp, Ksp+Kdp);

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
        for (uint32_t i = 0; i < Ksp; i++){
            //compute costs in the 1-2 block and fill in the 1-0 block
            for (uint32_t j = 0; j < Kdp; j++){
                etam = src.eta.row(Kp+i) + dest.eta.row(Kp+j) - model.getEta0().transpose();
                num(0) = src.nu(Kp+i) + dest.nu(Kp+j) - model.getNu0();
                model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
                costs(i, j) = loghm(0) - log(alpha)*(1.0-exp(logp0s(i)+logp0d(j))) - lgamma(Enks(i)+Enkd(j));
            }
            //compute costs in the 1-0 block
            etam = src.eta.row(Kp+i);
            num(0) = src.nu(Kp+i);
            model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
            double c10 = loghm(0) - log(alpha)*(1.0-exp(logp0s(i))) - lgamma(Enks(i));
            for (uint32_t j = Kdp; j < Ksp+Kdp; j++){
                costs(i, j) = c10;
            }
        }

        //compute costs in the 2-0 block
        for (uint32_t j = 0; j < Kdp; j++){
            etam = dest.eta.row(Kp+j);
            num(0) = dest.nu(Kp+j);
            model.getLogH(etam, num, loghm, dlogh_detam, dlogh_dnum);
            double c20 = loghm(0) - log(alpha)*(1.0-exp(logp0d(j))) - lgamma(Enkd(j));
            for (uint32_t i = Ksp; i < Ksp+Kdp; i++){
                costs(i, j) = c20;
            }
        }

        //the 0-0 block is a constant
        for (uint32_t i = Ksp; i < Ksp+Kdp; i++){
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
        for (uint32_t i = Kp; i < Ks; i++){
            uint32_t toIdx = Kp+matchings[i-Kp];
            if (toIdx < Kd){
                out.eta.row(toIdx) += src.eta.row(i) - model.getEta0().transpose();
                out.nu(toIdx) += src.nu(i) - model.getNu0();
                out.sumz(toIdx) += src.sumz(i);
                out.logp0(toIdx) += src.logp0(i);
            } else {
                out.eta.conservativeResize(out.T+1, Eigen::NoChange);
                out.nu.conservativeResize(out.T+1);
                out.sumz.conservativeResize(out.T+1);
                out.logp0.conservativeResize(out.T+1);
                out.T++;
                out.eta.row(out.T-1) = src.eta.row(i);
                out.nu(out.T-1) = src.nu(i);
                out.sumz(out.T-1) = src.sumz(i);
                out.logp0(out.T-1) = src.logp0(i);
            }
        }
        out.a.resize(out.T);
        out.b.resize(out.T);
        for (uint32_t k = 0; k < out.T; k++){
            out.u(k) = 1.0 + out.sumz(k);
            out.v(k) = alpha;
            for (uint32_t j = k+1; j < out.T; j++){
                out.v(k) += out.sumz(j);
            }
        }
    }
    return out;
}


#define __SDAHDP_IMPL_HPP
#endif /* __SDAHDP_IMPL_HPP */
