//
// Created by 黄韧 on 2022/6/28.
//

#include <sdabnp/sda_dp_mixture>
#include <sdabnp/sda_hdp_mixture>
#include <sdabnp/model/normal_inverse_wishart>
#include <Eigen/Dense>
#include <random>

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

int main(int argc, char** argv){
    //constants
    uint32_t T = 100;
    uint32_t K = 10;
    uint32_t N = 100;
    uint32_t Nl = 100;
    uint32_t Nt = 100;
    uint32_t Ntl = 100;
    uint32_t Nmini = 100;
    uint32_t D = 2;

    std::mt19937 rng;
    std::random_device rd;
    rng.seed(rd());
    std::uniform_real_distribution<> unir;


    //setup the generating model
    std::vector<VXd> mus;
    std::vector<MXd> sigs;
    std::vector<MXd> sigsqrts;
    std::vector<double> pis;
    double sumpis = 0.0;
    std::cout << "Creating generative model..." << std::endl;
    for (uint32_t k = 0; k < K; k++){
        mus.push_back(VXd::Zero(D));
        sigs.push_back(MXd::Zero(D, D));
        for(uint32_t d = 0; d < D; d++){
            mus.back()(d) = 100.0*unir(rng)-50.0;
            for(uint32_t f = 0; f < D; f++){
                sigs.back()(d, f) = 5.0*unir(rng);
            }
        }
        sigs.back() = (sigs.back().transpose()*sigs.back()).eval();//eval to stop aliasing
        sigsqrts.push_back(Eigen::LLT<MXd, Eigen::Upper>(sigs.back()).matrixL());
        pis.push_back(unir(rng));
        sumpis += pis.back();
        //std::cout << "Mu: " << mus.back().transpose() << std::endl << "Sig: " << sigs.back() << std::endl << "Wt: " << pis.back() << std::endl;
    }
    for (uint32_t k = 0; k < K; k++){
        pis[k] /= sumpis;
    }

    //output the generating model
    std::ofstream mout("model.log");
    for (uint32_t k = 0; k < K; k++){
        mout << mus[k].transpose() << " ";
        for (uint32_t j = 0; j < D; j++){
            mout << sigs[k].row(j) << " ";
        }
        mout << pis[k] << std::endl;
    }
    mout.close();



    //sample from the model
    std::vector< std::vector<VXd> > train_data, test_data;
    std::normal_distribution<> nrm;
    std::discrete_distribution<> disc(pis.begin(), pis.end());
    std::ofstream trout("train.log");
    std::ofstream teout("test.log");
    std::cout << "Sampling training/test data" << std::endl;

    // 下面两个循环多了两层
    for (uint32_t i = 0; i < N; i++){
        train_data.push_back(std::vector<VXd>());
        for (uint32_t j = 0; j < Nl; j++){
            VXd x = VXd::Zero(D);
            for (uint32_t m = 0; m < D; m++){
                x(m) = nrm(rng);
            }
            uint32_t k = disc(rng);
            train_data.back().push_back(mus[k] + sigsqrts[k]*x);
            trout << train_data.back().back().transpose() << std::endl;
        }
        //std::cout << train_data.back().transpose() << std::endl;
    }
    for (uint32_t i = 0; i < Nt; i++){
        test_data.push_back(std::vector<VXd>());
        for(uint32_t j = 0; j < Ntl; j++){
            VXd x = VXd::Zero(D);
            for (uint32_t m = 0; m < D; m++){
                x(m) = nrm(rng);
            }
            uint32_t k = disc(rng);
            test_data.back().push_back(mus[k] + sigsqrts[k]*x);
            teout << train_data.back().back().transpose() << std::endl;
        }
        //std::cout << test_data.back().transpose() << std::endl;
    }
    trout.close();
    teout.close();


    VXd mu0 = VXd::Zero(D);
    MXd psi0 = MXd::Identity(D, D);
    double kappa0 = 1e-6;
    double xi0 = D+2;
    NIWModel niw(mu0, kappa0, psi0, xi0);

    std::cout << "Running VarHDP..." << std::endl;

    SDAHDP<NIWModel> sdahdp (test_data, niw,1.0, 1.0, K,T, 8);

    uint32_t Nctr = 0;
    while(Nctr < N){
        std::cout<<Nctr<<std::endl;
        std::vector< std::vector<VXd> > minibatch;
        minibatch.insert(minibatch.begin(), train_data.begin()+Nctr, train_data.begin()+Nctr+Nmini);
        sdahdp.submitMinibatch(minibatch);
        Nctr += Nmini;
        std::cout<<Nctr<<std::endl;
    }
    sdahdp.waitUntilDone();
    VarHDP<NIWModel>::VarHDPResults res = sdahdp.getResults();
    res.save("sdahdpmix");

    return 0;
}

