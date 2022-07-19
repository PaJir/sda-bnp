//
// Created by 黄韧 on 2022/6/28.
//

#include <sda_dp_mixture>
#include <sda_hdp_mixture>
#include <model/normal_inverse_wishart>
#include <Eigen/Dense>
#include <random>

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

void randomModel(std::vector<std::vector<VXd> >& train_data, std::vector<std::vector<VXd> >& test_data, const uint32_t K, const uint32_t D, const uint32_t N, const uint32_t Nl,
                 const uint32_t Nt, const uint32_t Ntl) {
    std::mt19937 rng;
    std::random_device rd;
    rng.seed(rd());
    std::uniform_real_distribution<> unir;

    // setup the generating model
    std::vector<VXd> mus;
    std::vector<MXd> sigs;
    std::vector<MXd> sigsqrts;
    std::vector<double> pis;

    double sumpis = 0.0;
    std::cout << "Creating generative model..." << std::endl;
    for (uint32_t k = 0; k < K; k++) {
        mus.push_back(VXd::Zero(D));
        sigs.push_back(MXd::Zero(D, D));
        for (uint32_t d = 0; d < D; d++) {
            mus.back()(d) = 100.0 * unir(rng) - 50.0;
            for (uint32_t f = 0; f < D; f++) {
                sigs.back()(d, f) = 5.0 * unir(rng);
            }
        }
        sigs.back() = (sigs.back().transpose() * sigs.back()).eval();  // eval to stop aliasing
        sigsqrts.push_back(Eigen::LLT<MXd, Eigen::Upper>(sigs.back()).matrixL());
        pis.push_back(unir(rng));
        sumpis += pis.back();
        // std::cout << "Mu: " << mus.back().transpose() << std::endl << "Sig: " << sigs.back() << std::endl << "Wt: " << pis.back() << std::endl;
    }
    for (uint32_t k = 0; k < K; k++) {
        pis[k] /= sumpis;
    }

    // output the generating model
    std::ofstream mout("model.log");
    for (uint32_t k = 0; k < K; k++) {
        mout << mus[k].transpose() << " ";
        for (uint32_t j = 0; j < D; j++) {
            mout << sigs[k].row(j) << " ";
        }
        mout << pis[k] << std::endl;
    }
    mout.close();

    std::normal_distribution<> nrm;
    std::discrete_distribution<> disc(pis.begin(), pis.end());
    std::ofstream trout("train.log");
    std::ofstream teout("test.log");
    std::cout << "Sampling training/test data" << std::endl;

    // 下面两个循环多了两层
    for (uint32_t i = 0; i < N; i++) {
        train_data.push_back(std::vector<VXd>());
        for (uint32_t j = 0; j < Nl; j++) {
            VXd x = VXd::Zero(D);
            for (uint32_t m = 0; m < D; m++) {
                x(m) = nrm(rng);
            }
            uint32_t k = disc(rng);
            train_data.back().push_back(mus[k] + sigsqrts[k] * x);
            trout << train_data.back().back().transpose() << std::endl;
        }
        // std::cout << train_data.back().transpose() << std::endl;
    }
    for (uint32_t i = 0; i < Nt; i++) {
        test_data.push_back(std::vector<VXd>());
        for (uint32_t j = 0; j < Ntl; j++) {
            VXd x = VXd::Zero(D);
            for (uint32_t m = 0; m < D; m++) {
                x(m) = nrm(rng);
            }
            uint32_t k = disc(rng);
            test_data.back().push_back(mus[k] + sigsqrts[k] * x);
            teout << test_data.back().back().transpose() << std::endl;
        }
        // std::cout << test_data.back().transpose() << std::endl;
    }
    trout.close();
    teout.close();
}

void readModel(std::string data_name, std::vector<std::vector<VXd> >& data, const uint32_t D, const uint32_t N, const uint32_t Nl) {
    std::ifstream ifs(data_name);
    std::string buffer;
    // N * Nl 行，每行D个文档
    for (uint32_t i = 0; i < N; i++) {
        data.push_back(std::vector<VXd>());
        for (uint32_t j = 0; j < Nl; j++) {
            VXd x = VXd::Zero(D);
            getline(ifs, buffer);
            char* p = strtok((char*)buffer.c_str(), " ");
            double tmp;
            for (uint32_t m = 0; m < D; m++) {
                tmp = atof(p);
                x(m) = tmp;
                p = strtok(NULL, " ");
            }
            data.back().push_back(x);
        }
    }
    ifs.close();
}

void readArxiv(std::string data_name, std::vector<std::vector<VXd> >& data, uint32_t& N, const uint32_t& Nl) {
    const int size_vocab = 12388;
    int n = 0; // train data length
    int Nli = Nl;
    int length, count, word;
    FILE* fileptr = fopen(data_name.c_str(), "r");
    while(fscanf(fileptr, "%10d", &length) != EOF) {
        if (Nli == Nl) {
            Nli = 0;
            data.push_back(std::vector<VXd>(Nl, VXd::Zero(size_vocab))); // size: Nl, defalut value: Zero(size_vocab)
            n++;
        }
        for (int i = 0; i < length; i++) {
            fscanf(fileptr, "%10d:%10d", &word, &count);
            data.back()[Nli](word) = count; // update count
        }
        Nli++;
    }
    N = n;
    fclose(fileptr);
}

int main(int argc, char** argv) {
    // constants
    uint32_t T = 100;
    uint32_t K = 10;
    uint32_t N; // = 500;
    uint32_t Nl = 100;
    uint32_t Nt; // = 100;
    uint32_t Ntl = 100;
    uint32_t Nmini = 100;
    uint32_t D = 2;

    // sample from the model
    std::vector<std::vector<VXd> > train_data, test_data;

    // randomModel(train_data, test_data, K, D, N, Nl, Nt, Ntl);
    // readModel("train.log", train_data, D, N, Nl);
    // readModel("test.log", test_data, D, Nt, Ntl);
    readArxiv("data/arxiv_train.dat", train_data, N, Nl);
    readArxiv("data/arxiv_test.dat", test_data, Nt, Ntl);

    VXd mu0 = VXd::Zero(D);
    MXd psi0 = MXd::Identity(D, D);
    double kappa0 = 1e-6;
    double xi0 = D + 2;
    NIWModel niw(mu0, kappa0, psi0, xi0);

    std::cout << "Running VarHDP..." << std::endl;

    SDAHDP<NIWModel> sdahdp(test_data, niw, 1.0, 1.0, K, T, 8);

    uint32_t Nctr = 0;
    while (Nctr < N) {
        std::cout << Nctr << std::endl;
        std::vector<std::vector<VXd> > minibatch;
        minibatch.insert(minibatch.begin(), train_data.begin() + Nctr, train_data.begin() + Nctr + Nmini);
        sdahdp.submitMinibatch(minibatch);
        Nctr += Nmini;
        std::cout << Nctr << std::endl;
    }
    sdahdp.waitUntilDone();
    VarHDP<NIWModel>::VarHDPResults res = sdahdp.getResults();
    res.save("sdahdpmix");

    return 0;
}
