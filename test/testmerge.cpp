//
// Created by 黄韧 on 2022/7/16.
//

#include <sdabnp/sda_hdp_mixture>
#include <sdabnp/model/dirichlet>
#include <Eigen/Dense>
#include <random>

typedef Eigen::MatrixXd MXd;
typedef Eigen::VectorXd VXd;

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

int main(int argc, char** argv) {

    uint32_t T = 100;
    uint32_t K = 10;
    uint32_t N = 500;
    uint32_t Nl = 100;
    uint32_t Nt = 100;
    uint32_t Ntl = 100;
    uint32_t Nmini = 100;
    uint32_t D = 2;

    // 读取训练文件

    std::vector<std::vector<VXd> > train_data, test_data;


    readModel("./arxiv_train.dat", train_data, D, N, Nl);
    readModel("./arxiv_test.dat", test_data, D, Nt, Ntl);
    std::cout << "Reading training/test data" << std::endl;



    VXd alpha0 = VXd::Ones(D);
    DirModel dir(alpha0);

    SDAHDP<DirModel> sdahdp (test_data, dir, 1.0, 1.0, K, T, 1);

    // todo : train model to get dist1 dist2;


    VarHDP<DirModel>::VarHDPResults dist1,dist2,dist;
    std::vector<double> testlls;
    if (dist1.T >= dist2.T){
        // 将具有较大聚类簇T的合并到 具有较小聚类簇T的类中;
        std::cout<<dist1.T<<" "<<dist2.T<<std::endl;
        dist = sdahdp.mergeDistributions(dist2,dist1,dist1);
        double testll = sdahdp.computeTestLogLikelihood(dist);
        testlls.push_back(testll);
    }
    else{
        std::cout<<dist1.T<<" "<<dist2.T<<std::endl;
        dist = sdahdp.mergeDistributions(dist1,dist2,dist2);
        double testll = sdahdp.computeTestLogLikelihood(dist);
        testlls.push_back(testll);
    }

    return 0;
}