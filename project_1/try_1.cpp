#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

struct ClusterNode {
    size_t id;
    ClusterNode* leftChild;
    ClusterNode* rightChild;
    ClusterNode(size_t i) : id(i), leftChild(nullptr), rightChild(nullptr) {}
};

namespace ClusterAlgorithm {
    // Обновление матрицы расстояний после объединения кластеров
    void updateDistanceMatrix(MatrixXd& distMat, size_t newClusterFrontIndex) {
        size_t N = distMat.rows();
        VectorXd newDists = distMat.col(newClusterFrontIndex);
        distMat = 0.5 * (distMat + distMat.transpose());
        distMat.bottomRows(1).noalias() = newDists.transpose();
        distMat.rightCols(1).noalias() = newDists;
        distMat(N - 1, N - 1) = 0.0;
    }

    // Иерархическая кластеризация
    void perform(MatrixXd& corrMat, ClusterNode*& root) {
        size_t N = corrMat.rows();
        MatrixXd distMat = sqrt(0.5 * (1.0 - corrMat.array()));
        while (N > 1) {
            size_t minI = 0, minJ = 0;
            double minDist = numeric_limits<double>::max();
            for (size_t i = 0; i < N; ++i)
                for (size_t j = i + 1; j < N; ++j) {
                    double dist = distMat(i, j);
                    if (dist < minDist) {
                        minDist = dist;
                        minI = i;
                        minJ = j;
                    }
                }
            size_t newClusterFrontIndex = minI;
            updateDistanceMatrix(distMat, newClusterFrontIndex);
            ClusterNode* newNode = new ClusterNode(N);
            newNode->leftChild = new ClusterNode(minI);
            newNode->rightChild = new ClusterNode(minJ);
            root = newNode;
            distMat.row(minI).swap(distMat.row(N - 1));
            distMat.col(minI).swap(distMat.col(N - 1));
            distMat.row(minJ).swap(distMat.row(N - 2));
            distMat.col(minJ).swap(distMat.col(N - 2));
            distMat.conservativeResize(N - 1, N - 1);
            N--;
        }
    }
}

namespace MatrixTransformation {
    // Квазидиагонализация матрицы
    void perform(MatrixXd& mat) {
        size_t N = mat.rows();
        for (size_t i = 0; i < N; ++i) {
            size_t maxIndex = i;
            for (size_t j = i + 1; j < N; ++j) {
                if (mat(i, j) > mat(i, maxIndex)) {
                    maxIndex = j;
                }
            }
            if (maxIndex != i) {
                for (size_t j = 0; j < N; ++j) {
                    swap(mat(j, i), mat(j, maxIndex));
                }
            }
            for (size_t k = i + 1; k < N; ++k)
                for (size_t l = i + 1; k < N - 1; ++l)
                    if (mat(l, i) < mat(l + 1, i))
                        swap(mat(l, i), mat(l + 1, i));
            for (size_t k = i; k < N - 1; ++k)
                for (size_t l = i; l < N - 1; ++l)
                    if (mat(i, l) < mat(i, l + 1))
                        swap(mat(i, l), mat(i, l + 1));
        }
    }
}

namespace RecursivePartition {
    // Рекурсивная бисекция
    void perform(ClusterNode* node, const MatrixXd& covMatrix, VectorXd& assetWeights) {
        if (node) {
            if (node->leftChild && node->rightChild) {
                perform(node->leftChild, covMatrix, assetWeights);
                perform(node->rightChild, covMatrix, assetWeights);
                if (!node->leftChild->leftChild && !node->rightChild->rightChild) {
                    size_t clusterIndex = node->id;
                    VectorXd invDiagVar = covMatrix.diagonal().cwiseInverse();
                    VectorXd omega = invDiagVar / invDiagVar.sum();
                    VectorXd alphas = VectorXd::Zero(assetWeights.size());
                    for (size_t i = 0; i < assetWeights.size(); ++i) {
                        double sumV = covMatrix.col(clusterIndex).sum();
                        alphas(i) = 1.0 - (covMatrix(i, clusterIndex) / sumV);
                    }
                    assetWeights = alphas.cwiseProduct(omega);
                    cout << "\nПолучены веса активов:\n" << assetWeights.transpose() << endl;
                }
            }
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "Введите количество активов: ";
    int numAssets;
    cin >> numAssets;

    MatrixXd returns(numAssets, numAssets);

    cout << "Введите матрицу доходности (" << numAssets << "x" << numAssets << "):" << endl;
    for (int i = 0; i < numAssets; ++i) {
        for (int j = 0; j < numAssets; ++j) {
            cin >> returns(i, j);
        }
    }

    ClusterNode* root = nullptr;

    // Иерархическая кластеризация
    ClusterAlgorithm::perform(returns, root);

    // Квазидиагонализация матрицы
    MatrixTransformation::perform(returns);
    cout << "\nПолучена квазидиагональная матрица:\n\n" << returns << endl;

    VectorXd assetWeights = VectorXd::Ones(returns.rows());

    // Рекурсивная бисекция
    RecursivePartition::perform(root, returns, assetWeights);

    return 0;
}
