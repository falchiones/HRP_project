#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Структура для представления узла дерева
struct Node {
    size_t id;
    Node* left;
    Node* right;
    Node(size_t i) : id(i), left(nullptr), right(nullptr) {}
};

// Обновление матрицы расстояний после объединения кластеров
void updateMatrix(MatrixXd& distMatrix, size_t clusterId) {
    size_t size = distMatrix.rows();
    VectorXd newDists = distMatrix.col(clusterId);
    distMatrix = 0.5 * (distMatrix + distMatrix.transpose());
    distMatrix.bottomRows(1).noalias() = newDists.transpose();
    distMatrix.rightCols(1).noalias() = newDists;
    distMatrix(size - 1, size - 1) = 0.0;
}

// Иерархическая кластеризация
Node* hierarchicalClustering(MatrixXd& correlationMatrix) {
    size_t size = correlationMatrix.rows();
    MatrixXd distMatrix = sqrt(0.5 * (1.0 - correlationMatrix.array()));

    cout << "Введите матрицу корреляций (" << size << "x" << size << "):" << endl;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            cin >> correlationMatrix(i, j);
        }
    }

    cout << "\nИсходная матрица корреляций:\n" << correlationMatrix << endl;

    Node* root = nullptr;

    while (size > 1) {
        size_t minId1 = 0, minId2 = 0;
        double minDist = numeric_limits<double>::max();

        for (size_t i = 0; i < size; ++i)
            for (size_t j = i + 1; j < size; ++j) {
                double dist = distMatrix(i, j);
                if (dist < minDist) {
                    minDist = dist;
                    minId1 = i;
                    minId2 = j;
                }
            }

        size_t newClusterId = minId1;
        updateMatrix(distMatrix, newClusterId);

        Node* newNode = new Node(size);
        newNode->left = new Node(minId1);
        newNode->right = new Node(minId2);
        root = newNode;

        distMatrix.row(minId1).swap(distMatrix.row(size - 1));
        distMatrix.col(minId1).swap(distMatrix.col(size - 1));
        distMatrix.row(minId2).swap(distMatrix.row(size - 2));
        distMatrix.col(minId2).swap(distMatrix.col(size - 2));
        distMatrix.conservativeResize(size - 1, size - 1);
        size--;
    }

    return root;
}

// Квазидиагонализация матрицы
void quasidiagonalize(MatrixXd& mat) {
    size_t size = mat.rows();
    for (size_t i = 0; i < size; ++i) {
        size_t maxIndex = i;
        for (size_t j = i + 1; j < size; ++j) {
            if (mat(i, j) > mat(i, maxIndex)) {
                maxIndex = j;
            }
        }
        if (maxIndex != i) {
            for (size_t j = 0; j < size; ++j) {
                swap(mat(j, i), mat(j, maxIndex));
            }
        }
        for (size_t k = i + 1; k < size; ++k)
            for (size_t l = i + 1; l < size - 1; ++l)
                if (mat(l, i) < mat(l + 1, i))
                    swap(mat(l, i), mat(l + 1, i));
        for (size_t k = i; k < size - 1; ++k)
            for (size_t l = i; l < size - 1; ++l)
                if (mat(i, l) < mat(i, l + 1))
                    swap(mat(i, l), mat(i, l + 1));
    }
}

// Рекурсивная бисекция
void recursiveBisection(Node* node, const MatrixXd& covMatrix, VectorXd& weights) {
    if (node) {
        if (node->left && node->right) {
            recursiveBisection(node->left, covMatrix, weights);
            recursiveBisection(node->right, covMatrix, weights);
            if (!node->left->left && !node->right->right) {
                size_t clusterId = node->id;
                VectorXd invDiagVar = covMatrix.diagonal().cwiseInverse();
                VectorXd omega = invDiagVar / invDiagVar.sum();
                VectorXd alphas = VectorXd::Zero(weights.size());
                for (size_t i = 0; i < weights.size(); ++i) {
                    double sumV = covMatrix.col(clusterId).sum();
                    alphas(i) = 1.0 - (covMatrix(i, clusterId) / sumV);
                }
                weights = alphas.cwiseProduct(omega);
                cout << "\nВеса активов после применения алгоритма:\n" << weights.transpose() << endl;
            }
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    size_t numItems;
    cout << "Введите количество элементов: ";
    cin >> numItems;

    MatrixXd corrMatrix(numItems, numItems);
    Node* root = hierarchicalClustering(corrMatrix);

    quasidiagonalize(corrMatrix);
    cout << "\nКвазидиагонализированная матрица:\n\n" << corrMatrix << endl;

    VectorXd weights = VectorXd::Ones(corrMatrix.rows());
    recursiveBisection(root, corrMatrix, weights);

    return 0;
}