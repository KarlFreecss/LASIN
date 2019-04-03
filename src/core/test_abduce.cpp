#include "data_patch.hpp"

#include <SWI-cpp.h>
#include <SWI-Prolog.h>
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>

#include <stdlib.h>

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::neighbor;

typedef metric::EuclideanDistance EuDist;
typedef NeighborSearch<NearestNeighborSort, metric::EuclideanDistance> AllkNN;

int main(int argc, char **argv) {
    arma::mat d;
    mlpack::data::Load("../../data/test/5.csv", d, true);
    int n = d.n_cols;    
    sp_mat re(n, n);
    /*
    AllkNN a(d);
    arma::Mat<size_t> resultingNeighbors;
    arma::mat resultingDistances;
    int nn = 3;
    a.Search(nn, resultingNeighbors, resultingDistances);
    
    for (int i = 0; i < d.n_cols; i++)
        for (int j  = 0; j < nn; j++) {
            cout << i << "," << j << ": "; 
            cout << resultingNeighbors(j, i) << endl;
            re(resultingNeighbors(j, i), i) = 1.0;
        }
    */
    kernel::GaussianKernel gknl(0.4);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            vec pi = d.col(i);
            vec pj = d.col(j);
            re(j, i) = gknl.Evaluate(pi, pj);
        }
    sp_mat deg = arma::sum(re, 1);
    sp_mat D = diagmat(deg);
    sp_mat L = D - re;
    L.print("Laplacian: ");
    mat eigvec;
    vec eigval;
    arma::eigs_sym(eigval, eigvec, L, 2, "sa");
    eigval.print("eigval: ");
    eigvec.print("eigvec: ");

    Row<size_t> assignments; // Cluster assignments.
    mat centroids; // Cluster centroids
    kmeans::KMeans<> kmeans;
    kmeans.Cluster(eigvec.t(), 2, assignments, centroids);
    assignments.print("assign: ");

    /*
    DataPatch *data = new DataPatch("../../data/MNIST0.csv");
    cout << "Data loaded\n";
    DataPatch *copy = new DataPatch(*data);
    cout << "Data copied\n";


    sparse_coding::SparseCoding *sc = new sparse_coding::SparseCoding(0, 0.0);
    sc->Atoms() = (size_t) 10;
    sc->Lambda1() = 0.4;
    sc->Lambda2() = 1.0;
    sc->MaxIterations() = (size_t) 1;
    
    cout << "Start training\n";
    sc->Train(*(data->get_features()));
    cout << "End training\n";

    mat codes;
    sc->Encode(*(data->get_features()), codes);
    cout << "End encoding\n";
    data::Save("../../results/Coded_MNIST0.csv", codes, true);
    cout << "Coded data is saved.\n";
    data::DatasetInfo info(codes.n_rows);
    cout << data->get_features()->n_rows << "\n";
    cout << codes.n_rows << "\n";
    DataPatch *coded = new DataPatch(&codes, data->get_labels(), info);
    cout << "Created coded data.\n";
    delete coded;
    delete data;
    */
    return 0;
}
