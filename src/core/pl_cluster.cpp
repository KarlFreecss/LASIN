#include "memread.hpp"
#include "data_patch.hpp"
#include "utils.hpp"

#include <SWI-cpp.h>
#include <SWI-Prolog.h>
#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>

using namespace std;
using namespace arma;
using namespace mlpack;

typedef metric::EuclideanDistance EuDist;
typedef neighbor::NeighborSearch<neighbor::NearestNeighborSort, EuDist> AllkNN;

/* declarations of useful functions*/

/* create similarity graph for spectral clustering
 * @data: data matrix, column-wise stored
 * @type: 1) epsilon-neighbor, 2) nearest neighbor,
 * 3) full connected (Gaussian kernel)
 * @param: 1) epsilon, 2) k of NN, 3) Gaussian kernel bandwidth
 * @returned: similarity graph
 */
sp_mat sim_graph(mat data, int type, double param);

/* reassign data points into lists of groups according to clustering results
 * @points: data points (vector of feature vectors)
 * @group_num: number of groups
 * @assignments: clustering assignments of data points
 * @returned: a Prolog term, list of groups of points. L = [[Grp1], ...],
 *    Grp = [Pt1, ...], Pt = [x1, x2, ...].
 */
PlTerm group2lists(vector<vector<double>> points, int group_num, Row<size_t> assignments);

/* kmeans(List_of_Points, Num_Clusters, Groups)
 * @List_of_points: [[Point1], [Point2], ...] list of list
 * @Num_clusters: number of clusters
 * @Groups: [[Grp1], [Grp2], ...], Grp = [[P1], [P2], ...].
 */
PREDICATE(kmeans, 3) {
    vector<vector<double>> points = list2vecvec<double>(A1);
    int num_clusters = (int) A2;
    size_t nrows = points[0].size();
    size_t ncols = points.size();
    mat data(nrows, ncols);
    // put points into data
    for (size_t i = 0; i < ncols; i++)
        data.col(i) = Col<double>(points[i]);
    Row<size_t> assignments; // Cluster assignments.
    mat centroids; // Cluster centroids
    kmeans::KMeans<> kmeans;
    kmeans.Cluster(data, num_clusters, assignments, centroids);
    // make new term for groups
    return A3 = group2lists(points, num_clusters, assignments);
}

/* spectral_cluster(List_of_Points, SimType, Param, Cluster_num, Groups)
 * compute spectral coordintate for a set of points, using it with kmeans to
 * perform spectral clustering.
 * @List_of_points: [[Point1], [Point2], ...] list of list
 * @SimType: 1 - epsilon, 2 - nearest neighbors, 3 - gaussian kernel
 * @Param: epsilon/neighbor num/gaussian bandwidth
 * @Cluster_num: number of clusters
 * @Groups: output groups
 */
PREDICATE(spectral_cluster, 5) {
    // get data
    vector<vector<double>> points = list2vecvec<double>(A1);
    size_t nrows = points[0].size();
    size_t ncols = points.size();
    mat data(nrows, ncols);
    // put points into data
    for (size_t i = 0; i < ncols; i++)
        data.col(i) = Col<double>(points[i]);
    // build similarity graph
    int sim_type = (int) A2;
    double param = (double) A3;
    sp_mat sgraph = sim_graph(data, sim_type, param);
    // compute Laplacian and spectral value
    int num_clusters = (int) A4;
    sp_mat deg = arma::sum(sgraph, 1); 
    sp_mat D = diagmat(deg); // degree matrix
    sp_mat L = D - sgraph; // Laplacian
    // eigen decompose
    mat eigvec;
    vec eigval;
    // L is semi positive definit, smallest eigen value >= 0, use
    //     "small absolute" option to get rid of an libarmadillo 6.7.6
    //     bug (caused by "sm" option)
    if (!arma::eigs_sym(eigval, eigvec, L, num_clusters, "sa"))
        return FALSE;
    // kmeans clustering on eigen vectors
    Row<size_t> assignments; // Cluster assignments.
    mat centroids; // Cluster centroids
    kmeans::KMeans<> kmeans;
    kmeans.Cluster(eigvec.t(), num_clusters, assignments, centroids);
    return A5 = group2lists(points, num_clusters, assignments);
}


/* implementations */
sp_mat sim_graph(mat data, int type, double param) {
    int n = data.n_cols;
    sp_mat re(n, n);
    switch (type) {
    case 1: // epsilon-neighbor
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                vec pi = data.col(i);
                vec pj = data.col(j);
                if ((i != j) && EuDist::Evaluate(pi, pj) <= param)
                    re(j, i) = 1.0;
            }
    };
    case 2: // nearest neighbor
    {
        AllkNN nn(data);
        Mat<size_t> nbrs;
        mat dists;
        nn.Search((size_t) param, nbrs, dists);
        for (int i = 0; i < n; i++) {
            for (int j  = 0; j < (int) param; j++) {
                re(nbrs(j, i), j) = 1.0;
            }
        }
    };
    default: // fully connected (gaussian kernel)
    {
        kernel::GaussianKernel gknl(param);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                if (i == j)
                    continue;
                vec pi = data.col(i);
                vec pj = data.col(j);
                re(j, i) = gknl.Evaluate(pi, pj);
            }
    };
    }
    return re;
}

PlTerm group2lists(vector<vector<double>> points, int group_num, Row<size_t> assignments) {
    // assign points to groups
    vector<vector<vector<double>>> groups; // groups<points<features>>
    for (int i = 0; i < group_num; i++)
        groups.push_back(vector<vector<double>>());
    for (size_t i = 0; i < assignments.n_cols; i++) {
        size_t cls = assignments(i);
        groups[cls].push_back(points[i]);
    }
    // make new term for groups
    term_t groups_ref = PL_new_term_ref();
    PlTerm groups_term(groups_ref);
    PlTail groups_tail(groups_term);
    for (int grp = 0; grp < group_num; grp++) {
        term_t points_ref = PL_new_term_ref();
        PlTerm points_term(points_ref);
        PlTail points_tail(points_term);
        size_t points_num = groups[grp].size();
        for (size_t pt = 0; pt < points_num; pt++) {
            PlTerm point = vec2list<double>(groups[grp][pt]);
            points_tail.append(point);
        }
        points_tail.close();
        groups_tail.append(points_term);
    }
    groups_tail.close();
    return groups_term;    
}
