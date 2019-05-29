#include "memread.hpp"
#include "data_patch.hpp"
#include "utils.hpp"

#include <SWI-cpp.h>
#include <SWI-Prolog.h>
#include <mlpack/core.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>

#define DEBUG_INFO(x) cout << __FILE__ << "_" << __LINE__ << " : " << #x << " = " << x << endl

using namespace std;
using namespace arma;

/* swap int for bresenham algorithm */
inline void swap_int(int *a, int *b);

struct PointI2D{
    int x;
    int y;
    PointI2D(const int _x, const int _y): x(_x), y(_y){}
    PointI2D(): x(0), y(0){}
};

using namespace std;

const int thresh = 127;

int point_in_image(const vector<vector<int>> & img, const PointI2D & A){
    const int x = A.x;
    const int y = A.y;
    return img[x][y] > thresh;
}

int equal_point(const PointI2D & A, const PointI2D & B){
    if (A.x != B.x)
        return false;
    if (A.y != B.y)
        return false;
    return true;
}

int used_image(vector<vector<bool>> & used, PointI2D A, PointI2D B){
    while (equal_point(A, B) == false){
        const int dx = B.x - A.x;
        const int dy = B.y - A.y;
        int ty = 0, tx = 0, mx = 0, my = 0;
    
        if (abs(dx) == abs(dy)){
            mx = 1;
            my = 1;
        } else if (abs(dx) > abs(dy)){
            mx = 1;
            my = 0;
        } else {
            mx = 0;
            my = 1;
        }
        if (dx > 0)
            tx = 1;
        else
            tx = -1;

        if (dy > 0)
            ty = 1;
        else
            ty = -1;

        const int x = A.x;
        const int y = A.y;
        A.x = (x + tx * mx);
        A.y = (y + ty * my);
        used[x + tx * mx][y + ty * my] = true;
    }
    return true;
}

int line_in_image(const vector<vector<int>> & img, PointI2D A, PointI2D B){
    if (point_in_image(img, A) == false or point_in_image(img, B) == false){
        return false;
    }
    while (equal_point(A, B) == false){
        const int dx = B.x - A.x;
        const int dy = B.y - A.y;
        int ty = 0, tx = 0, mx = 0, my = 0;
    
        if (abs(dx) == abs(dy)){
            mx = 1;
            my = 1;
        } else if (abs(dx) > abs(dy)){
            mx = 1;
            my = 0;
        } else {
            mx = 0;
            my = 1;
        }
        if (dx > 0)
            tx = 1;
        else
            tx = -1;

        if (dy > 0)
            ty = 1;
        else
            ty = -1;

        const int x = A.x;
        const int y = A.y;
        A.x = (x + tx * mx);
        A.y = (y + ty * my);
        if (point_in_image(img, A) == false){
            return false;
        }
    }
    return point_in_image(img, A);
}

double distance(const PointI2D &A, const PointI2D &B){
    const int dx = A.x - B.x;
    const int dy = A.y - B.y;
    return dx * dx + dy * dy;
}

vector<pair<PointI2D, PointI2D>> get_strokes(const vector<vector<int>> & img, const vector<PointI2D> & pointList) {
    vector<vector<bool>> used(img.size(), vector<bool>(img[0].size(), false));
    vector<pair<PointI2D, PointI2D>> ret;
    for (int i = 0; i < pointList.size(); ++i){
        const int x = pointList[i].x;
        const int y = pointList[i].y;
        if (used[x][y]){
            continue;
        }
        used[x][y] = true;
        double max_dist = 0;
        int max_dist_index = -1;
        for (int j = i + 1; j < pointList.size(); ++j){
            if (line_in_image(img, pointList[i], pointList[j])){
                const double points_dist = distance(pointList[i], pointList[j]);
                if (points_dist > max_dist){
                    max_dist = points_dist;
                    max_dist_index = j;
                }
            }
        }
        if (max_dist_index != -1) {
            const int jx = pointList[max_dist_index].x;
            const int jy = pointList[max_dist_index].y;
            used[jx][jy] = true;
            used_image(used, pointList[i], pointList[max_dist_index]);
            ret.push_back(make_pair(pointList[i], pointList[max_dist_index]));
        }
    }
    return ret;
}

PREDICATE(clw_get_strokes, 3){
    char *add = (char*) A1;
    vec *data = str2ptr<vec>(add);
    mat ori_img(*data);
    ori_img.reshape(28, 28);

    vector<vector<int>> img(28);
    for (int i = 0; i < 28; ++i){
        for (int j = 0; j < 28; ++j){
            img[i].push_back((int)(ori_img(i, j) * 255));
        }
    }
    vector<vector<int>> points = list2vecvec<int>(A2, -1, 2);
    vector<PointI2D> pointList;
    for (int i = 0; i < points.size(); ++i){
        pointList.push_back(PointI2D(points[i][0], points[i][1]));
    }
    vector<pair<PointI2D, PointI2D>> strokes = get_strokes(img, pointList);
    vector<vector<long>> ret;
    vector<long> tmp;
    for (int i = 0; i < strokes.size(); ++i){
        tmp.clear();
        tmp.push_back(strokes[i].first.x);
        tmp.push_back(strokes[i].first.y);
        ret.push_back(tmp);
        tmp.clear();
        tmp.push_back(strokes[i].second.x);
        tmp.push_back(strokes[i].second.y);
        ret.push_back(tmp);
    }
    cout << ret.size() << ' ' << ret[0].size() << endl;
    return A3 = vecvec2list<long>(ret);  
}

/* mnist_create_mask(Centers, Neighbor_size, Mask)
 * Centers: a group of 2d positions, [[x1, y1], [x2, y2], ...]
 * Neighbor_size: size of neighborhood
 * Mask: an arma::mat(784 by 1), each element is 1 or 0
 */
PREDICATE(mnist_create_mask, 3) {
    vector<vector<double>> points = list2vecvec<double>(A1, -1, 2);
    mat mask(28, 28);
    mask.zeros(); // set mask to all zero
    int ns = (int) A2;
    for (auto pt = points.begin(); pt != points.end(); ++pt) {
        int x = ((vector<double>) *pt)[0];
        int y = ((vector<double>) *pt)[1];
        // left up most point
        int lu_x = x - ns > 0 ? x - ns : 0;
        int lu_y = y - ns > 0 ? y - ns : 0;
        // right dowm most point
        int rd_x = x + ns < 27 ? x + ns : 27;
        int rd_y = y + ns < 27 ? y + ns : 27;
        for (int xx = lu_x; xx <= rd_x; xx++)
            for (int yy = lu_y; yy <= rd_y; yy++)
                mask(xx, yy) = 1.0;
    }
    mask.reshape(784, 1);
    Col<double> *re = new Col<double>(mask.col(0));
    string addr = ptr2str(re);
    return A3 = (char *) addr.c_str();
}

/* ink_(Mat, [X1, Y1]).
 * determine whether there is ink between P1 and P2
 */
PREDICATE(ink_, 2) {
    char *add = (char*) A1;
    vec *data = str2ptr<vec>(add);
    mat img(*data);
    img.reshape(28, 28);
    vector<int> p = list2vec<int>(A2, 2);
    int x = p[0];
    int y = p[1];
    return img(x, y) >= 0.5;
}

/* line_points(START, END, POINTS)
 * use bresenham algorithm to get points between two points
 */
PREDICATE(line_points, 3) {
    vector<int> p1 = list2vec<int>(A1);
    vector<int> p2 = list2vec<int>(A2);
    
    vector<vector<long>> points;

    int x1 = p1[0],
        y1 = p1[1],
        x2 = p2[0],
        y2 = p2[1];
	int dx = abs(x2 - x1),
		dy = abs(y2 - y1),
		yy = 0;
	if (dx < dy) {
		yy = 1;
		swap_int(&x1, &y1);
		swap_int(&x2, &y2);
		swap_int(&dx, &dy);
	}
	int ix = (x2 - x1) > 0 ? 1 : -1,
        iy = (y2 - y1) > 0 ? 1 : -1,
        cx = x1,
        cy = y1,
        n2dy = dy * 2,
        n2dydx = (dy - dx) * 2,
        d = dy * 2 - dx;
	if (yy) { 
		while (cx != x2) {
			if (d < 0) {
				d += n2dy;
			} else {
				cy += iy;
				d += n2dydx;
			}
            points.push_back(vector<long>({(long) cy, (long) cx}));
			cx += ix;
		}
	} else { 
		while (cx != x2) {
			if (d < 0) {
				d += n2dy;
			} else {
				cy += iy;
				d += n2dydx;
			}
            points.push_back(vector<long>({(long) cx, (long) cy}));
			cx += ix;
		}
	}
    return A3 = vecvec2list<long>(points);
}

/*
PREDICATE(above_of, 2) {
    return FALSE;
}

PREDICATE(left_of, 2) {
    return FALSE;
}
*/


inline void swap_int(int *a, int *b){
    int t = *a;
    *a = *b;
    *b = t;
}
//inline void swap_int(int *a, int *b) {
//	*a ^= *b;
//	*b ^= *a;
//	*a ^= *b;
//}
