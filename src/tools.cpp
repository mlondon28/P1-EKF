#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
    *
    * rsme = sqrt(1/n * sum(x_est - x_true)^2)
  */

    //init rmse vector
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check if vectors are the same size
    if((estimations.size() == 0) || (estimations.size() != ground_truth.size())){
        cout << "Error! Estimations vector incorrect size" << endl;
        cout << "estimations.size(): " << estimations.size() << endl;
        cout << "ground_truth.size(): " << ground_truth.size() << endl;
        return rmse;
    }

    // accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];

        // coeff-wise mult
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    // calculate the mean
    rmse = rmse / estimations.size();

    // calc sq root
    rmse = rmse.array().sqrt();

    cout << "rmse: " << rmse << endl;
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */

    MatrixXd Hj(3,4);
    //recover state params
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    // precompute
    float c1 = px*px + py*py;
    float c2 = sqrtf(c1);
    float c3 = (c1*c2);

    // check for division by zero
    if(fabs(c1) < 0.0001){
        cout << "CalculateJacobian() - Error - Division by zero" << endl;
        return Hj;
    }

    // compute jacobian matrix
    Hj <<   (px/c2), (py/c2), 0, 0,
            -(py/c1), (px/c1), 0, 0,
            py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

    return Hj;
}

