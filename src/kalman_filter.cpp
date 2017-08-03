#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float min_divisor = 0.000001;

  // Ensure that we will not be dividing by zero
  if(fabsf(px) < min_divisor && fabsf(py) < min_divisor) {
    std::cout << "px & py are almost zero. (px,py): " << px << py << std::endl;
    px = min_divisor;
    py = min_divisor;
  } else if(fabsf(px) < min_divisor) {
    std::cout << "px < 0.00001. px = " << px << std::endl;
    px = min_divisor;
  }

  float rho = sqrtf(px*px + py*py);
  float phi = atan2f(py, px);
  float rho_dot =((px*vx + py*vy) / rho);

  VectorXd hx(3);
  hx << rho, phi, rho_dot;

  MatrixXd Ht = H_.transpose();
  VectorXd y = z - hx;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  long size = x_.size();
  MatrixXd I = MatrixXd::Identity(size,size);

  // make sure angle is [0:2pi]
  y[1] -= (2 * M_PI) * floor((y[1] + M_PI) / (2 * M_PI));

  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;
}
