#include "liw/eskfEstimator.h"

eskfEstimator::eskfEstimator() {
  noise = Eigen::Matrix<double, 12, 12>::Zero();
  delta_state = Eigen::Matrix<double, 17, 1>::Zero();
  covariance = Eigen::Matrix<double, 17, 17>::Identity();

  p = Eigen::Vector3d::Zero();
  q = Eigen::Quaterniond::Identity();
  v = Eigen::Vector3d::Zero();
  ba = Eigen::Vector3d::Zero();
  bg = Eigen::Vector3d::Zero();
  g << 0.0, 0.0, 9.81;

  mean_gyr = Eigen::Vector3d(0, 0, 0);
  mean_acc = Eigen::Vector3d(0, 0, 9.81);

  is_first_imu_meas = true;
  num_init_meas = 1;
}

void eskfEstimator::setAccCov(double para) {
  acc_cov_scale << para, para, para;
}

void eskfEstimator::setGyrCov(double para) {
  gyr_cov_scale << para, para, para;
}

void eskfEstimator::setBiasAccCov(double para) {
  b_acc_cov << para, para, para;
}

void eskfEstimator::setBiasGyrCov(double para) {
  b_gyr_cov << para, para, para;
}

/**
 * 尝试初始化ESKF估计器
 * 使用静止状态下的IMU测量数据估计初始偏置和重力向量
 * @param imu_meas IMU测量数据向量：每个元素包含时间戳、加速度计和陀螺仪测量值
 */
void eskfEstimator::tryInit(
  const std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>& imu_meas) {

// === 执行IMU数据初始化处理 ===
// 计算IMU测量数据的统计信息（均值、协方差等）
initialization(imu_meas);

// === 检查初始化条件 ===
// 条件1：测量数量足够（超过最小初始化计数）
// 条件2：时间窗口足够长（超过最小初始化时间）
if (num_init_meas > MIN_INI_COUNT && imu_meas.back().first - time_first_imu > MIN_INI_TIME) {
  
  // === 调整加速度计协方差 ===
  // 根据重力加速度的理论值和实际测量值的比值来调整协方差
  // 这样可以补偿由于加速度计标定不准确导致的误差
  acc_cov *= std::pow(G_norm / mean_acc.norm(), 2);

  // === 检查陀螺仪噪声水平 ===
  // 如果陀螺仪噪声过大，说明IMU质量较差或环境不稳定
  if (gyr_cov.norm() > MAX_GYR_VAR) {
    LOG(ERROR) << "Too large noise of gyroscope measurements. " << gyr_cov.norm() << " > " << MAX_GYR_VAR;
    return;  // 初始化失败，返回
  }

  // === 检查加速度计噪声水平 ===
  // 如果加速度计噪声过大，说明IMU质量较差或环境不稳定
  if (acc_cov.norm() > MAX_ACC_VAR) {
    LOG(ERROR) << "Too large noise of accelerometer measurements. " << acc_cov.norm() << " > " << MAX_ACC_VAR;
    return;  // 初始化失败，返回
  }

  // === 设置初始化完成标志 ===
  initial_flag = true;

  // === 重置噪声协方差为预设值 ===
  // 使用预设的噪声模型替代初始化阶段计算的统计值
  // 这样可以避免初始化阶段的特殊条件影响后续的滤波性能
  gyr_cov = gyr_cov_scale;    // 陀螺仪噪声协方差
  acc_cov = acc_cov_scale;    // 加速度计噪声协方差

  // === 计算初始偏置和重力向量 ===
  // 陀螺仪偏置：静止状态下陀螺仪的平均读数即为偏置
  Eigen::Vector3d init_bg = mean_gyr;
  
  // 重力向量：将加速度计平均值归一化后乘以标准重力加速度
  // 静止状态下加速度计主要感受重力加速度
  Eigen::Vector3d init_gravity = mean_acc / mean_acc.norm() * G_norm;

  // === 设置估计器的初始状态 ===
  setBg(init_bg);              // 设置陀螺仪偏置
  setGravity(init_gravity);    // 设置重力向量

  // === 调整状态协方差矩阵 ===
  // 对特定状态变量的协方差进行缩放，反映初始化后的置信度
  
  // 陀螺仪偏置的协方差（状态向量的第9-11维）
  // 乘以0.001表示对初始化得到的陀螺仪偏置有较高的置信度
  covariance.block<3, 3>(9, 9) *= 0.001;
  
  // 加速度计偏置的协方差（状态向量的第12-14维）
  // 乘以0.0001表示对加速度计偏置的初始估计有很高的置信度
  covariance.block<3, 3>(12, 12) *= 0.0001;
  
  // 重力向量的协方差（状态向量的第15-16维，因为重力向量在流形上只有2个自由度）
  // 乘以0.00001表示对重力向量的初始估计有极高的置信度
  covariance.block<2, 2>(15, 15) *= 0.00001;

  // === 初始化噪声参数 ===
  // 设置滤波器的过程噪声和观测噪声参数
  initializeNoise();

  // === 输出初始化结果 ===
  ROS_INFO("IMU Initialization Done.");
  
  // 打印初始化得到的重要参数
  std::cout << "init_gravity = " << init_gravity.transpose() << std::endl;
  std::cout << "init_bg = " << init_bg.transpose() << std::endl;
} 
else {
  // === 初始化条件不满足 ===
  ROS_INFO("Wait more IMU measurements...");
}

return;
}

void eskfEstimator::initialization(
    const std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>>& imu_meas) {
  if (is_first_imu_meas) {
    num_init_meas = 1;
    is_first_imu_meas = false;
    time_first_imu = imu_meas.front().first;
    mean_gyr = imu_meas.front().second.first;
    mean_acc = imu_meas.front().second.second;
  }

  for (const auto& imu : imu_meas) {
    mean_gyr += (imu.second.first - mean_gyr) / num_init_meas;
    mean_acc += (imu.second.second - mean_acc) / num_init_meas;

    gyr_cov = gyr_cov * (num_init_meas - 1.0) / num_init_meas +
              (imu.second.first - mean_gyr).cwiseProduct(imu.second.first - mean_gyr) * (num_init_meas - 1.0) /
                  (num_init_meas * num_init_meas);

    acc_cov = acc_cov * (num_init_meas - 1.0) / num_init_meas +
              (imu.second.second - mean_acc).cwiseProduct(imu.second.second - mean_acc) * (num_init_meas - 1.0) /
                  (num_init_meas * num_init_meas);

    num_init_meas++;
  }

  gyr_0 = imu_meas.back().second.first;
  acc_0 = imu_meas.back().second.second;
}

void eskfEstimator::initializeNoise() {
  noise.block<3, 3>(0, 0).diagonal() = acc_cov;
  noise.block<3, 3>(3, 3).diagonal() = gyr_cov;
  noise.block<3, 3>(6, 6).diagonal() = b_acc_cov;
  noise.block<3, 3>(9, 9).diagonal() = b_gyr_cov;
}

void eskfEstimator::initializeImuData(const Eigen::Vector3d& acc_0_, const Eigen::Vector3d& gyr_0_) {
  acc_0 = acc_0_;
  gyr_0 = gyr_0_;
}

void eskfEstimator::setTranslation(const Eigen::Vector3d& p_) {
  p = p_;
}

void eskfEstimator::setRotation(const Eigen::Quaterniond& q_) {
  q = q_;
}

void eskfEstimator::setVelocity(const Eigen::Vector3d& v_) {
  v = v_;
}

void eskfEstimator::setBa(const Eigen::Vector3d& ba_) {
  ba = ba_;
}

void eskfEstimator::setBg(const Eigen::Vector3d& bg_) {
  bg = bg_;
}

void eskfEstimator::setGravity(const Eigen::Vector3d& g_) {
  g = g_;
}

Eigen::Vector3d eskfEstimator::getTranslation() {
  return p;
}

Eigen::Quaterniond eskfEstimator::getRotation() {
  return q;
}

Eigen::Vector3d eskfEstimator::getVelocity() {
  return v;
}

Eigen::Vector3d eskfEstimator::getBa() {
  return ba;
}

Eigen::Vector3d eskfEstimator::getBg() {
  return bg;
}

Eigen::Vector3d eskfEstimator::getGravity() {
  return g;
}

Eigen::Vector3d eskfEstimator::getLastAcc() {
  return acc_0;
}

Eigen::Vector3d eskfEstimator::getLastGyr() {
  return gyr_0;
}

void eskfEstimator::setCovariance(const Eigen::Matrix<double, 17, 17>& covariance_) {
  covariance = covariance_;
}

Eigen::Matrix<double, 17, 17> eskfEstimator::getCovariance() {
  return covariance;
}

void eskfEstimator::predict(double dt_, const Eigen::Vector3d& acc_1_, const Eigen::Vector3d& gyr_1_) {
  dt = dt_;
  acc_1 = acc_1_;
  gyr_1 = gyr_1_;

  Eigen::Vector3d avr_acc = 0.5 * (acc_0 + acc_1);
  Eigen::Vector3d avr_gyr = 0.5 * (gyr_0 + gyr_1);

  Eigen::Vector3d p_before = p;
  Eigen::Quaterniond q_before = q;
  Eigen::Vector3d v_before = v;
  Eigen::Vector3d ba_before = ba;
  Eigen::Vector3d bg_before = bg;
  Eigen::Vector3d g_before = g;

  Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - bg;
  Eigen::Vector3d un_acc = 0.5 * (acc_0 + acc_1) - ba;
  q = q * numType::so3ToQuat(un_gyr * dt);
  p = p + v * dt;
  v = v + q_before.toRotationMatrix() * un_acc * dt - g * dt;

  Eigen::Matrix3d R_omega_x, R_acc_x;
  R_omega_x << 0, -un_gyr(2), un_gyr(1), un_gyr(2), 0, -un_gyr(0), -un_gyr(1), un_gyr(0), 0;
  R_acc_x << 0, -un_acc(2), un_acc(1), un_acc(2), 0, -un_acc(0), -un_acc(1), un_acc(0), 0;

  Eigen::Matrix<double, 3, 2> B_x = numType::derivativeS2(g);

  Eigen::Matrix<double, 17, 17> F_x = Eigen::MatrixXd::Zero(17, 17);
  F_x.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  F_x.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * dt;
  F_x.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_omega_x * dt;
  F_x.block<3, 3>(3, 12) = -Eigen::Matrix3d::Identity() * dt;
  F_x.block<3, 3>(6, 3) = -q_before.toRotationMatrix() * R_acc_x * dt;
  F_x.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
  F_x.block<3, 3>(6, 9) = -q_before.toRotationMatrix() * dt;
  F_x.block<3, 2>(6, 15) = numType::skewSymmetric(g) * B_x * dt;
  F_x.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
  F_x.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
  F_x.block<2, 2>(15, 15) =
      -1.0 / (g.norm() * g.norm()) * B_x.transpose() * numType::skewSymmetric(g) * numType::skewSymmetric(g) * B_x;

  Eigen::Matrix<double, 17, 12> F_w = Eigen::MatrixXd::Zero(17, 12);
  F_w.block<3, 3>(6, 0) = -q_before.toRotationMatrix() * dt;
  F_w.block<3, 3>(3, 3) = -Eigen::Matrix3d::Identity() * dt;
  F_w.block<3, 3>(9, 6) = -Eigen::Matrix3d::Identity() * dt;
  F_w.block<3, 3>(12, 9) = -Eigen::Matrix3d::Identity() * dt;

  covariance = F_x * covariance * F_x.transpose() + F_w * noise * F_w.transpose();

  acc_0 = acc_1;
  gyr_0 = gyr_1;
}

void eskfEstimator::observe(const Eigen::Matrix<double, 17, 1>& d_x_) {
  p = p + d_x_.head<3>();
  q = (q * numType::so3ToQuat(d_x_.segment<3>(3))).normalized();
  v = v + d_x_.segment<3>(6);
  ba = ba + d_x_.segment<3>(9);
  bg = bg + d_x_.segment<3>(12);

  Eigen::Matrix<double, 3, 2> B_x = numType::derivativeS2(g);
  Eigen::Vector3d so3_dg = B_x * d_x_.tail<2>();
  g = numType::so3ToRotation(so3_dg) * g;
}

void eskfEstimator::observePose(
    const Eigen::Vector3d& translation,
    const Eigen::Quaterniond& rotation,
    double trans_noise,
    double ang_noise) {
  Eigen::Matrix<double, 6, 17> H = Eigen::Matrix<double, 6, 17>::Zero();
  H.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
  Eigen::Vector3d so3 = numType::quatToSo3(q);
  H.block<3, 3>(3, 3) = numType::invJrightSo3(so3);  // gaobo: Identity();

  Eigen::Matrix<double, 6, 1> noise_vec;
  noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise, ang_noise;

  Eigen::Matrix<double, 6, 6> V = noise_vec.asDiagonal();
  Eigen::Matrix<double, 17, 6> K = covariance * H.transpose() * (H * covariance * H.transpose() + V).inverse();

  Eigen::Quaterniond update_q = q.inverse() * rotation;
  Eigen::Vector3d update_so3 = numType::quatToSo3(update_q);
  Eigen::Vector3d update_t = translation - p;

  Eigen::Matrix<double, 6, 1> update_vec = Eigen::Matrix<double, 6, 1>::Zero();
  update_vec.head<3>() = update_t;
  update_vec.tail<3>() = update_so3;

  Eigen::Matrix<double, 17, 1> predict_vec = Eigen::Matrix<double, 17, 1>::Zero();

  delta_state = predict_vec + K * update_vec;
  covariance = (Eigen::MatrixXd::Identity(17, 17) - K * H) * covariance;

  updateAndReset();
}

void eskfEstimator::updateAndReset() {
  p = p + delta_state.block<3, 1>(0, 0);
  q = q * numType::so3ToQuat(delta_state.block<3, 1>(3, 0));
  v = v + delta_state.block<3, 1>(6, 0);
  ba = ba + delta_state.block<3, 1>(9, 0);
  bg = bg + delta_state.block<3, 1>(12, 0);

  g = g + lxly * delta_state.block<2, 1>(15, 0);
  calculateLxly();

  projectCovariance();

  delta_state.setZero();
}

void eskfEstimator::projectCovariance() {
  Eigen::Matrix<double, 17, 17> J;
  J.block<17, 17>(0, 0) = Eigen::MatrixXd::Identity(17, 17);
  J.block<3, 3>(3, 3) = Eigen::MatrixXd::Identity(3, 3) - 0.5 * numType::skewSymmetric(delta_state.block<3, 1>(3, 0));
  covariance = J * covariance * J.transpose();
}

void eskfEstimator::calculateLxly() {
  Eigen::Vector3d b, c;
  Eigen::Vector3d a = g.normalized();
  Eigen::Vector3d temp(0, 0, 1);
  if (a == temp)
    temp << 1, 0, 0;
  b = (temp - a * (a.transpose() * temp)).normalized();
  c = a.cross(b);

  lxly.block<3, 1>(0, 0) = b;
  lxly.block<3, 1>(0, 1) = c;
}
