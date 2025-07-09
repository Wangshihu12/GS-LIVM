#include "liw/imageProcessing.h"
#include "liw/utility.h"

imageProcessing::imageProcessing() {
  time_last_process = -1e5;

  op_tracker = new opticalFlowTracker();
  map_tracker = new rgbMapTracker();

  maximum_tracked_points = 300;
  track_windows_size = 40;

  tracker_minimum_depth = 0.1;
  tracker_maximum_depth = 100;

  num_iterations = 2;

  cam_measurement_weight = 1e-3;

  ifEstimateCameraIntrinsic = false;
  ifEstimateExtrinsic = false;

  first_data = true;

  setInitialCov();
}

void imageProcessing::setImageWidth(int& para) {
  image_width = para;
}

void imageProcessing::setImageRatio(double& para) {
  image_resize_ratio = para;
}

void imageProcessing::setImageHeight(int& para) {
  image_height = para;
}

void imageProcessing::initCameraParams() {
  camera_intrinsic(0, 0) = camera_intrinsic(0, 0) * image_resize_ratio;
  camera_intrinsic(0, 2) = camera_intrinsic(0, 2) * image_resize_ratio;
  camera_intrinsic(1, 1) = camera_intrinsic(1, 1) * image_resize_ratio;
  camera_intrinsic(1, 2) = camera_intrinsic(1, 2) * image_resize_ratio;

  cv::eigen2cv(camera_intrinsic, intrinsic);
  cv::eigen2cv(camera_dist_coeffs, dist_coeffs);

  cv::initUndistortRectifyMap(
      intrinsic,
      dist_coeffs,
      cv::Mat(),
      intrinsic,
      cv::Size(image_width * image_resize_ratio, image_height * image_resize_ratio),
      CV_16SC2,
      m_ud_map1,
      m_ud_map2);

  Eigen::Matrix3d newK = Eigen::Matrix3d::Identity();
  cv::cv2eigen(intrinsic, newK);
  // change
  op_tracker->setIntrinsic(
      newK, camera_dist_coeffs * 0, cv::Size(image_width * image_resize_ratio, image_height * image_resize_ratio));
  op_tracker->maximum_tracked_points = maximum_tracked_points;

  map_tracker->minimum_depth_for_projection = tracker_minimum_depth;
  map_tracker->maximum_depth_for_projection = tracker_maximum_depth;
}

void imageProcessing::setCameraIntrinsic(std::vector<double>& v_camera_intrinsic) {
  camera_intrinsic << v_camera_intrinsic[0], v_camera_intrinsic[1], v_camera_intrinsic[2], v_camera_intrinsic[3],
      v_camera_intrinsic[4], v_camera_intrinsic[5], v_camera_intrinsic[6], v_camera_intrinsic[7], v_camera_intrinsic[8];
}

void imageProcessing::setCameraDistCoeffs(std::vector<double>& v_camera_dist_coeffs) {
  camera_dist_coeffs << v_camera_dist_coeffs[0], v_camera_dist_coeffs[1], v_camera_dist_coeffs[2],
      v_camera_dist_coeffs[3], v_camera_dist_coeffs[4];
}

void imageProcessing::setExtrinR(Eigen::Matrix3d& R) {
  R_imu_camera = R;
}

void imageProcessing::setExtrinT(Eigen::Vector3d& t) {
  t_imu_camera = t;
}

void imageProcessing::setInitialCov() {
  // Set cov
  covariance = Eigen::MatrixXd::Identity(11, 11) * INIT_COV;
  covariance(0, 0) = 0.00001;
  covariance.block<6, 6>(1, 1) = Eigen::MatrixXd::Identity(6, 6) * 1e-3;  // extrinsic between camera and IMU
  covariance.block<4, 4>(7, 7) = Eigen::MatrixXd::Identity(4, 4) * 1e-3;  // camera intrinsic
}

Eigen::Matrix3d imageProcessing::getCameraIntrinsic() {
  return camera_intrinsic;
}

void imageProcessing::printParameter() {
  // std::cout << "image_width: " << image_width << std::endl;
  // std::cout << "image_height: " << image_height << std::endl;
  // std::cout << "camera_intrinsic: \n" << std::fixed << camera_intrinsic << std::endl;
  // std::cout << "camera_dist_coeffs: \n" << std::fixed << camera_dist_coeffs.transpose() << std::endl;
  // std::cout << "R_imu_camera: \n" << std::fixed << R_imu_camera << std::endl;
  // std::cout << "t_imu_camera: \n" << std::fixed << t_imu_camera.transpose() << std::endl;
}

/**
 * 图像处理主函数
 * 执行视觉惯性里程计(VIO)的完整处理流程，包括图像预处理、特征跟踪、位姿优化等
 * @param voxel_map 体素哈希地图引用
 * @param p_frame 当前点云帧指针（包含RGB图像数据）
 * @return 处理是否成功
 */
bool imageProcessing::process(voxelHashMap& voxel_map, cloudFrame* p_frame) {
  
  // === 图像尺寸调整 ===
  // 根据配置的缩放比例调整图像大小，用于控制计算负载
  common::Timer::Evaluate(
      log_time,
      ros::Time::now().toSec(),
      [&]() {
        // 检查是否需要调整图像尺寸
        if (fabs(image_resize_ratio - 1.0) > 1e-6) {
          cv::Mat temp_img;
          // 按比例缩放图像
          cv::resize(
              p_frame->rgb_image,  // 原始RGB图像
              temp_img,           // 缩放后的临时图像
              cv::Size(image_width * image_resize_ratio, image_height * image_resize_ratio));
          p_frame->rgb_image = temp_img;  // 更新帧中的图像

          // 更新图像的列数和行数
          p_frame->image_cols = (int)image_width * image_resize_ratio;
          p_frame->image_rows = (int)image_height * image_resize_ratio;
        } else {
          // 如果不需要缩放，直接使用原始尺寸
          p_frame->image_cols = (int)image_width;
          p_frame->image_rows = (int)image_height;
        }
      },
      "resizeImage");

  // === 图像去畸变处理 ===
  cv::Mat image_undistort;  // 去畸变后的图像
  common::Timer::Evaluate(
      log_time,
      ros::Time::now().toSec(),
      [&]() { 
        // 使用预计算的映射表对图像进行去畸变
        // m_ud_map1和m_ud_map2是相机标定时预计算的重映射表
        cv::remap(p_frame->rgb_image, image_undistort, m_ud_map1, m_ud_map2, cv::INTER_LINEAR); 
      },
      "remapImage");
  
  // 初始化立方插值并生成灰度图像，用于后续的特征跟踪
  p_frame->gray_image = initCubicInterpolation(image_undistort);

  // === 首次数据处理：初始化跟踪器 ===
  if (first_data) {
    std::vector<cv::Point2f> points_2d_vec_temp;   // 临时2D点向量
    std::vector<rgbPoint*> rgb_points_vec_temp;    // 临时RGB点指针向量
    
    // 从体素地图中选择用于投影的点
    map_tracker->selectPointsForProjection(
        voxel_map, p_frame, &rgb_points_vec_temp, &points_2d_vec_temp, 
        track_windows_size * image_resize_ratio, 1);
    
    // 使用选择的点初始化光流跟踪器
    op_tracker->init(p_frame, rgb_points_vec_temp, points_2d_vec_temp);

    first_data = false;  // 标记首次处理完成
  }

  // === 图像特征跟踪 ===
  bool reOfTrack = false;  // 跟踪结果标志
  common::Timer::Evaluate(
      log_time, 
      ros::Time::now().toSec(), 
      [&]() { 
        // 执行图像跟踪，参数-20可能是跟踪质量阈值
        reOfTrack = op_tracker->trackImage(p_frame, -20); 
      }, 
      "trackImage");

  // 检查跟踪是否成功
  if (!reOfTrack) {
    std::cout << ANSI_COLOR_RED_BOLD << "****** Track Error*****" << ANSI_COLOR_RESET << std::endl;
    return false;  // 跟踪失败，返回false
  }

  // === 更新相机内参（如果需要估计） ===
  if (ifEstimateCameraIntrinsic) {
    p_frame->p_state->fx = camera_intrinsic(0, 0);  // 焦距x
    p_frame->p_state->fy = camera_intrinsic(1, 1);  // 焦距y
    p_frame->p_state->cx = camera_intrinsic(0, 2);  // 主点x
    p_frame->p_state->cy = camera_intrinsic(1, 2);  // 主点y
  }

  // === 更新相机外参（如果需要估计） ===
  if (ifEstimateExtrinsic) {
    p_frame->p_state->R_imu_camera = R_imu_camera;  // IMU到相机的旋转矩阵
    p_frame->p_state->t_imu_camera = t_imu_camera;  // IMU到相机的平移向量
  }

  bool enough_points = true;      // 是否有足够的点进行优化
  bool reOfRemovedPoints = false; // 外点去除结果标志

  // === 使用RANSAC PnP去除外点 ===
  common::Timer::Evaluate(
      log_time,
      ros::Time::now().toSec(),
      [&]() { 
        // 使用RANSAC算法结合PnP求解去除跟踪中的外点
        reOfRemovedPoints = op_tracker->removeOutlierUsingRansacPnp(p_frame); 
      },
      "removeOutlierUsingRansacPnp");

  // 检查外点去除是否成功
  if (!reOfRemovedPoints) {
    enough_points = false;
    std::cout << ANSI_COLOR_RED_BOLD << "****** Remove_outlier_using_ransac_pnp error*****" << ANSI_COLOR_RESET
              << std::endl;
    return false;  // 外点去除失败，返回false
  }

  bool res_esikf = true, res_photometric = true;  // 优化结果标志

  // === VIO扩展卡尔曼滤波优化 ===
  if (enough_points) {
    common::Timer::Evaluate(
        log_time, 
        ros::Time::now().toSec(), 
        [&]() { 
          // 执行基于扩展卡尔曼滤波的视觉惯性里程计优化
          res_esikf = vioEsikf(p_frame); 
        }, 
        "vioEsikf");
  }

  // === VIO光度优化 ===
  if (enough_points) {
    common::Timer::Evaluate(
        log_time, 
        ros::Time::now().toSec(), 
        [&]() { 
          // 执行基于光度误差的视觉惯性里程计优化
          res_photometric = vioPhotometric(p_frame); 
        }, 
        "vioPhotometric");
  }

  // === 在最近访问的体素中渲染点 ===
  if (enough_points) {
    common::Timer::Evaluate(
        log_time,
        ros::Time::now().toSec(),
        [&]() {
          // 在最近访问的体素中渲染3D点到2D图像平面
          // 用于后续的视觉跟踪和地图更新
          map_tracker->renderPointsInRecentVoxel(
              voxel_map, p_frame, &map_tracker->voxels_recent_visited, p_frame->time_sweep_end);
        },
        "renderPointsInRecentVoxel");
  }

  // === 更新投影位姿和跟踪点 ===
  if (enough_points) {
    common::Timer::Evaluate(
        log_time,
        ros::Time::now().toSec(),
        [&]() {
          // 更新用于投影的位姿（参数-0.4可能是深度阈值）
          map_tracker->updatePoseForProjection(p_frame, -0.4);
          
          // 刷新用于投影的3D点
          map_tracker->refreshPointsForProjection(voxel_map);
          
          // 更新并添加新的跟踪点
          // 参数：跟踪窗口大小、最大点数限制
          op_tracker->updateAndAppendTrackPoints(
              p_frame, map_tracker, track_windows_size * image_resize_ratio, 1000000);
        },
        "vioOthers");
  }

  // 更新上次处理时间
  time_last_process = p_frame->time_sweep_end;
  return true;  // 处理成功
}

void imageProcessing::imageEqualize(cv::Mat& image, int amp) {
  cv::Mat image_temp;
  cv::Size eqa_image_size = cv::Size(std::max(image.cols * 32.0 / 640, 4.0), std::max(image.cols * 32.0 / 640, 4.0));
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(amp, eqa_image_size);
  clahe->apply(image, image_temp);
  image = image_temp;
}

cv::Mat imageProcessing::initCubicInterpolation(cv::Mat& image) {
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, cv::COLOR_RGB2GRAY);

  return image_gray;
}

cv::Mat imageProcessing::equalizeColorImageYcrcb(cv::Mat& image) {
  cv::Mat hist_equalized_image;
  cv::cvtColor(image, hist_equalized_image, cv::COLOR_BGR2YCrCb);

  // Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
  std::vector<cv::Mat> vec_channels;
  cv::split(hist_equalized_image, vec_channels);

  // Equalize the histogram of only the Y channel
  imageEqualize(vec_channels[0], 1);
  cv::merge(vec_channels, hist_equalized_image);
  cv::cvtColor(hist_equalized_image, hist_equalized_image, cv::COLOR_YCrCb2BGR);

  return hist_equalized_image;
}

double getHuberLoss(double residual, double outlier_threshold = 1.0) {
  double scale = 1.0;

  if (residual / outlier_threshold < 1.0) {
    scale = 1.0;
  } else {
    scale = (2 * sqrt(residual) / sqrt(outlier_threshold) - 1.0) / residual;
  }

  return scale;
}

const int minimum_iteration_points = 10;

/**
 * 视觉惯性里程计扩展卡尔曼滤波优化函数
 * 使用2D-3D点对应关系优化相机的位姿、外参和内参
 * @param p_frame 当前点云帧指针
 * @return 优化是否成功
 */
bool imageProcessing::vioEsikf(cloudFrame* p_frame) {
  scope_color(ANSI_COLOR_BLUE_BOLD);

  // === 声明优化所需的矩阵和向量 ===
  Eigen::Matrix<double, -1, -1> H_mat;         // 雅可比矩阵（观测方程的线性化）
  Eigen::Matrix<double, 11, 1> solution;       // 优化解向量（状态增量）
  Eigen::Matrix<double, -1, 1> residual_vec;   // 残差向量（重投影误差）
  Eigen::Matrix<double, 11, 1> HTr;            // H^T * residual
  Eigen::Matrix<double, 11, 11> HTH;           // H^T * H

  Eigen::Matrix<double, 11, -1> K;             // 卡尔曼增益矩阵
  Eigen::Matrix<double, 11, 1> d_x;            // 状态变量相对于预测值的偏差

  // === 获取跟踪点数量并检查 ===
  int total_point_size = op_tracker->map_rgb_points_in_cur_image_pose.size();

  // 如果跟踪点太少，无法进行可靠的优化
  if (total_point_size < minimum_iteration_points) {
    return false;
  }

  // === 动态分配矩阵大小 ===
  // 每个点贡献2个残差（u和v方向），状态向量维度为11
  H_mat.resize(total_point_size * 2, 11);      // 雅可比矩阵：2n × 11
  residual_vec.resize(total_point_size * 2, 1); // 残差向量：2n × 1

  K.resize(11, total_point_size * 2);          // 卡尔曼增益：11 × 2n

  // === 获取当前状态的预测值（作为线性化点） ===
  double t_predict = p_frame->p_state->time_td;              // 时间偏移预测值
  Eigen::Vector3d p_predict = p_frame->p_state->t_imu_camera; // IMU到相机平移预测值
  Eigen::Quaterniond q_predict = Eigen::Quaterniond(p_frame->p_state->R_imu_camera); // IMU到相机旋转预测值
  double fx_predict = p_frame->p_state->fx;                  // 焦距x预测值
  double fy_predict = p_frame->p_state->fy;                  // 焦距y预测值
  double cx_predict = p_frame->p_state->cx;                  // 主点x预测值
  double cy_predict = p_frame->p_state->cy;                  // 主点y预测值

  int num_used_point_count = 0;  // 实际使用的点数计数器

  // === 初始化残差相关变量 ===
  double acc_residual = 0;        // 当前迭代的累积残差
  double last_acc_residual = 3e8; // 上一次迭代的累积残差（用于收敛判断）

  // === 计算相机测量权重 ===
  // 根据新访问体素数量自适应调整权重，新体素越多权重越小
  cam_measurement_weight = std::max(0.001, std::min(5.0 / map_tracker->number_of_new_visited_voxel, 0.01));

  // === 迭代优化循环 ===
  for (int iter_count = 0; iter_count < num_iterations; iter_count++) {
    int point_idx = -1;      // 点索引计数器
    acc_residual = 0;        // 重置累积残差

    // 声明点处理相关变量
    Eigen::Vector3d point_world, point_camera;           // 3D点的世界坐标和相机坐标
    Eigen::Vector2d pixel_match, pixel_projection, pixel_velocity; // 2D像素：匹配点、投影点、速度

    // === 初始化本次迭代的矩阵 ===
    H_mat.setZero();
    solution.setZero();
    residual_vec.setZero();

    K.setZero();
    d_x.setZero();

    // === 计算当前状态相对于预测值的偏差 ===
    double d_t = p_frame->p_state->time_td - t_predict;                                    // 时间偏移偏差
    Eigen::Vector3d d_p = p_frame->p_state->t_imu_camera - p_predict;                     // 平移偏差
    Eigen::Quaterniond d_q = q_predict.inverse() * Eigen::Quaterniond(p_frame->p_state->R_imu_camera); // 旋转偏差
    Eigen::Vector3d d_so3 = numType::quatToSo3(d_q);                                      // 旋转偏差的so(3)表示
    double d_fx = p_frame->p_state->fx - fx_predict;                                       // 焦距x偏差
    double d_fy = p_frame->p_state->fy - fy_predict;                                       // 焦距y偏差
    double d_cx = p_frame->p_state->cx - cx_predict;                                       // 主点x偏差
    double d_cy = p_frame->p_state->cy - cy_predict;                                       // 主点y偏差

    // === 组装状态偏差向量 d_x ===
    // 状态向量：[td, so3(3), t_ic(3), fx, fy, cx, cy] = 11维
    d_x(0) = d_t;                    // 时间偏移
    d_x.segment<3>(1) = d_so3;       // 旋转（so3表示）
    d_x.segment<3>(4) = d_p;         // 平移
    d_x(7) = d_fx;                   // 焦距x
    d_x(8) = d_fy;                   // 焦距y
    d_x(9) = d_cx;                   // 主点x
    d_x(10) = d_cy;                  // 主点y

    num_used_point_count = 0;

    // === 遍历所有跟踪点构建观测方程 ===
    for (auto it = op_tracker->map_rgb_points_in_last_image_pose.begin();
         it != op_tracker->map_rgb_points_in_last_image_pose.end();
         it++) {
      
      // === 获取点的信息 ===
      point_world = ((rgbPoint*)it->first)->getPosition();    // 3D点世界坐标
      pixel_velocity = ((rgbPoint*)it->first)->image_velocity; // 图像平面速度
      pixel_match = Eigen::Vector2d(it->second.x, it->second.y); // 匹配的2D像素位置

      // === 计算3D点在相机坐标系下的位置 ===
      point_camera = p_frame->p_state->q_camera_world.toRotationMatrix() * point_world + 
                     p_frame->p_state->t_camera_world;

      // === 计算投影到图像平面的像素位置（包含运动补偿） ===
      pixel_projection = Eigen::Vector2d(
                             p_frame->p_state->fx * point_camera(0) / point_camera(2) + p_frame->p_state->cx,
                             p_frame->p_state->fy * point_camera(1) / point_camera(2) + p_frame->p_state->cy) +
                         p_frame->p_state->time_td * pixel_velocity;

      // === 计算重投影残差和Huber损失 ===
      double residual = (pixel_projection - pixel_match).norm();
      double huber_loss = getHuberLoss(residual);  // Huber损失函数，减少外点影响

      point_idx++;
      acc_residual += residual;

      // 将加权残差存入残差向量
      residual_vec.block<2, 1>(point_idx * 2, 0) = (pixel_projection - pixel_match) * huber_loss;

      num_used_point_count++;

      // === 计算雅可比矩阵 ===
      
      // 像素坐标对相机坐标的雅可比：∂u/∂p_c
      Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_u_pc;
      J_u_pc << p_frame->p_state->fx / point_camera.z(), 0,
          -(p_frame->p_state->fx * point_camera.x()) / (point_camera.z() * point_camera.z()), 
          0, p_frame->p_state->fy / point_camera.z(),
          -(p_frame->p_state->fy * point_camera.y()) / (point_camera.z() * point_camera.z());

      // 像素坐标对相机内参的雅可比：∂u/∂K
      Eigen::Matrix<double, 2, 4, Eigen::RowMajor> J_u_K;
      J_u_K << point_camera.x() / point_camera.z(), 0, 1, 0, 
               0, point_camera.y() / point_camera.z(), 0, 1;

      // === 填充雅可比矩阵 H_mat ===
      
      // 对时间偏移的雅可比（运动补偿项）
      H_mat.block<2, 1>(point_idx * 2, 0) = pixel_velocity * huber_loss;

      // 如果估计外参，计算对旋转和平移的雅可比
      if (ifEstimateExtrinsic) {
        // 对旋转的雅可比：∂u/∂δθ = ∂u/∂p_c * ∂p_c/∂δθ
        H_mat.block<2, 3>(point_idx * 2, 1) = J_u_pc * numType::skewSymmetric(point_camera) * huber_loss;
        // 对平移的雅可比：∂u/∂t = -∂u/∂p_c * R^T
        H_mat.block<2, 3>(point_idx * 2, 4) = -J_u_pc * p_frame->p_state->R_imu_camera.transpose() * huber_loss;
      }

      // 如果估计内参，计算对相机内参的雅可比
      if (ifEstimateCameraIntrinsic) {
        H_mat.block<2, 4>(point_idx * 2, 7) = J_u_K * huber_loss;
      }
    }

    // === 计算平均残差 ===
    acc_residual /= total_point_size;

    // 如果可用点数太少，退出迭代
    if (num_used_point_count < minimum_iteration_points) {
      break;
    }

    // === 扩展卡尔曼滤波更新 ===
    
    // 构建雅可比矩阵J（考虑SO(3)流形结构）
    Eigen::Matrix<double, 11, 11> J_zero = Eigen::MatrixXd::Identity(11, 11);
    // 对于SO(3)旋转，使用左雅可比近似：J ≈ I - 0.5 * [δθ]×
    J_zero.block<3, 3>(1, 1) = Eigen::Matrix3d::Identity() - 0.5 * numType::skewSymmetric(d_x.segment<3>(1));

    // 计算卡尔曼增益：K = (H^T*H + (J*P*J^T * weight)^{-1})^{-1} * H^T
    K = (H_mat.transpose() * H_mat + (J_zero * covariance * J_zero.transpose() * cam_measurement_weight).inverse())
            .inverse() *
        H_mat.transpose();
    
    // 计算状态更新：δx = -K*r - (I-K*H)*J*d_x
    solution = -K * residual_vec - (Eigen::Matrix<double, 11, 11>::Identity() - K * H_mat) * J_zero * d_x;

    // === 更新相机参数 ===
    updateCameraParameters(p_frame, solution);

    // === 收敛性检查 ===
    if (fabs(acc_residual - last_acc_residual) < 0.01) {
      break;  // 残差变化小于阈值，认为收敛
    }

    last_acc_residual = acc_residual;
  }

  // === 更新协方差矩阵 ===
  // 使用Joseph形式保证数值稳定性：P = J*(I-K*H)*P*J^T
  Eigen::Matrix<double, 11, 11> J_k = Eigen::MatrixXd::Identity(11, 11);
  J_k.block<3, 3>(1, 1) = Eigen::Matrix3d::Identity() - 0.5 * numType::skewSymmetric(solution.segment<3>(1));

  covariance = J_k * (Eigen::Matrix<double, 11, 11>::Identity() - K * H_mat) * covariance * J_k.transpose();

  return true;
}

/**
 * 更新相机参数函数
 * 根据优化求解得到的状态增量，更新相机的内参、外参以及相关的位姿变换
 * @param p_frame 当前点云帧指针
 * @param d_x 状态增量向量（11维）：[td, so3(3), t_ic(3), fx, fy, cx, cy]
 */
void imageProcessing::updateCameraParameters(cloudFrame* p_frame, Eigen::Matrix<double, 11, 1>& d_x) {
  
  // === 更新时间偏移参数 ===
  // 时间偏移td表示图像时间戳与IMU时间戳之间的偏差
  p_frame->p_state->time_td += d_x(0);

  // === 更新IMU到相机的旋转外参 ===
  // 获取当前的IMU到相机旋转四元数
  Eigen::Quaterniond q_imu_camera = Eigen::Quaterniond(p_frame->p_state->R_imu_camera);
  
  // 在SO(3)流形上进行旋转更新
  // 1. 将so(3)增量转换为四元数：exp(δθ) ≈ I + [δθ]×/2 对于小角度
  // 2. 右乘更新：q_new = q_old * exp(δθ)
  // 3. 归一化保证四元数的单位性质
  q_imu_camera = (q_imu_camera * numType::so3ToQuat(d_x.segment<3>(1))).normalized();

  // 将更新后的四元数转换回旋转矩阵存储
  p_frame->p_state->R_imu_camera = q_imu_camera.toRotationMatrix();
  
  // === 更新IMU到相机的平移外参 ===
  // 平移在欧几里得空间中，直接进行向量加法更新
  p_frame->p_state->t_imu_camera += d_x.segment<3>(4);
  
  // === 更新相机内参 ===
  p_frame->p_state->fx += d_x(7);   // 焦距x方向增量
  p_frame->p_state->fy += d_x(8);   // 焦距y方向增量
  p_frame->p_state->cx += d_x(9);   // 主点x坐标增量
  p_frame->p_state->cy += d_x(10);  // 主点y坐标增量

  // === 计算世界到相机的复合变换 ===
  // 变换链：World → IMU → Camera
  // 旋转复合：R_world_camera = R_world_imu * R_imu_camera
  p_frame->p_state->q_world_camera =
      Eigen::Quaterniond(p_frame->p_state->rotation.toRotationMatrix() * p_frame->p_state->R_imu_camera);
  
  // 平移复合：t_world_camera = R_world_imu * t_imu_camera + t_world_imu
  p_frame->p_state->t_world_camera =
      p_frame->p_state->rotation.toRotationMatrix() * p_frame->p_state->t_imu_camera + p_frame->p_state->translation;

  // === 刷新投影相关的缓存变量 ===
  // 更新相机到世界的逆变换等投影计算所需的缓存数据
  p_frame->refreshPoseForProjection();
}

/**
 * 视觉惯性里程计光度优化函数
 * 使用RGB颜色信息优化相机外参（IMU到相机的旋转和平移）
 * @param p_frame 当前点云帧指针
 * @return 优化是否成功
 */
bool imageProcessing::vioPhotometric(cloudFrame* p_frame) {
  // === 声明优化所需的矩阵和向量 ===
  Eigen::Matrix<double, -1, -1> H_mat, R_mat_inv, sqrt_info; // 雅可比矩阵、信息矩阵、平方根信息矩阵
  Eigen::Matrix<double, 6, 1> solution;                      // 优化解向量（6维：3D旋转+3D平移）
  Eigen::Matrix<double, -1, 1> residual_vec;                 // 残差向量（RGB颜色差异）
  Eigen::Matrix<double, 6, 1> HTr;                           // H^T * residual
  Eigen::Matrix<double, 6, 6> HTH;                           // H^T * H

  Eigen::Matrix<double, 6, -1> K;                            // 卡尔曼增益矩阵
  Eigen::Matrix<double, 6, 1> d_x;                           // 状态变量相对于预测值的偏差

  // === 获取跟踪点数量并检查 ===
  int total_point_size = op_tracker->map_rgb_points_in_cur_image_pose.size();

  // 如果跟踪点太少，无法进行可靠的优化
  if (total_point_size < minimum_iteration_points) {
    return false;
  }

  // === 动态分配矩阵大小 ===
  // 每个点贡献3个残差（R、G、B），状态向量维度为6（外参：3D旋转+3D平移）
  H_mat.resize(total_point_size * 3, 6);        // 雅可比矩阵：3n × 6
  residual_vec.resize(total_point_size * 3, 1); // 残差向量：3n × 1
  R_mat_inv.resize(total_point_size * 3, total_point_size * 3); // 信息矩阵：3n × 3n
  sqrt_info.resize(total_point_size * 3, total_point_size * 3); // 平方根信息矩阵：3n × 3n

  K.resize(6, total_point_size * 3);            // 卡尔曼增益：6 × 3n

  // === 获取外参的预测值（作为线性化点） ===
  Eigen::Vector3d p_predict = p_frame->p_state->t_imu_camera;              // IMU到相机平移预测值
  Eigen::Quaterniond q_predict = Eigen::Quaterniond(p_frame->p_state->R_imu_camera); // IMU到相机旋转预测值

  int num_used_point_count = 0;  // 实际使用的点数计数器

  // === 初始化残差相关变量 ===
  double acc_residual = 0;        // 当前迭代的累积残差
  double last_acc_residual = 3e8; // 上一次迭代的累积残差（用于收敛判断）

  // === 计算相机测量权重 ===
  // 根据新访问体素数量自适应调整权重
  cam_measurement_weight = std::max(0.001, std::min(5.0 / map_tracker->number_of_new_visited_voxel, 0.01));

  // === 迭代优化循环 ===
  for (int iter_count = 0; iter_count < num_iterations; iter_count++) {
    int point_idx = -1;      // 点索引计数器
    acc_residual = 0;        // 重置累积残差

    // 声明点处理相关变量
    Eigen::Vector3d point_world, point_camera;                     // 3D点的世界坐标和相机坐标
    Eigen::Vector2d pixel_match, pixel_projection, pixel_velocity; // 2D像素：匹配点、投影点、速度

    // === 初始化本次迭代的矩阵 ===
    H_mat.setZero();
    solution.setZero();
    residual_vec.setZero();
    R_mat_inv.setZero();
    sqrt_info.setZero();

    K.setZero();
    d_x.setZero();

    // === 计算当前外参相对于预测值的偏差 ===
    Eigen::Vector3d d_p = p_frame->p_state->t_imu_camera - p_predict;                     // 平移偏差
    Eigen::Quaterniond d_q = q_predict.inverse() * Eigen::Quaterniond(p_frame->p_state->R_imu_camera); // 旋转偏差
    Eigen::Vector3d d_so3 = numType::quatToSo3(d_q);                                      // 旋转偏差的so(3)表示

    // === 组装状态偏差向量 d_x（6维：仅外参） ===
    d_x.head<3>() = d_so3;   // 旋转偏差（so3表示）
    d_x.tail<3>() = d_p;     // 平移偏差

    num_used_point_count = 0;

    // === 遍历所有跟踪点构建光度观测方程 ===
    for (auto it = op_tracker->map_rgb_points_in_last_image_pose.begin();
         it != op_tracker->map_rgb_points_in_last_image_pose.end();
         it++) {
      
      // === 质量检查：只使用被观测足够次数的点 ===
      if (((rgbPoint*)it->first)->N_rgb < 3)
        continue;  // 跳过观测次数少于3次的点

      point_idx++;

      // === 获取点的信息 ===
      point_world = ((rgbPoint*)it->first)->getPosition();    // 3D点世界坐标
      pixel_velocity = ((rgbPoint*)it->first)->image_velocity; // 图像平面速度

      // === 计算3D点在相机坐标系下的位置 ===
      point_camera = p_frame->p_state->q_camera_world.toRotationMatrix() * point_world + 
                     p_frame->p_state->t_camera_world;

      // === 计算投影到图像平面的像素位置（包含运动补偿） ===
      pixel_projection = Eigen::Vector2d(
                             p_frame->p_state->fx * point_camera(0) / point_camera(2) + p_frame->p_state->cx,
                             p_frame->p_state->fy * point_camera(1) / point_camera(2) + p_frame->p_state->cy) +
                         p_frame->p_state->time_td * pixel_velocity;

      // === 获取点的颜色信息和不确定性 ===
      Eigen::Vector3d point_color = ((rgbPoint*)it->first)->getRgb();        // 地图中存储的RGB颜色
      Eigen::Matrix3d point_rgb_info = Eigen::Matrix3d::Zero();              // RGB信息矩阵（协方差的逆）
      Eigen::Matrix3d point_rgb_cov = ((rgbPoint*)it->first)->getCovRgb();   // RGB协方差矩阵

      // === 构建RGB信息矩阵（对角矩阵） ===
      for (int i = 0; i < 3; i++) {
        point_rgb_info(i, i) = 1.0 / point_rgb_cov(i, i);  // 信息 = 1/方差
        R_mat_inv(point_idx * 3 + i, point_idx * 3 + i) = point_rgb_info(i, i);
        sqrt_info(point_idx * 3 + i, point_idx * 3 + i) = sqrt(R_mat_inv(point_idx * 3 + i, point_idx * 3 + i));
      }

      // === 从当前图像中采样RGB颜色及其梯度 ===
      Eigen::Vector3d obs_color_dx, obs_color_dy;  // 颜色对像素坐标的梯度
      Eigen::Vector3d obs_color = p_frame->getRgb(pixel_projection(0), pixel_projection(1), 0, 
                                                   &obs_color_dx, &obs_color_dy);

      // === 计算光度残差 ===
      Eigen::Vector3d residual = obs_color - point_color;  // 观测颜色 - 地图颜色

      // === 应用Huber损失函数增强鲁棒性 ===
      double huber_loss = getHuberLoss(residual.norm());  // 基于残差范数计算Huber权重
      residual *= huber_loss;

      // 将加权残差存入残差向量
      residual_vec.block<3, 1>(point_idx * 3, 0) = (obs_color - point_color) * huber_loss;

      // 累积加权残差（用于收敛判断）
      acc_residual += residual.transpose() * point_rgb_info * residual;

      // === 计算雅可比矩阵链式法则 ===
      
      // 颜色对像素坐标的雅可比：∂color/∂u
      Eigen::Matrix<double, 3, 2, Eigen::RowMajor> J_color_u;
      J_color_u.block<3, 1>(0, 0) = obs_color_dx;  // ∂color/∂u
      J_color_u.block<3, 1>(0, 1) = obs_color_dy;  // ∂color/∂v

      num_used_point_count++;

      // 像素坐标对相机坐标的雅可比：∂u/∂p_c
      Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_u_pc;
      J_u_pc << p_frame->p_state->fx / point_camera.z(), 0,
          -(p_frame->p_state->fx * point_camera.x()) / (point_camera.z() * point_camera.z()), 
          0, p_frame->p_state->fy / point_camera.z(),
          -(p_frame->p_state->fy * point_camera.y()) / (point_camera.z() * point_camera.z());

      // 链式法则：∂color/∂p_c = ∂color/∂u * ∂u/∂p_c
      Eigen::Matrix3d J_color_pc = J_color_u * J_u_pc;

      // === 填充雅可比矩阵 H_mat（仅外参） ===
      if (ifEstimateExtrinsic) {
        // 对旋转的雅可比：∂color/∂δθ = ∂color/∂p_c * ∂p_c/∂δθ
        H_mat.block<3, 3>(point_idx * 3, 0) = J_color_pc * numType::skewSymmetric(point_camera) * huber_loss;
        // 对平移的雅可比：∂color/∂t = -∂color/∂p_c * R^T
        H_mat.block<3, 3>(point_idx * 3, 3) = -J_color_pc * p_frame->p_state->R_imu_camera.transpose() * huber_loss;
      }
    }

    // === 检查可用点数 ===
    if (num_used_point_count < minimum_iteration_points) {
      break;  // 可用点数太少，退出迭代
    }

    // === 扩展卡尔曼滤波更新（仅外参） ===
    
    // 构建雅可比矩阵J（考虑SO(3)流形结构）
    Eigen::Matrix<double, 6, 6> J_zero = Eigen::MatrixXd::Identity(6, 6);
    
    // 计算SO(3)的左雅可比近似
    common::Timer::Evaluate(
        log_time,
        ros::Time::now().toSec(),
        [&]() { J_zero.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() - 0.5 * numType::skewSymmetric(d_x.head<3>()); },
        "eq_skew");

    // === 高效矩阵计算 ===
    Eigen::Matrix<double, 6, 6> eq_Inv, eq_H_mat1;

    // 使用CUDA加速的矩阵乘法：H^T * R^{-1}
    eq_H_mat1 = cublasMul.multiply(H_mat.transpose(), R_mat_inv);

    // 计算先验项的逆：(J*P*J^T * weight)^{-1}
    eq_Inv = (J_zero * covariance.block<6, 6>(1, 1) * J_zero.transpose() * cam_measurement_weight).inverse();

    // 计算卡尔曼增益：K = (H^T*R^{-1}*H + Prior^{-1})^{-1} * H^T*R^{-1}
    K = (eq_H_mat1 * H_mat + eq_Inv).inverse() * eq_H_mat1;
    
    // 计算状态更新：δx = -K*r - (I-K*H)*J*d_x
    solution = -K * residual_vec - (Eigen::Matrix<double, 6, 6>::Identity() - K * H_mat) * J_zero * d_x;
    
    // === 更新相机外参 ===
    updateCameraParameters(p_frame, solution);

    // === 收敛性检查 ===
    
    // 如果平均残差足够小，认为收敛
    if ((acc_residual / total_point_size) < 10) {
      break;
    }

    // 如果残差变化足够小，认为收敛
    if (fabs(acc_residual - last_acc_residual) < 0.01) {
      break;
    }

    last_acc_residual = acc_residual;
  }

  // === 更新协方差矩阵（仅外参部分） ===
  // 使用Joseph形式保证数值稳定性
  Eigen::Matrix<double, 6, 6> J_k = Eigen::MatrixXd::Identity(6, 6);
  J_k.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() - 0.5 * numType::skewSymmetric(solution.head<3>());

  // 更新协方差矩阵的外参部分：P_{ext} = J*(I-K*H)*P_{ext}*J^T
  covariance.block<6, 6>(1, 1) =
      J_k * (Eigen::Matrix<double, 6, 6>::Identity() - K * H_mat) * covariance.block<6, 6>(1, 1) * J_k.transpose();

  return true;
}

/**
 * 更新相机参数函数（6维版本）
 * 仅更新外参，用于光度优化
 * @param p_frame 当前点云帧指针
 * @param d_x 状态增量向量（6维）：[so3(3), t_ic(3)]
 */
void imageProcessing::updateCameraParameters(cloudFrame* p_frame, Eigen::Matrix<double, 6, 1>& d_x) {
  // === 更新IMU到相机的旋转外参 ===
  Eigen::Quaterniond q_imu_camera = Eigen::Quaterniond(p_frame->p_state->R_imu_camera);
  q_imu_camera = (q_imu_camera * numType::so3ToQuat(d_x.head<3>())).normalized();
  p_frame->p_state->R_imu_camera = q_imu_camera.toRotationMatrix();
  
  // === 更新IMU到相机的平移外参 ===
  p_frame->p_state->t_imu_camera += d_x.tail<3>();

  // === 计算世界到相机的复合变换 ===
  p_frame->p_state->q_world_camera =
      Eigen::Quaterniond(p_frame->p_state->rotation.toRotationMatrix() * p_frame->p_state->R_imu_camera);
  p_frame->p_state->t_world_camera =
      p_frame->p_state->rotation.toRotationMatrix() * p_frame->p_state->t_imu_camera + p_frame->p_state->translation;

  // === 刷新投影相关的缓存变量 ===
  p_frame->refreshPoseForProjection();
}