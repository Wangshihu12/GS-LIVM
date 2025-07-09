#include "liw/opticalFlowTracker.h"

opticalFlowTracker::opticalFlowTracker() {
  cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.05);

  if (lk_optical_flow_kernel == nullptr)
    lk_optical_flow_kernel =
        std::make_shared<LKOpticalFlowKernel>(cv::Size(21, 21), 3, criteria, cv_OPTFLOW_LK_GET_MIN_EIGENVALS);

  maximum_tracked_points = 300;
}

/**
 * 更新和添加跟踪点函数
 * 管理光流跟踪点的生命周期：删除质量差的点，补充新的高质量点
 * @param p_frame 当前点云帧指针
 * @param map_tracker 地图跟踪器指针（提供候选点）
 * @param mini_distance 最小点间距离（用于稀疏化）
 * @param minimum_frame_diff 最小帧差（目前未使用）
 */
void opticalFlowTracker::updateAndAppendTrackPoints(
  cloudFrame* p_frame,
  rgbMapTracker* map_tracker,
  double mini_distance,
  int minimum_frame_diff) {

// === 声明局部变量 ===
double u_d, v_d;  // 浮点像素坐标
int u_i, v_i;     // 量化后的整数像素坐标

// === 计算重投影误差阈值 ===
// 根据图像尺寸自适应调整阈值，大图像允许更大误差
double max_allow_reproject_error = 2.0 * p_frame->image_cols / 320.0;

// === 创建2D占用图 ===
// 用于记录已被占用的像素位置，避免重复放置点
Hash_map_2d<int, float> map_2d_points_occupied;

// === 第一阶段：检查并清理现有跟踪点 ===
for (auto it = map_rgb_points_in_last_image_pose.begin(); 
     it != map_rgb_points_in_last_image_pose.end();) {
  
  // 获取RGB点指针和3D位置
  rgbPoint* rgb_point = ((rgbPoint*)it->first);
  Eigen::Vector3d point_3d = ((rgbPoint*)it->first)->getPosition();

  // === 重投影检查 ===
  // 将3D点投影到当前图像，获取理论像素位置
  bool res = p_frame->project3dPointInThisImage(point_3d, u_d, v_d, nullptr, 1.0);

  // === 量化像素坐标 ===
  // 将连续坐标量化到最小距离的整数倍，用于稀疏化和去重
  u_i = std::round(u_d / mini_distance) * mini_distance;
  v_i = std::round(v_d / mini_distance) * mini_distance;

  // === 计算重投影误差 ===
  // 比较3D点的理论投影位置与上次跟踪记录的位置
  double error = Eigen::Vector2d(u_d - it->second.x, v_d - it->second.y).norm();

  // === 外点检测和删除机制 ===
  if (error > max_allow_reproject_error) {
    // 重投影误差过大，增加外点计数
    rgb_point->is_out_lier_count++;

    // 外点删除条件：
    // 1. 连续两次被标记为外点，或
    // 2. 单次误差超过阈值的两倍
    if ((rgb_point->is_out_lier_count > 1) || (error > max_allow_reproject_error * 2)) {
      rgb_point->is_out_lier_count = 0;  // 重置计数器
      it = map_rgb_points_in_last_image_pose.erase(it);  // 删除该跟踪点
      continue;  // 跳过后续处理，继续下一个点
    }
  } else {
    // 重投影误差在可接受范围内，重置外点计数
    rgb_point->is_out_lier_count = 0;
  }

  // === 更新占用图 ===
  if (res) {
    // 如果投影成功，计算点到相机的距离（深度）
    double depth = (point_3d - p_frame->p_state->t_world_camera).norm();

    // 如果该像素位置未被占用，记录深度信息
    if (map_2d_points_occupied.if_exist(u_i, v_i) == false) {
      map_2d_points_occupied.insert(u_i, v_i, depth);
    }
  }

  it++;  // 移动到下一个跟踪点
}

// === 第二阶段：添加新的跟踪点 ===
// 检查地图跟踪器是否提供了候选点集合
if (map_tracker->points_rgb_vec_for_projection != nullptr) {
  int point_size = map_tracker->points_rgb_vec_for_projection->size();

  // 遍历所有候选点
  for (int i = 0; i < point_size; i++) {
    // === 检查是否已存在 ===
    // 如果该点已经在跟踪列表中，跳过
    if (map_rgb_points_in_last_image_pose.find((*(map_tracker->points_rgb_vec_for_projection))[i]) !=
        map_rgb_points_in_last_image_pose.end()) {
      continue;
    }

    // === 获取候选点信息 ===
    Eigen::Vector3d point_3d = (*(map_tracker->points_rgb_vec_for_projection))[i]->getPosition();

    // === 投影检查 ===
    // 尝试将3D点投影到当前图像
    bool res = p_frame->project3dPointInThisImage(point_3d, u_d, v_d, nullptr, 1.0);

    // 量化像素坐标
    u_i = std::round(u_d / mini_distance) * mini_distance;
    v_i = std::round(v_d / mini_distance) * mini_distance;

    if (res) {
      // === 深度和占用检查 ===
      double depth = (point_3d - p_frame->p_state->t_world_camera).norm();

      // 只有当该像素位置未被占用时，才添加新点
      if (map_2d_points_occupied.if_exist(u_i, v_i) == false) {
        // 更新占用图
        map_2d_points_occupied.insert(u_i, v_i, depth);

        // === 添加新跟踪点 ===
        // 将新的3D点及其2D投影位置添加到跟踪映射中
        map_rgb_points_in_last_image_pose[(*(map_tracker->points_rgb_vec_for_projection))[i]] = 
            cv::Point2f(u_d, v_d);
      }
    }

    // === 跟踪点数量限制 ===
    // 如果跟踪点数量达到上限，停止添加新点
    if (map_rgb_points_in_last_image_pose.size() >= maximum_tracked_points) {
      break;
    }
  }
}

// === 更新跟踪数据结构 ===
// 重新组织跟踪点数据，为下一次光流跟踪准备向量和ID
updateLastTrackingVectorAndIds();
}

void opticalFlowTracker::setIntrinsic(
    Eigen::Matrix3d intrinsic_,
    Eigen::Matrix<double, 5, 1> dist_coeffs_,
    cv::Size image_size_) {
  cv::eigen2cv(intrinsic_, intrinsic);
  // cv::eigen2cv(dist_coeffs_, dist_coeffs);
  // initUndistortRectifyMap(intrinsic, dist_coeffs, cv::Mat(), intrinsic, image_size_, CV_16SC2, m_ud_map1, m_ud_map2);
}

/**
 * 光流跟踪图像主函数
 * 对当前帧执行特征点跟踪，包括光流计算、异常点过滤、速度估计等
 * @param p_frame 当前点云帧指针（包含RGB和灰度图像）
 * @param distance 距离阈值，用于错误点拒绝（大于0时启用）
 * @return 跟踪是否成功
 */
bool opticalFlowTracker::trackImage(cloudFrame* p_frame, double distance) {
  // === 设置当前帧数据 ===
  cur_image = p_frame->rgb_image;              // 获取当前RGB图像
  current_image_time = p_frame->time_sweep_end; // 获取当前图像时间戳
  map_rgb_points_in_cur_image_pose.clear();    // 清空当前帧的RGB点映射

  // === 检查图像有效性 ===
  if (cur_image.empty())
    return false;  // 如果图像为空，跟踪失败

  // 获取当前帧的灰度图像（光流算法在灰度图上进行）
  cv::Mat gray_image = p_frame->gray_image;

  // === 准备光流跟踪的数据结构 ===
  std::vector<uchar> status;  // 跟踪状态向量（成功/失败）
  std::vector<float> error;   // 跟踪误差向量

  // 将上一帧的跟踪点作为当前帧的初始猜测
  cur_tracked_points = last_tracked_points;

  // 记录跟踪前的点数，用于统计
  int before_track = last_tracked_points.size();

  // === 检查跟踪点数量 ===
  // 如果跟踪点太少（少于30个），无法进行可靠的跟踪
  if (last_tracked_points.size() < 30) {
    last_image_time = current_image_time;
    return false;  // 跟踪失败
  }

  // === 执行LK光流跟踪 ===
  // 使用金字塔LK光流算法跟踪特征点
  lk_optical_flow_kernel->trackImage(gray_image, last_tracked_points, cur_tracked_points, status, 2);

  // === 过滤跟踪失败的点 ===
  // 根据status向量，移除跟踪失败的点
  reduce_vector(last_tracked_points, status);  // 过滤上一帧跟踪点
  reduce_vector(old_ids, status);              // 过滤对应的ID
  reduce_vector(cur_tracked_points, status);   // 过滤当前帧跟踪点

  // 记录初步过滤后的点数
  int after_track = last_tracked_points.size();
  cv::Mat mat_F;  // 基础矩阵

  // === 使用基础矩阵进一步过滤异常点 ===
  unsigned int points_before_F = last_tracked_points.size();
  
  // 计算基础矩阵，用RANSAC算法过滤外点
  // 参数：重投影误差1.0像素，置信度99.7%
  mat_F = cv::findFundamentalMat(last_tracked_points, cur_tracked_points, cv::FM_RANSAC, 1.0, 0.997, status);

  unsigned int size_a = cur_tracked_points.size();
  
  // 根据基础矩阵的内点标记，再次过滤点集
  reduce_vector(last_tracked_points, status);
  reduce_vector(old_ids, status);
  reduce_vector(cur_tracked_points, status);

  // === 建立当前帧的RGB点映射 ===
  map_rgb_points_in_cur_image_pose.clear();

  // 计算帧间时间差，用于速度计算
  double frame_time_diff = (current_image_time - last_image_time);

  // 遍历所有成功跟踪的点
  for (uint i = 0; i < last_tracked_points.size(); i++) {
    // === 检查点是否在有效图像区域内 ===
    // 参数：缩放比例1.0，边界0.05（5%的边界缓冲）
    if (p_frame->if2dPointsAvailable(cur_tracked_points[i].x, cur_tracked_points[i].y, 1.0, 0.05)) {
      // 获取对应的RGB点指针
      rgbPoint* rgb_point_ptr = ((rgbPoint*)rgb_points_ptr_vec_in_last_image[old_ids[i]]);
      
      // 建立RGB点到当前2D位置的映射
      map_rgb_points_in_cur_image_pose[rgb_point_ptr] = cur_tracked_points[i];

      // === 计算图像平面上的点运动速度 ===
      cv::Point2f point_image_velocity;

      // 避免除零错误：如果时间差太小，设置最小速度
      if (frame_time_diff < 1e-5)
        point_image_velocity = cv::Point2f(1e-3, 1e-3);
      else
        // 计算像素位移除以时间差得到速度（像素/秒）
        point_image_velocity = (cur_tracked_points[i] - last_tracked_points[i]) / frame_time_diff;

      // 将速度信息存储到RGB点对象中
      rgb_point_ptr->image_velocity = Eigen::Vector2d(point_image_velocity.x, point_image_velocity.y);
    }
  }

  // === 检查最终跟踪点数量 ===
  // 如果成功跟踪的点太少（少于10个），认为跟踪失败
  if (map_rgb_points_in_cur_image_pose.size() < 10) {
    return false;
  }

  // === 可选的错误点拒绝 ===
  // 如果提供了距离阈值，执行基于几何一致性的错误点拒绝
  if (distance > 0)
    rejectErrorTrackingPoints(p_frame, distance);

  // === 更新图像缓冲区 ===
  // 将当前帧数据保存为"上一帧"，为下次跟踪做准备
  old_gray = gray_image.clone();
  old_image = cur_image;

  // === 更新跟踪点数据 ===
  // 释放旧的跟踪点内存并更新
  std::vector<cv::Point2f>().swap(last_tracked_points);
  last_tracked_points = cur_tracked_points;

  // 更新跟踪向量和ID，为下一帧跟踪准备数据结构
  updateLastTrackingVectorAndIds();

  // === 更新状态变量 ===
  image_idx++;                              // 图像索引递增
  last_image_time = current_image_time;     // 更新时间戳

  return true;  // 跟踪成功
}

/**
 * 光流跟踪器初始化函数
 * 设置初始跟踪点并执行第一次光流跟踪
 * @param p_frame 当前点云帧指针（包含RGB和灰度图像）
 * @param rgb_points_vec RGB点向量（3D点集合）
 * @param points_2d_vec 对应的2D投影点向量
 */
void opticalFlowTracker::init(
  cloudFrame* p_frame,
  std::vector<rgbPoint*>& rgb_points_vec,
  std::vector<cv::Point2f>& points_2d_vec) {

// === 设置跟踪点 ===
// 将RGB图像、3D点和对应的2D投影点设置为跟踪器的初始跟踪点
// 这些点将作为光流跟踪的起始特征点
setTrackPoints(p_frame->rgb_image, rgb_points_vec, points_2d_vec);

// === 更新时间戳 ===
// 设置当前图像的时间戳为扫描结束时间
current_image_time = p_frame->time_sweep_end;
// 将上一帧时间也设置为相同值，表示这是第一帧
last_image_time = current_image_time;

// === 执行初始光流跟踪 ===
// 创建状态向量，用于记录跟踪的成功/失败状态
std::vector<uchar> status;

// 使用LK光流算法在灰度图像上跟踪特征点
// 参数说明：
// - p_frame->gray_image: 当前帧的灰度图像
// - last_tracked_points: 上一帧的跟踪点（初始化时与当前点相同）
// - cur_tracked_points: 当前帧跟踪到的点位置（输出）
// - status: 每个点的跟踪状态（成功/失败）
lk_optical_flow_kernel->trackImage(p_frame->gray_image, last_tracked_points, cur_tracked_points, status);
}

/**
 * 设置跟踪点函数
 * 为光流跟踪器设置新的跟踪点集合，建立3D点与2D投影点的对应关系
 * @param image 输入的RGB图像
 * @param rgb_points_vec RGB点向量（3D点集合）
 * @param points_2d_vec 对应的2D投影点向量
 */
void opticalFlowTracker::setTrackPoints(
  cv::Mat& image,
  std::vector<rgbPoint*>& rgb_points_vec,
  std::vector<cv::Point2f>& points_2d_vec) {

// === 图像数据准备 ===
// 克隆输入图像，避免外部修改影响跟踪器内部状态
old_image = image.clone();

// 将RGB图像转换为灰度图像
// 光流跟踪通常在灰度图像上进行，因为灰度信息足够且计算效率更高
cv::cvtColor(old_image, old_gray, cv::COLOR_BGR2GRAY);

// === 清空上一帧的映射关系 ===
// 清空之前存储的RGB点到图像位置的映射，为新的跟踪点做准备
map_rgb_points_in_last_image_pose.clear();

// === 建立3D点到2D位置的映射 ===
// 遍历所有输入的RGB点，建立3D点指针到2D像素坐标的映射关系
for (unsigned int i = 0; i < rgb_points_vec.size(); i++) {
  // 使用3D点的指针作为键，2D投影点作为值
  // 这样可以在后续跟踪中快速查找每个3D点对应的2D位置
  map_rgb_points_in_last_image_pose[(void*)rgb_points_vec[i]] = points_2d_vec[i];
}

// === 更新跟踪向量和ID ===
// 根据新建立的映射关系，更新内部的跟踪点向量和对应的ID
// 这一步为后续的光流跟踪准备数据结构
updateLastTrackingVectorAndIds();
}

void opticalFlowTracker::rejectErrorTrackingPoints(cloudFrame* p_frame, double distance) {
  double u, v;

  int remove_count = 0;
  int total_count = map_rgb_points_in_cur_image_pose.size();

  scope_color(ANSI_COLOR_BLUE_BOLD);

  for (auto it = map_rgb_points_in_cur_image_pose.begin(); it != map_rgb_points_in_cur_image_pose.end(); it++) {
    cv::Point2f predicted_point = it->second;

    Eigen::Vector3d point_position = ((rgbPoint*)it->first)->getPosition();

    int res = p_frame->project3dPointInThisImage(point_position, u, v, nullptr, 1.0);

    if (res) {
      if ((fabs(u - predicted_point.x) > distance) || (fabs(v - predicted_point.y) > distance)) {
        // Remove tracking pts
        map_rgb_points_in_cur_image_pose.erase(it);
        remove_count++;
      }
    } else {
      map_rgb_points_in_cur_image_pose.erase(it);
      remove_count++;
    }
  }
}

/**
 * 更新上一帧跟踪向量和ID
 * 将映射中的3D点和2D位置数据重新组织成向量形式，为光流跟踪算法准备数据结构
 */
void opticalFlowTracker::updateLastTrackingVectorAndIds() {
  // === 初始化索引计数器 ===
  int idx = 0;

  // === 清空所有相关的向量容器 ===
  // 清空上一帧的跟踪点坐标向量
  last_tracked_points.clear();
  
  // 清空上一帧图像中RGB点指针向量
  // 这个向量保存3D点的指针，用于维护3D-2D对应关系
  rgb_points_ptr_vec_in_last_image.clear();

  // 清空旧的ID向量
  // ID用于标识和管理每个跟踪点
  old_ids.clear();

  // === 遍历映射并重新组织数据 ===
  // 从哈希映射中提取数据，按顺序重新组织成向量形式
  for (auto it = map_rgb_points_in_last_image_pose.begin(); 
       it != map_rgb_points_in_last_image_pose.end(); it++) {
    
    // 将3D点指针添加到指针向量中
    // it->first 是3D点的指针（void*类型）
    rgb_points_ptr_vec_in_last_image.push_back(it->first);
    
    // 将对应的2D像素坐标添加到跟踪点向量中
    // it->second 是cv::Point2f类型的2D坐标
    last_tracked_points.push_back(it->second);

    // 为每个点分配一个唯一的索引ID
    // 这个ID用于在光流跟踪过程中识别和管理各个特征点
    old_ids.push_back(idx);

    // 递增索引，为下一个点准备
    idx++;
  }
}

/**
 * 使用RANSAC PnP算法移除异常点
 * 通过求解3D-2D点对应关系的相机位姿，识别并可选地移除几何不一致的异常点
 * @param p_frame 当前点云帧指针
 * @param if_remove_ourlier 是否实际移除异常点（1=移除，0=仅检测）
 * @return PnP求解是否成功
 */
bool opticalFlowTracker::removeOutlierUsingRansacPnp(cloudFrame* p_frame, int if_remove_ourlier) {
  // === 位姿变量声明 ===
  cv::Mat cv_so3, cv_trans;           // OpenCV格式的旋转向量和平移向量
  Eigen::Vector3d eigen_so3, eigen_trans; // Eigen格式的旋转和平移（未使用）

  // === 准备PnP算法的输入数据 ===
  std::vector<cv::Point3f> points_3d; // 3D点集合（世界坐标系）
  std::vector<cv::Point2f> points_2d; // 对应的2D点集合（图像坐标系）

  std::vector<void*> map_ptr;         // 保存RGB点指针，用于后续的映射更新

  // === 从当前帧映射中提取3D-2D对应点 ===
  for (auto it = map_rgb_points_in_cur_image_pose.begin(); 
       it != map_rgb_points_in_cur_image_pose.end(); it++) {
    
    // 保存RGB点指针，用于识别哪些点是内点
    map_ptr.push_back(it->first);
    
    // 获取3D点的世界坐标
    Eigen::Vector3d point_3d = ((rgbPoint*)it->first)->getPosition();

    // 转换为OpenCV格式并添加到3D点集合
    points_3d.push_back(cv::Point3f(point_3d(0), point_3d(1), point_3d(2)));
    
    // 添加对应的2D图像坐标
    points_2d.push_back(it->second);
  }

  // === 检查点数是否足够 ===
  // PnP算法至少需要4对3D-2D对应点
  if (points_3d.size() < 4) {
    return false;  // 点数不够，无法求解
  }

  // === RANSAC PnP求解 ===
  std::vector<int> status; // 内点标记向量（RANSAC输出的内点索引）

  try {
    // 使用RANSAC PnP算法求解相机位姿
    cv::solvePnPRansac(
        points_3d,       // 输入：3D点集合
        points_2d,       // 输入：对应的2D点集合
        intrinsic,       // 输入：相机内参矩阵
        cv::Mat(),       // 输入：畸变系数（这里为空，表示无畸变）
        cv_so3,          // 输出：旋转向量（轴角表示）
        cv_trans,        // 输出：平移向量
        false,           // 是否使用外部初始猜测
        200,             // RANSAC最大迭代次数
        1.5,             // 重投影误差阈值（像素）
        0.99,            // 期望的内点比例（99%置信度）
        status);         // 输出：内点状态向量
  } catch (cv::Exception& e) {
    // === 异常处理 ===
    scope_color(ANSI_COLOR_RED_BOLD);
    std::cout << "Catching a cv exception: " << e.msg << std::endl;
    exit(-1);  // 如果PnP求解出现异常，终止程序
  }

  // === 可选的异常点移除 ===
  if (if_remove_ourlier) {
    // 清空现有的映射，准备重建只包含内点的映射
    map_rgb_points_in_last_image_pose.clear();
    map_rgb_points_in_cur_image_pose.clear();

    // 遍历RANSAC识别的所有内点
    for (unsigned int i = 0; i < status.size(); i++) {
      int inlier_idx = status[i];  // 获取内点的索引
      {
        // 将内点重新添加到上一帧和当前帧的映射中
        // 注意：这里似乎有个逻辑问题，两个映射都使用了当前帧的2D坐标
        map_rgb_points_in_last_image_pose[map_ptr[inlier_idx]] = points_2d[inlier_idx];
        map_rgb_points_in_cur_image_pose[map_ptr[inlier_idx]] = points_2d[inlier_idx];
      }
    }
  }

  // === 更新跟踪数据结构 ===
  // 根据更新后的映射，重新组织跟踪向量和ID
  updateLastTrackingVectorAndIds();

  return true;  // PnP求解成功
}