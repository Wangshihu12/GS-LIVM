#include "liw/rgbMapTracker.h"

rgbMapTracker::rgbMapTracker() {
  std::vector<point3D> v_point_temp;
  state* p_state_ = new state();
  p_cloud_frame = new cloudFrame(v_point_temp, p_state_);

  minimum_depth_for_projection = 0.1;
  maximum_depth_for_projection = 200;

  recent_visited_voxel_activated_time = 1.0;

  number_of_new_visited_voxel = 0;

  updated_frame_index = -1;

  in_appending_points = false;

  points_rgb_vec_for_projection = nullptr;

  mutex_rgb_points_vec = std::make_shared<std::mutex>();
  mutex_frame_index = std::make_shared<std::mutex>();
}

/**
 * 刷新用于投影的点集合
 * 根据当前帧的位姿和视野，重新选择适合进行图像投影和视觉跟踪的3D点集合
 * @param map 体素哈希地图引用，包含所有3D点数据
 */
void rgbMapTracker::refreshPointsForProjection(voxelHashMap& map) {
  // === 获取当前帧引用 ===
  cloudFrame* p_frame = p_cloud_frame;

  // === 检查图像有效性 ===
  // 如果图像尺寸无效（宽度或高度为0），则无法进行投影，直接返回
  if (p_frame->image_cols == 0 || p_frame->image_rows == 0)
    return;

  // === 检查是否需要更新 ===
  // 如果当前帧ID与上次更新的帧ID相同，说明已经为此帧更新过了
  // 避免重复计算，直接返回
  if (p_frame->frame_id == updated_frame_index)
    return;

  // === 创建临时点集合 ===
  // 创建新的临时向量来存储筛选出的RGB点指针
  // 使用动态分配，便于后续的指针交换操作
  std::vector<rgbPoint*>* points_rgb_vec_for_projection_temp = new std::vector<rgbPoint*>();

  // === 选择适合投影的点 ===
  // 从体素地图中选择适合当前视角投影的3D点
  // 参数说明：
  // - map: 输入的体素哈希地图
  // - p_frame: 当前帧信息（包含相机位姿和内参）
  // - points_rgb_vec_for_projection_temp: 输出的3D点集合
  // - nullptr: 不需要输出2D投影点（第四个参数）
  // - 10.0: 最小距离阈值，用于点的稀疏化
  // - 1: 跳过步长，1表示不跳过任何点
  selectPointsForProjection(map, p_frame, points_rgb_vec_for_projection_temp, nullptr, 10.0, 1);

  // === 更新全局点集合指针 ===
  // 将新选择的点集合赋值给全局变量
  // 这样其他组件就可以使用最新的投影点集合了
  points_rgb_vec_for_projection = points_rgb_vec_for_projection_temp;

  // === 线程安全更新帧索引 ===
  // 使用互斥锁保护帧索引的更新，确保多线程安全
  mutex_frame_index->lock();
  
  // 记录已处理的帧ID，避免重复处理同一帧
  updated_frame_index = p_frame->frame_id;
  
  // 释放互斥锁
  mutex_frame_index->unlock();
}

/**
 * 选择用于投影的点集
 * 从3D点云中选择可以投影到当前图像的点，用于特征跟踪和视觉里程计
 * @param map 体素哈希地图引用
 * @param p_frame 当前点云帧指针
 * @param pc_out_vec 输出的3D点指针向量
 * @param pc_2d_out_vec 输出的2D投影点向量
 * @param minimum_dis 最小距离阈值（用于点之间的距离过滤）
 * @param skip_step 跳过步长（用于稀疏采样）
 * @param use_all_points 是否使用所有点（否则只使用最近访问的点）
 */
void rgbMapTracker::selectPointsForProjection(
  voxelHashMap& map,
  cloudFrame* p_frame,
  std::vector<rgbPoint*>* pc_out_vec,
  std::vector<cv::Point2f>* pc_2d_out_vec,
  double minimum_dis,
  int skip_step,
  bool use_all_points) {

// === 初始化输出容器 ===
// 清空输出向量，准备存储新的结果
if (pc_out_vec != nullptr) {
  pc_out_vec->clear();
}

if (pc_2d_out_vec != nullptr) {
  pc_2d_out_vec->clear();
}

// === 创建2D哈希映射用于掩码和深度缓冲 ===
// 用于记录已处理的像素位置，避免重复处理
Hash_map_2d<int, int> mask_index;   // 记录每个像素位置对应的点索引
Hash_map_2d<int, float> mask_depth; // 记录每个像素位置的深度值

// === 创建绘制中心映射 ===
// 存储点索引到2D像素坐标的映射
std::map<int, cv::Point2f> map_idx_draw_center;         // 量化后的像素位置
std::map<int, cv::Point2f> map_idx_draw_center_raw_pose; // 原始的像素位置

// === 像素坐标变量 ===
int u, v;          // 整数像素坐标
double u_f, v_f;   // 浮点像素坐标

// === 统计变量 ===
int acc = 0;     // 接受的点数
int blk_rej = 0; // 因为遮挡而被拒绝的点数

// === 准备投影用的点集 ===
std::vector<rgbPoint*> points_for_projection;

// 获取最近访问的体素列表
std::vector<voxelId> boxes_recent_hitted = voxels_recent_visited;

// === 选择点集策略 ===
if ((!use_all_points) && boxes_recent_hitted.size()) {
  // 策略1：只使用最近访问的体素中的点（效率优化）
  for (std::vector<voxelId>::iterator it = boxes_recent_hitted.begin(); 
       it != boxes_recent_hitted.end(); it++) {
    // 检查体素中是否有点，如果有则选择最新的点
    if (map[voxel((*it).kx, (*it).ky, (*it).kz)].NumPoints() > 0) {
      points_for_projection.push_back(&(map[voxel((*it).kx, (*it).ky, (*it).kz)].points.back()));
    }
  }
} else {
  // 策略2：使用所有RGB点（完整但可能较慢）
  mutex_rgb_points_vec->lock();
  points_for_projection = rgb_points_vec;
  mutex_rgb_points_vec->unlock();
}

int point_size = points_for_projection.size();

// === 遍历所有待投影的点 ===
for (int point_index = 0; point_index < point_size; point_index += skip_step) {
  // 获取3D点的世界坐标
  Eigen::Vector3d point_world = points_for_projection[point_index]->getPosition();

  // === 深度检查 ===
  // 计算点到相机的距离
  double depth = (point_world - p_frame->p_state->t_world_camera).norm();

  // 跳过距离过远的点
  if (depth > maximum_depth_for_projection) {
    continue;
  }

  // 跳过距离过近的点
  if (depth < minimum_depth_for_projection) {
    continue;
  }

  // === 3D到2D投影 ===
  // 将3D点投影到当前图像平面
  bool res = p_frame->project3dPointInThisImage(point_world, u_f, v_f, nullptr, 1.0);

  // 如果投影失败（点不在视野内），跳过此点
  if (res == false) {
    continue;
  }

  // === 像素坐标量化 ===
  // 将浮点坐标量化到最小距离的整数倍上，用于去重
  u = std::round(u_f / minimum_dis) * minimum_dis;
  v = std::round(v_f / minimum_dis) * minimum_dis;

  // === 深度缓冲处理 ===
  // 检查该像素位置是否已有点，或者当前点是否更近
  if ((!mask_depth.if_exist(u, v)) || mask_depth.m_map_2d_hash_map[u][v] > depth) {
    acc++; // 接受的点数加1

    // 如果该位置已有点，需要移除旧点
    if (mask_index.if_exist(u, v)) {
      int old_idx = mask_index.m_map_2d_hash_map[u][v];
      blk_rej++; // 被遮挡拒绝的点数加1

      // 从映射中移除旧点的记录
      map_idx_draw_center.erase(map_idx_draw_center.find(old_idx));
      map_idx_draw_center_raw_pose.erase(map_idx_draw_center_raw_pose.find(old_idx));
    }

    // === 更新掩码和映射 ===
    // 记录新的点索引和深度
    mask_index.m_map_2d_hash_map[u][v] = (int)point_index;
    mask_depth.m_map_2d_hash_map[u][v] = (float)depth;

    // 记录点索引到像素坐标的映射
    map_idx_draw_center[point_index] = cv::Point2f(v, u);              // 量化后的坐标
    map_idx_draw_center_raw_pose[point_index] = cv::Point2f(u_f, v_f); // 原始坐标
  }
}

// === 输出结果 ===
// 将选中的3D点指针添加到输出向量
if (pc_out_vec != nullptr) {
  for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
    pc_out_vec->push_back(points_for_projection[it->first]);
}

// 将对应的2D投影点添加到输出向量
if (pc_2d_out_vec != nullptr) {
  for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++)
    pc_2d_out_vec->push_back(map_idx_draw_center_raw_pose[it->first]);
}
}

void rgbMapTracker::updatePoseForProjection(cloudFrame* p_frame, double fov_margin) {
  p_cloud_frame->p_state->fx = p_frame->p_state->fx;
  p_cloud_frame->p_state->fy = p_frame->p_state->fy;
  p_cloud_frame->p_state->cx = p_frame->p_state->cx;
  p_cloud_frame->p_state->cy = p_frame->p_state->cy;

  p_cloud_frame->image_cols = p_frame->image_cols;
  p_cloud_frame->image_rows = p_frame->image_rows;

  p_cloud_frame->p_state->fov_margin = fov_margin;
  p_cloud_frame->frame_id = p_frame->frame_id;

  p_cloud_frame->p_state->q_world_camera = p_frame->p_state->q_world_camera;
  p_cloud_frame->p_state->t_world_camera = p_frame->p_state->t_world_camera;

  p_cloud_frame->rgb_image = p_frame->rgb_image;
  p_cloud_frame->gray_image = p_frame->gray_image;

  p_cloud_frame->refreshPoseForProjection();
}

const double image_obs_cov = 15;
const double process_noise_sigma = 0.1;

std::atomic<long> render_point_count;

/**
 * 多线程渲染体素中的点函数
 * 将指定范围内体素中的3D点投影到当前图像，并更新其RGB颜色信息
 * @param map 体素哈希地图引用
 * @param voxel_start 要处理的体素起始索引
 * @param voxel_end 要处理的体素结束索引（不包含）
 * @param p_frame 当前点云帧指针（包含图像和位姿信息）
 * @param voxels_for_render 需要渲染的体素ID列表指针
 * @param obs_time 观测时间戳
 */
void rgbMapTracker::threadRenderPointsInVoxel(
  voxelHashMap& map,
  const int& voxel_start,
  const int& voxel_end,
  cloudFrame* p_frame,
  const std::vector<voxelId>* voxels_for_render,
  const double obs_time) {

// === 声明局部变量 ===
Eigen::Vector3d point_world;     // 3D点的世界坐标
Eigen::Vector3d point_color;     // 从图像中采样得到的RGB颜色

double u, v;                     // 2D图像坐标（像素位置）
double point_camera_norm;        // 点到相机的距离（深度）

// === 遍历分配给当前线程的体素范围 ===
for (int voxel_index = voxel_start; voxel_index < voxel_end; voxel_index++) {
  // 根据体素ID获取对应的体素块引用
  voxelBlock& voxel_block = map[voxel(
      (*voxels_for_render)[voxel_index].kx,  // 体素x坐标
      (*voxels_for_render)[voxel_index].ky,  // 体素y坐标
      (*voxels_for_render)[voxel_index].kz)]; // 体素z坐标

  // === 遍历当前体素块中的所有点 ===
  for (int point_index = 0; point_index < voxel_block.NumPoints(); point_index++) {
    // 获取当前点的引用
    auto& point = voxel_block.points[point_index];

    // 获取点在世界坐标系下的位置
    point_world = point.getPosition();

    // === 3D点投影到2D图像平面 ===
    // 尝试将3D点投影到当前图像上，获取像素坐标(u,v)
    if (p_frame->project3dPointInThisImage(point_world, u, v, nullptr, 1.0) == false)
      continue;  // 如果投影失败（点不在视野内），跳过此点

    // === 计算观测距离 ===
    // 计算3D点到相机中心的欧几里得距离，用于观测质量评估
    point_camera_norm = (point_world - p_frame->p_state->t_world_camera).norm();

    // === 从图像中采样RGB颜色 ===
    // 在投影位置(u,v)处从图像中获取RGB颜色值
    // 第三个参数0表示使用原图像层级（非金字塔层级）
    point_color = p_frame->getRgb(u, v, 0);

    // === 线程安全的RGB更新 ===
    // 使用互斥锁保护共享资源，确保多线程安全
    mutex_rgb_points_vec->lock();
    
    // 使用卡尔曼滤波更新点的RGB信息
    // 参数说明：
    // - point_color: 当前观测到的RGB颜色
    // - point_camera_norm: 观测距离（用于权重计算）
    // - Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov): 观测噪声协方差
    // - obs_time: 观测时间戳
    // 返回值：如果成功更新RGB信息则返回非零值
    if (voxel_block.points[point_index].updateRgb(
            point_color, 
            point_camera_norm, 
            Eigen::Vector3d(image_obs_cov, image_obs_cov, image_obs_cov), 
            obs_time)) {
      render_point_count++;  // 成功渲染的点数计数器递增
    }
    
    // 释放互斥锁
    mutex_rgb_points_vec->unlock();
  }
}
}

std::vector<voxelId> g_voxel_for_render;

/**
 * 在最近访问的体素中渲染点的主控函数
 * 使用多线程并行处理最近访问的体素，为其中的3D点更新RGB颜色信息
 * @param map 体素哈希地图引用
 * @param p_frame 当前点云帧指针（包含图像和位姿信息）
 * @param voxels_for_render 需要渲染的体素ID列表指针
 * @param obs_time 观测时间戳引用
 */
void rgbMapTracker::renderPointsInRecentVoxel(
  voxelHashMap& map,
  cloudFrame* p_frame,
  std::vector<voxelId>* voxels_for_render,
  const double& obs_time) {

// === 清空并准备全局渲染体素列表 ===
// 清空全局体素渲染列表，为新的渲染任务做准备
g_voxel_for_render.clear();

// 释放内存并重新分配，确保内存使用效率
// 这种写法强制释放vector的内存，而clear()只是清空元素但不释放内存
std::vector<voxelId>().swap(g_voxel_for_render);

// === 复制体素列表到全局变量 ===
// 将输入的体素列表复制到全局变量中，便于多线程访问
// 使用全局变量避免在lambda函数中传递大量数据
for (std::vector<voxelId>::iterator it = (*voxels_for_render).begin(); 
     it != (*voxels_for_render).end(); it++) {
  g_voxel_for_render.push_back(*it);
}

// === 准备多线程处理 ===
// 声明存储异步任务结果的向量（虽然在这里没有使用）
std::vector<std::future<double>> results;

// 获取需要处理的体素总数
int number_of_voxels = g_voxel_for_render.size();

// === 初始化渲染统计 ===
// 重置成功渲染的点数计数器
// 这是一个原子变量，用于多线程安全的计数
render_point_count = 0;

// === 执行并行渲染 ===
// 使用OpenCV的并行循环框架进行多线程处理
// cv::parallel_for_会自动将任务分配给多个线程
cv::parallel_for_(cv::Range(0, number_of_voxels), [&](const cv::Range& r) {
  // 调用线程渲染函数处理分配给当前线程的体素范围
  // 参数说明：
  // - map: 体素哈希地图
  // - r.start: 当前线程处理的起始体素索引
  // - r.end: 当前线程处理的结束体素索引（不包含）
  // - p_frame: 当前帧信息
  // - &g_voxel_for_render: 全局体素列表的指针
  // - obs_time: 观测时间戳
  threadRenderPointsInVoxel(map, r.start, r.end, p_frame, &g_voxel_for_render, obs_time);
});
}