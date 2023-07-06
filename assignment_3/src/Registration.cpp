#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  PointDistance(Eigen::Vector3d source_point, Eigen::Vector3d target_point)
          : source_pt(source_point), target_pt(target_point) {}

  template<typename T>
  bool operator()(const T *const camera_parameters, T *residuals) const
  {
      // camera_parameters [0,1,2] are the angle-axis rotation.
      T src_pt_transformed[3];
      T src[] = {T(source_pt(0)), T(source_pt(1)), T(source_pt(2))};
      ceres::AngleAxisRotatePoint(camera_parameters, src, src_pt_transformed);
      // camera_parameters [3,4,5] are the translation.
      src_pt_transformed[0] += camera_parameters[3];
      src_pt_transformed[1] += camera_parameters[4];
      src_pt_transformed[2] += camera_parameters[5];

      // Calculate residuals (difference between transformed point and target)
      residuals[0] = src_pt_transformed[0] - T(target_pt(0));
      residuals[1] = src_pt_transformed[1] - T(target_pt(1));
      residuals[2] = src_pt_transformed[2] - T(target_pt(2));
      return true;
  }

  // Factory to hide the construction of the CostFunction object from the client code
  static ceres::CostFunction *Create(const Eigen::Vector3d &source_point, const Eigen::Vector3d &target_point)
  {
      return (new ceres::AutoDiffCostFunction<PointDistance, 3, 6>(new PointDistance(source_point, target_point)));
  }

  Eigen::Vector3d source_pt;
  Eigen::Vector3d target_pt;
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  double prev_rmse = std::numeric_limits<double>::max();
  for (int iter = 0; iter < max_iteration; iter++)
  {
      std::tuple<std::vector<size_t>, std::vector<size_t>, double> closest_points = find_closest_point(threshold);
      double curr_rmse = get<2>(closest_points);
      std::cout << "Iteration: " << iter << " -> prev_rmse: " << prev_rmse << " - rmse: " << curr_rmse << std::endl;

      // Check convergence criteria
      if (std::fabs(prev_rmse - curr_rmse) <= relative_rmse)
      {
          std::cout << "Iteration: " << iter << "- ICP converged." << std::endl;
          return;
      }
      else
      {
          //std::cout << "Iteration: " << iter << " - ICP NOT converged." << std::endl;
      }
      prev_rmse = curr_rmse;

      // Check the required mode
      Eigen::Matrix4d current_transformation;
      if (mode == "lm")
      {
          current_transformation = get_lm_icp_registration(get<0>(closest_points), get<1>(closest_points));
      }
      else if (mode == "svd")
      {
          current_transformation = get_svd_icp_transformation(get<0>(closest_points), get<1>(closest_points));
      }
      else
      {
          std::cout << "INVALID mode" << std::endl;
          return;
      }
      // Update transformed source points
      source_for_icp_.Transform(current_transformation);
      // update transformation_ class variable
      transformation_ *= current_transformation;
  }
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse;

  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  int j = 0;
  // For each source point
  for (int i = 0; i < source_for_icp_.points_.size(); i++)
  {
      // Find the closest one in the target
      source_point = source_for_icp_.points_.at(i);
      target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
      // Discard if their distance is bigger than threshold.
      // Even if taking only the distance ("dist2.at(0)"), the result are basically the same
      if (pow(dist2.at(0), 2) <= threshold)
      {
          source_indices.push_back(i);
          target_indices.push_back(idx.at(0));
          rmse = rmse * j / (j + 1) + dist2.at(0) / (j + 1);
          j++;
      }
  }
  rmse = sqrt(rmse);
  // std::cout << "Source point: " << source_for_icp_.points_.size() << " - Inlier: " << source_indices.size() << std::endl;
  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  open3d::geometry::PointCloud source_clone = source_for_icp_;
  open3d::geometry::PointCloud target_clone = target_;

  // Calculate centroids of the point clouds
  Eigen::Vector3d source_centroid = source_clone.GetCenter();
  Eigen::Vector3d target_centroid = target_clone.GetCenter();

  // Subtract centroids from the point clouds
  for (Eigen::Vector3d &point: source_clone.points_)
  {
      point -= source_centroid;
  }
  for (Eigen::Vector3d &point: target_clone.points_)
  {
      point -= target_centroid;
  }

  // Multiply the point matrices obtained after subtracting centroids
  Eigen::Matrix3d orderedMatrix = Eigen::Matrix3d::Zero();
  for (int i = 0; i < source_indices.size(); i++)
  {
      orderedMatrix += target_clone.points_.at(target_indices.at(i)) *
                       source_clone.points_.at(source_indices.at(i)).transpose();
  }

  // Perform SVD decomposition
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(orderedMatrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd U = svd.matrixU();
  Eigen::MatrixXd V = svd.matrixV();

  // Handle the special reflection case -> det(R) = det(U) * det(V)
  if (U.determinant() * V.determinant() < 0)
  {
      // std::cout << "reflection case" << std::endl;
      // std::cout << "Pre U: " << U << std::endl;
      U.col(2) *= -1;
      // std::cout << "Post U: " << U << std::endl;
  }

  // Calculate the rotation matrix
  Eigen::Matrix3d R = U * V.transpose();

  // Calculate the translation vector
  Eigen::Vector3d T = target_centroid - R * source_centroid;

  // Update the local transformation matrix
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = T;

  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  ceres::Problem problem;
  ceres::Solver::Summary summary;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();
  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
      // Add a residual block inside the Ceres solver problem.
      Eigen::Vector3d src = source_for_icp_.points_.at(source_indices.at(i));
      Eigen::Vector3d target = target_.points_.at(target_indices.at(i));
      ceres::CostFunction *cost_function = PointDistance::Create(src, target);
      problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data());
  }

  // Solve this problem
  ceres::Solve(options, &problem, &summary);

  // Calculate the rotation matrix
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(transformation_arr.at(0), Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(transformation_arr.at(1), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(transformation_arr.at(2), Eigen::Vector3d::UnitZ());

  // Calculate the translation vector
  Eigen::Vector3d T(transformation_arr.at(3), transformation_arr.at(4), transformation_arr.at(5));

  // Update the local transformation matrix
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = T;

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


