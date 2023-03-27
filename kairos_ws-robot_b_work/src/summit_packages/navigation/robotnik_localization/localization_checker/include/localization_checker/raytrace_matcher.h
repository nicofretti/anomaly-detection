#ifndef _LOCALIZATION_CHECKER_RAYTRACE_MATCHER_
#define _LOCALIZATION_CHECKER_RAYTRACE_MATCHER_

#include <pcl/common/distances.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

template <typename PointT>
class RaytraceMatcher
{
public:
  typedef boost::shared_ptr<RaytraceMatcher<PointT>> Ptr;
  RaytraceMatcher()
  {
  }

  RaytraceMatcher(const typename pcl::PointCloud<PointT>::Ptr& map_cloud,
                  const typename pcl::PointCloud<PointT>::Ptr& scan_cloud)
  {
    typename pcl::KdTree<PointT>::Ptr map_tree_(new pcl::KdTreeFLANN<PointT>);
    typename pcl::KdTree<PointT>::Ptr scan_tree_(new pcl::KdTreeFLANN<PointT>);

    map_tree_->setInputCloud(map_cloud);
    scan_tree_->setInputCloud(scan_cloud);

    init();
  }

  RaytraceMatcher(const typename pcl::KdTree<PointT>::Ptr& map_tree, const typename pcl::KdTree<PointT>::Ptr& scan_tree)
  {
    map_tree_ = map_tree;
    scan_tree_ = scan_tree;
    init();
  }

  void setMapTree(const typename pcl::KdTree<PointT>::Ptr& tree)
  {
    map_tree_ = tree;
  }

  void setScanTree(const typename pcl::KdTree<PointT>::Ptr& tree)
  {
    scan_tree_ = tree;
  }

  virtual void init()
  {
  }

  void reset()
  {
    init();
  }

  virtual bool hasFinished()
  {
    return false;
  }

  virtual double score()
  {
    return 0;
  }

  virtual double absoluteScore()
  {
    return 0;
  }

  virtual void match(const PointT& raytrace_point, const PointT& scan_point)
  {
  }

protected:
  typename pcl::KdTree<PointT>::Ptr map_tree_;
  typename pcl::KdTree<PointT>::Ptr scan_tree_;
  double resolution_;
};

template <typename PointT>
class LocalizationLikelihoodMatcher : public RaytraceMatcher<PointT>
{
public:
  typedef boost::shared_ptr<LocalizationLikelihoodMatcher<PointT>> Ptr;
  LocalizationLikelihoodMatcher()
  {
    init();
  }

  LocalizationLikelihoodMatcher(const typename pcl::PointCloud<PointT>::Ptr& map_cloud,
                                const typename pcl::PointCloud<PointT>::Ptr& scan_cloud)
    : RaytraceMatcher<PointT>(map_cloud, scan_cloud)
  {
    init();
  }

  LocalizationLikelihoodMatcher(const typename pcl::KdTree<PointT>::Ptr& map_tree,
                                const typename pcl::KdTree<PointT>::Ptr& scan_tree)
    : RaytraceMatcher<PointT>(map_tree, scan_tree)
  {
    init();
  }

  virtual void init()
  {
    matches_with_map_ = false;
  }

  virtual bool hasFinished()
  {
    return matches_with_map_;
  }

  virtual double score()
  {
    if (matches_with_map_)
      return 1.0;
    return 0.0;
  }

  virtual void match(const PointT& raytrace_point, const PointT& scan_point)
  {
    std::vector<int> nn_indices(1);
    std::vector<float> nn_sqr_dists(1);
    RaytraceMatcher<PointT>::resolution_ = 0.05;
    if (RaytraceMatcher<PointT>::map_tree_->radiusSearch(raytrace_point, RaytraceMatcher<PointT>::resolution_,
                                                         nn_indices, nn_sqr_dists, 1) != 0)
    {
      // std::cout << raytrace_point << " matches with "
      //           << RaytraceMatcher<PointT>::map_tree_->getInputCloud()->points[nn_indices[0]] << " with a distance of
      //           "
      //           << nn_sqr_dists[0] << "\n";
      matches_with_map_ = true;
    }
  }

protected:
  bool matches_with_map_;
};

template <typename PointT>
class LocalizationMatcher : public RaytraceMatcher<PointT>
{
public:
  typedef boost::shared_ptr<LocalizationMatcher<PointT>> Ptr;
  LocalizationMatcher()
  {
    init();
  }
  LocalizationMatcher(const typename pcl::PointCloud<PointT>::Ptr& map_cloud,
                      const typename pcl::PointCloud<PointT>::Ptr& scan_cloud)
    : RaytraceMatcher<PointT>(map_cloud, scan_cloud)
  {
    init();
  }

  LocalizationMatcher(const typename pcl::KdTree<PointT>::Ptr& map_tree,
                      const typename pcl::KdTree<PointT>::Ptr& scan_tree)
    : RaytraceMatcher<PointT>(map_tree, scan_tree)
  {
    init();
  }

  virtual void init()
  {
    matches_with_map_ = false;
    matches_with_scan_ = false;
  }

  virtual bool hasFinished()
  {
    return matches_with_map_;
  }

  virtual double score()
  {
    if (matches_with_map_ and matches_with_scan_)
      return 1.0;
    return 0.0;
    // if (matches_with_map_ and not matches_with_scan_)
    //  return -1.0;
  }

  virtual double absoluteScore()
  {
    if (matches_with_map_ and matches_with_scan_)
      return 1.0;
    return 0.0;
    // if (matches_with_map_ and not matches_with_scan_)
    //  return -1.0;
  }

  virtual void match(const PointT& raytrace_point, const PointT& scan_point)
  {
    if (true == matches_with_map_)
      return;

    std::vector<int> nn_indices(1);
    std::vector<float> nn_sqr_dists(1);
    RaytraceMatcher<PointT>::resolution_ = 0.05;
    if (RaytraceMatcher<PointT>::map_tree_->radiusSearch(raytrace_point, RaytraceMatcher<PointT>::resolution_,
                                                         nn_indices, nn_sqr_dists, 1) != 0)
    {
      // std::cout << raytrace_point << " matches with index " << nn_indices[0] << " which is "
      //          << RaytraceMatcher<PointT>::map_tree_->getInputCloud()->points[nn_indices[0]] << " with a distance of
      //          "
      //          << nn_sqr_dists[0] << "\n";
      matches_with_map_ = true;
    }

    if (true == matches_with_map_)
    {
      if (pcl::euclideanDistance(raytrace_point, scan_point) < RaytraceMatcher<PointT>::resolution_)
        matches_with_scan_ = true;
      else
      {
        // std::cout << "raytrace: " << raytrace_point << " scan_point: " << scan_point << "\n";
        //        std::cout << "Distance between raytrace and scan_point: " << pcl::euclideanDistance(raytrace_point,
        //        scan_point)
        //                << "\n";
      }
    }
  }

protected:
  bool matches_with_map_;
  bool matches_with_scan_;
};

#endif  // _LOCALIZATION_CHECKER_RAYTRACE_MATCHER_
