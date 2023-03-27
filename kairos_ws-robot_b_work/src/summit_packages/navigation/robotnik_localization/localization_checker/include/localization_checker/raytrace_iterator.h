#ifndef _LOCALIZATION_CHECKER_RAYTRACE_ITERATOR_
#define _LOCALIZATION_CHECKER_RAYTRACE_ITERATOR_

#include <Eigen/Core>
#include <pcl/common/distances.h>

template <typename T>
inline bool sameSign(T one, T two)
{
  // epsilon is required as pcl::PointXYZ is defined by floats, so there are precission errors
  static T epsilon = 1e-5;
  return (one > -epsilon == two > -epsilon) || (one < epsilon == two < epsilon);
}

inline float side(const Eigen::Vector3f& point, const Eigen::Vector3f& plane_origin,
                  const Eigen::Vector3f& plane_normal)
{
  Eigen::Vector3f po = point - plane_origin;
  const Eigen::Vector3f normal = plane_normal;
  return normal.dot(po);
}

template <typename PointT>
class Raytracer
{
public:
  typedef boost::shared_ptr<Raytracer<PointT>> Ptr;

  Raytracer()
  {
  }

  Raytracer(PointT end, double step) : Raytracer(PointT(), end, step)
  {
  }

  Raytracer(PointT start, PointT end, double step)
  {
    init(start, end, step);
  }

  virtual void init(PointT end, double step)
  {
    init(PointT(), end, step);
  }

  virtual void init(PointT start, PointT end, double step)
  {
    start_ = start.getVector3fMap();
    end_ = end.getVector3fMap();
    current_iteration_ = 0;

    current_ = start_;
    Eigen::Vector3f director = end_ - start_;
    direction_ = director.normalized();
    double distance = director.norm();

    iterations_ = std::ceil(distance / step);
    step_ = step;
  }

  virtual void reset()
  {
    current_ = start_;
    current_iteration_ = 0;
  }

  virtual PointT current()
  {
    PointT p;
    p.getVector3fMap() = current_;
    return p;
  }

  virtual PointT next()
  {
    current_iteration_++;
    current_ = nextImpl();
    return current();
  }

  virtual bool hasNext()
  {
    return current_iteration_ < iterations_;
  }

  void print()
  {
    std::cout << "start: " << start_.transpose() << "\n";
    std::cout << "end: " << end_.transpose() << "\n";
    std::cout << "current: " << current_.transpose() << "\n";
    std::cout << "step: " << step_ << "\n";
    std::cout << "iterations: " << iterations_ << "\n";
    std::cout << "current_iteration: " << current_iteration_ << "\n\n";
  }

protected:
  Eigen::Vector3f start_;
  Eigen::Vector3f end_;
  Eigen::Vector3f current_;
  Eigen::Vector3f direction_;

  double step_;
  int current_iteration_;
  int iterations_;

  // helper method. not to be used externally
  inline Eigen::Vector3f nextImpl()
  {
    if (current_iteration_ == iterations_)
      return end_;
    else
      return start_ + direction_ * step_ * current_iteration_;
  }
};

// TODO: add iterator that does not raytrace, but instead goes to the end position
// template <typename PointT>
// class RaytracerNoTrace : public Raytracer<PointT>
//{
// public:
//  void init(PointT start, PointT end, double step)
//  {
//    Raytracer<PointT>::start_ = start.getVector3fMap();
//    Raytracer<PointT>::end_ = start.getVector3fMap();
//    Raytracer<PointT>::current_iteration_ = 0;
//
//    Raytracer<PointT>::start_ = Raytracer<PointT>::start_;
//  }
//
// protected:
//  inline Eigen::Vector3f nextImpl()
//  {
//    if (current_iteration_ == iterations_)
//      return end_;
//    else
//      return start_ + direction_ * step_ * current_iteration_;
//  }
//};

#endif  // _LOCALIZATION_CHECKER_RAYTRACE_ITERATOR_
