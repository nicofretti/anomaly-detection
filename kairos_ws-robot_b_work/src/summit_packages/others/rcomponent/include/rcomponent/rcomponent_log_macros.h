#ifndef _RCOMPONENT_LOG_
#define _RCOMPONENT_LOG_

// debug
#define RCOMPONENT_DEBUG(format_string, ...)                                                                           \
  ROS_DEBUG("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_NAMED(name, format_string, ...)                                                               \
  ROS_DEBUG_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,            \
                  ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_COND(cond, format_string, ...)                                                                \
  ROS_DEBUG_COND(cond, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,             \
                 ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_COND_NAMED(cond, name, format_string, ...)                                                    \
  ROS_DEBUG_COND_NAMED(cond, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, \
                       ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_ONCE(format_string, ...)                                                                      \
  ROS_DEBUG_ONCE("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_ONCE_NAMED(name, format_string, ...)                                                          \
  ROS_DEBUG_ONCE_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,       \
                       ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_THROTTLE(rate, format_string, ...)                                                            \
  ROS_DEBUG_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,         \
                     ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_THROTTLE_NAMED(rate, name, format_string, ...)                                                \
  ROS_DEBUG_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,       \
                           __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_DELAYED_THROTTLE(rate, format_string, ...)                                                    \
  ROS_DEBUG_DELAYED_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, \
                             ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_DELAYED_THROTTLE_NAMED(rate, name, format_string, ...)                                        \
  ROS_DEBUG_DELAYED_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(),             \
                                   __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_FILTER(filter, format_string, ...)                                                            \
  ROS_DEBUG_FILTER(filter, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,         \
                   ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_FILTER_NAMED(filter, name, format_string, ...)                                                \
  ROS_DEBUG_FILTER_NAMED(filter, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,       \
                         __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_DEBUG_STREAM(args)                                                                                  \
  ROS_DEBUG_STREAM(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_NAMED(name, args)                                                                      \
  ROS_DEBUG_STREAM_NAMED(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_COND(cond, args)                                                                       \
  ROS_DEBUG_STREAM_COND(cond, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_COND_NAMED(cond, name, args)                                                           \
  ROS_DEBUG_STREAM_COND_NAMED(cond, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "     \
                                                               << args)

#define RCOMPONENT_DEBUG_STREAM_ONCE(args)                                                                             \
  ROS_DEBUG_STREAM_ONCE(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_ONCE_NAMED(name, args)                                                                 \
  ROS_DEBUG_STREAM_ONCE_NAMED(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_THROTTLE(rate, args)                                                                   \
  ROS_DEBUG_STREAM_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_THROTTLE_NAMED(rate, name, args)                                                       \
  ROS_DEBUG_STREAM_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " \
                                                                   << args)
#define RCOMPONENT_DEBUG_STREAM_DELAYED_THROTTLE(rate, args)                                                           \
  define ROS_DEBUG_STREAM_DELAYED_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__      \
                                                                      << ": " << args)
#define RCOMPONENT_DEBUG_STREAM_DELAYED_THROTTLE_NAMED(rate, name, args)                                               \
  ROS_DEBUG_STREAM_DELAYED_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ \
                                                                           << ": " << args)
#define RCOMPONENT_DEBUG_STREAM_FILTER(filter, args)                                                                   \
  ROS_DEBUG_STREAM_FILTER(filter, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_DEBUG_STREAM_FILTER_NAMED(filter, name, args)                                                       \
  ROS_DEBUG_STREAM_FILTER_NAMED(filter, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " \
                                                                   << args)

// info
#define RCOMPONENT_INFO(format_string, ...)                                                                            \
  ROS_INFO("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_INFO_NAMED(name, format_string, ...)                                                                \
  ROS_INFO_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,             \
                 ##__VA_ARGS__)

#define RCOMPONENT_INFO_COND(cond, format_string, ...)                                                                 \
  ROS_INFO_COND(cond, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_INFO_COND_NAMED(cond, name, format_string, ...)                                                     \
  ROS_INFO_COND_NAMED(cond, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,  \
                      ##__VA_ARGS__)

#define RCOMPONENT_INFO_ONCE(format_string, ...)                                                                       \
  ROS_INFO_ONCE("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_INFO_ONCE_NAMED(name, format_string, ...)                                                           \
  ROS_INFO_ONCE_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,        \
                      ##__VA_ARGS__)

#define RCOMPONENT_INFO_THROTTLE(rate, format_string, ...)                                                             \
  ROS_INFO_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,          \
                    ##__VA_ARGS__)

#define RCOMPONENT_INFO_THROTTLE_NAMED(rate, name, format_string, ...)                                                 \
  ROS_INFO_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,        \
                          __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_INFO_DELAYED_THROTTLE(rate, format_string, ...)                                                     \
  ROS_INFO_DELAYED_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,  \
                            ##__VA_ARGS__)

#define RCOMPONENT_INFO_DELAYED_THROTTLE_NAMED(rate, name, format_string, ...)                                         \
  ROS_INFO_DELAYED_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(),              \
                                  __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_INFO_FILTER(filter, format_string, ...)                                                             \
  ROS_INFO_FILTER(filter, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,          \
                  ##__VA_ARGS__)

#define RCOMPONENT_INFO_FILTER_NAMED(filter, name, format_string, ...)                                                 \
  ROS_INFO_FILTER_NAMED(filter, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,        \
                        __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_INFO_STREAM(args)                                                                                   \
  ROS_INFO_STREAM(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_NAMED(name, args)                                                                       \
  ROS_INFO_STREAM_NAMED(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_COND(cond, args)                                                                        \
  ROS_INFO_STREAM_COND(cond, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_COND_NAMED(cond, name, args)                                                            \
  ROS_INFO_STREAM_COND_NAMED(cond, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "      \
                                                              << args)

#define RCOMPONENT_INFO_STREAM_ONCE(args)                                                                              \
  ROS_INFO_STREAM_ONCE(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_ONCE_NAMED                                                                              \
  (name, args)                                                                                                         \
      ROS_INFO_STREAM_ONCE(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_THROTTLE(rate, args)                                                                    \
  ROS_INFO_STREAM_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_THROTTLE_NAMED(rate, name, args)                                                        \
  ROS_INFO_STREAM_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "  \
                                                                  << args)
#define RCOMPONENT_INFO_STREAM_DELAYED_THROTTLE(rate, args)                                                            \
  define ROS_INFO_STREAM_DELAYED_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__       \
                                                                     << ": " << args)
#define RCOMPONENT_INFO_STREAM_DELAYED_THROTTLE_NAMED(rate, name, args)                                                \
  ROS_INFO_STREAM_DELAYED_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__  \
                                                                          << ": " << args)
#define RCOMPONENT_INFO_STREAM_FILTER(filter, args)                                                                    \
  ROS_INFO_STREAM_FILTER(filter, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_INFO_STREAM_FILTER_NAMED(filter, name, args)                                                        \
  ROS_INFO_STREAM_FILTER_NAMED(filter, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "  \
                                                                  << args)

// warn
#define RCOMPONENT_WARN(format_string, ...)                                                                            \
  ROS_WARN("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_WARN_NAMED(name, format_string, ...)                                                                \
  ROS_WARN_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,             \
                 ##__VA_ARGS__)

#define RCOMPONENT_WARN_COND(cond, format_string, ...)                                                                 \
  ROS_WARN_COND(cond, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_WARN_COND_NAMED(cond, name, format_string, ...)                                                     \
  ROS_WARN_COND_NAMED(cond, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,  \
                      ##__VA_ARGS__)

#define RCOMPONENT_WARN_ONCE(format_string, ...)                                                                       \
  ROS_WARN_ONCE("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_WARN_ONCE_NAMED(name, format_string, ...)                                                           \
  ROS_WARN_ONCE_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,        \
                      ##__VA_ARGS__)

#define RCOMPONENT_WARN_THROTTLE(rate, format_string, ...)                                                             \
  ROS_WARN_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,          \
                    ##__VA_ARGS__)

#define RCOMPONENT_WARN_THROTTLE_NAMED(rate, name, format_string, ...)                                                 \
  ROS_WARN_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,        \
                          __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_WARN_DELAYED_THROTTLE(rate, format_string, ...)                                                     \
  ROS_WARN_DELAYED_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,  \
                            ##__VA_ARGS__)

#define RCOMPONENT_WARN_DELAYED_THROTTLE_NAMED(rate, name, format_string, ...)                                         \
  ROS_WARN_DELAYED_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(),              \
                                  __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_WARN_FILTER(filter, format_string, ...)                                                             \
  ROS_WARN_FILTER(filter, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,          \
                  ##__VA_ARGS__)

#define RCOMPONENT_WARN_FILTER_NAMED(filter, name, format_string, ...)                                                 \
  ROS_WARN_FILTER_NAMED(filter, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,        \
                        __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_WARN_STREAM(args)                                                                                   \
  ROS_WARN_STREAM(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_NAMED(name, args)                                                                       \
  ROS_WARN_STREAM_NAMED(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_COND(cond, args)                                                                        \
  ROS_WARN_STREAM_COND(cond, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_COND_NAMED(cond, name, args)                                                            \
  ROS_WARN_STREAM_COND_NAMED(cond, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "      \
                                                              << args)

#define RCOMPONENT_WARN_STREAM_ONCE(args)                                                                              \
  ROS_WARN_STREAM_ONCE(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_ONCE_NAMED                                                                              \
  (name, args)                                                                                                         \
      ROS_WARN_STREAM_ONCE(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_THROTTLE(rate, args)                                                                    \
  ROS_WARN_STREAM_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_THROTTLE_NAMED(rate, name, args)                                                        \
  ROS_WARN_STREAM_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "  \
                                                                  << args)
#define RCOMPONENT_WARN_STREAM_DELAYED_THROTTLE(rate, args)                                                            \
  define ROS_WARN_STREAM_DELAYED_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__       \
                                                                     << ": " << args)
#define RCOMPONENT_WARN_STREAM_DELAYED_THROTTLE_NAMED(rate, name, args)                                                \
  ROS_WARN_STREAM_DELAYED_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__  \
                                                                          << ": " << args)
#define RCOMPONENT_WARN_STREAM_FILTER(filter, args)                                                                    \
  ROS_WARN_STREAM_FILTER(filter, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_WARN_STREAM_FILTER_NAMED(filter, name, args)                                                        \
  ROS_WARN_STREAM_FILTER_NAMED(filter, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "  \
                                                                  << args)

// error
#define RCOMPONENT_ERROR(format_string, ...)                                                                           \
  ROS_ERROR("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_ERROR_NAMED(name, format_string, ...)                                                               \
  ROS_ERROR_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,            \
                  ##__VA_ARGS__)

#define RCOMPONENT_ERROR_COND(cond, format_string, ...)                                                                \
  ROS_ERROR_COND(cond, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,             \
                 ##__VA_ARGS__)

#define RCOMPONENT_ERROR_COND_NAMED(cond, name, format_string, ...)                                                    \
  ROS_ERROR_COND_NAMED(cond, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, \
                       ##__VA_ARGS__)

#define RCOMPONENT_ERROR_ONCE(format_string, ...)                                                                      \
  ROS_ERROR_ONCE("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_ERROR_ONCE_NAMED(name, format_string, ...)                                                          \
  ROS_ERROR_ONCE_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,       \
                       ##__VA_ARGS__)

#define RCOMPONENT_ERROR_THROTTLE(rate, format_string, ...)                                                            \
  ROS_ERROR_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,         \
                     ##__VA_ARGS__)

#define RCOMPONENT_ERROR_THROTTLE_NAMED(rate, name, format_string, ...)                                                \
  ROS_ERROR_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,       \
                           __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_ERROR_DELAYED_THROTTLE(rate, format_string, ...)                                                    \
  ROS_ERROR_DELAYED_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, \
                             ##__VA_ARGS__)

#define RCOMPONENT_ERROR_DELAYED_THROTTLE_NAMED(rate, name, format_string, ...)                                        \
  ROS_ERROR_DELAYED_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(),             \
                                   __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_ERROR_FILTER(filter, format_string, ...)                                                            \
  ROS_ERROR_FILTER(filter, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,         \
                   ##__VA_ARGS__)

#define RCOMPONENT_ERROR_FILTER_NAMED(filter, name, format_string, ...)                                                \
  ROS_ERROR_FILTER_NAMED(filter, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,       \
                         __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_ERROR_STREAM(args)                                                                                  \
  ROS_ERROR_STREAM(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_NAMED(name, args)                                                                      \
  ROS_ERROR_STREAM_NAMED(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_COND(cond, args)                                                                       \
  ROS_ERROR_STREAM_COND(cond, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_COND_NAMED(cond, name, args)                                                           \
  ROS_ERROR_STREAM_COND_NAMED(cond, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "     \
                                                               << args)

#define RCOMPONENT_ERROR_STREAM_ONCE(args)                                                                             \
  ROS_ERROR_STREAM_ONCE(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_ONCE_NAMED                                                                             \
  (name, args)                                                                                                         \
      ROS_ERROR_STREAM_ONCE(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_THROTTLE(rate, args)                                                                   \
  ROS_ERROR_STREAM_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_THROTTLE_NAMED(rate, name, args)                                                       \
  ROS_ERROR_STREAM_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " \
                                                                   << args)
#define RCOMPONENT_ERROR_STREAM_DELAYED_THROTTLE(rate, args)                                                           \
  define ROS_ERROR_STREAM_DELAYED_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__      \
                                                                      << ": " << args)
#define RCOMPONENT_ERROR_STREAM_DELAYED_THROTTLE_NAMED(rate, name, args)                                               \
  ROS_ERROR_STREAM_DELAYED_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ \
                                                                           << ": " << args)
#define RCOMPONENT_ERROR_STREAM_FILTER(filter, args)                                                                   \
  ROS_ERROR_STREAM_FILTER(filter, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_ERROR_STREAM_FILTER_NAMED(filter, name, args)                                                       \
  ROS_ERROR_STREAM_FILTER_NAMED(filter, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " \
                                                                   << args)

// fatal
#define RCOMPONENT_FATAL(format_string, ...)                                                                           \
  ROS_FATAL("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_FATAL_NAMED(name, format_string, ...)                                                               \
  ROS_FATAL_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,            \
                  ##__VA_ARGS__)

#define RCOMPONENT_FATAL_COND(cond, format_string, ...)                                                                \
  ROS_FATAL_COND(cond, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,             \
                 ##__VA_ARGS__)

#define RCOMPONENT_FATAL_COND_NAMED(cond, name, format_string, ...)                                                    \
  ROS_FATAL_COND_NAMED(cond, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, \
                       ##__VA_ARGS__)

#define RCOMPONENT_FATAL_ONCE(format_string, ...)                                                                      \
  ROS_FATAL_ONCE("%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_FATAL_ONCE_NAMED(name, format_string, ...)                                                          \
  ROS_FATAL_ONCE_NAMED(name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,       \
                       ##__VA_ARGS__)

#define RCOMPONENT_FATAL_THROTTLE(rate, format_string, ...)                                                            \
  ROS_FATAL_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,         \
                     ##__VA_ARGS__)

#define RCOMPONENT_FATAL_THROTTLE_NAMED(rate, name, format_string, ...)                                                \
  ROS_FATAL_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,       \
                           __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_FATAL_DELAYED_THROTTLE(rate, format_string, ...)                                                    \
  ROS_FATAL_DELAYED_THROTTLE(rate, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__, \
                             ##__VA_ARGS__)

#define RCOMPONENT_FATAL_DELAYED_THROTTLE_NAMED(rate, name, format_string, ...)                                        \
  ROS_FATAL_DELAYED_THROTTLE_NAMED(rate, name, "%s::%s::%d: " format_string, this->component_name.c_str(),             \
                                   __FUNCTION__, __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_FATAL_FILTER(filter, format_string, ...)                                                            \
  ROS_FATAL_FILTER(filter, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__, __LINE__,         \
                   ##__VA_ARGS__)

#define RCOMPONENT_FATAL_FILTER_NAMED(filter, name, format_string, ...)                                                \
  ROS_FATAL_FILTER_NAMED(filter, name, "%s::%s::%d: " format_string, this->component_name.c_str(), __FUNCTION__,       \
                         __LINE__, ##__VA_ARGS__)

#define RCOMPONENT_FATAL_STREAM(args)                                                                                  \
  ROS_FATAL_STREAM(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_NAMED(name, args)                                                                      \
  ROS_FATAL_STREAM_NAMED(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_COND(cond, args)                                                                       \
  ROS_FATAL_STREAM_COND(cond, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_COND_NAMED(cond, name, args)                                                           \
  ROS_FATAL_STREAM_COND_NAMED(cond, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": "     \
                                                               << args)

#define RCOMPONENT_FATAL_STREAM_ONCE(args)                                                                             \
  ROS_FATAL_STREAM_ONCE(this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_ONCE_NAMED                                                                             \
  (name, args)                                                                                                         \
      ROS_FATAL_STREAM_ONCE(name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_THROTTLE(rate, args)                                                                   \
  ROS_FATAL_STREAM_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_THROTTLE_NAMED(rate, name, args)                                                       \
  ROS_FATAL_STREAM_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " \
                                                                   << args)
#define RCOMPONENT_FATAL_STREAM_DELAYED_THROTTLE(rate, args)                                                           \
  define ROS_FATAL_STREAM_DELAYED_THROTTLE(rate, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__      \
                                                                      << ": " << args)
#define RCOMPONENT_FATAL_STREAM_DELAYED_THROTTLE_NAMED(rate, name, args)                                               \
  ROS_FATAL_STREAM_DELAYED_THROTTLE_NAMED(rate, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ \
                                                                           << ": " << args)
#define RCOMPONENT_FATAL_STREAM_FILTER(filter, args)                                                                   \
  ROS_FATAL_STREAM_FILTER(filter, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " << args)

#define RCOMPONENT_FATAL_STREAM_FILTER_NAMED(filter, name, args)                                                       \
  ROS_FATAL_STREAM_FILTER_NAMED(filter, name, this->component_name << "::" << __FUNCTION__ << "::" << __LINE__ << ": " \
                                                                   << args)
#endif  // _RCOMPONENT_LOG_
