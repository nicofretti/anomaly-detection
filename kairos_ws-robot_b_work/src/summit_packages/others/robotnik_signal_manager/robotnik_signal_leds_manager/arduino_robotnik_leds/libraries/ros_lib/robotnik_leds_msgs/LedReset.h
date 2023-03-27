#ifndef _ROS_SERVICE_LedReset_h
#define _ROS_SERVICE_LedReset_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace robotnik_leds_msgs
{

static const char LEDRESET[] = "robotnik_leds_msgs/LedReset";

  class LedResetRequest : public ros::Msg
  {
    public:
      typedef const char* _password_type;
      _password_type password;
      typedef bool _reset_type;
      _reset_type reset;

    LedResetRequest():
      password(""),
      reset(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_password = strlen(this->password);
      varToArr(outbuffer + offset, length_password);
      offset += 4;
      memcpy(outbuffer + offset, this->password, length_password);
      offset += length_password;
      union {
        bool real;
        uint8_t base;
      } u_reset;
      u_reset.real = this->reset;
      *(outbuffer + offset + 0) = (u_reset.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->reset);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_password;
      arrToVar(length_password, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_password; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_password-1]=0;
      this->password = (char *)(inbuffer + offset-1);
      offset += length_password;
      union {
        bool real;
        uint8_t base;
      } u_reset;
      u_reset.base = 0;
      u_reset.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->reset = u_reset.real;
      offset += sizeof(this->reset);
     return offset;
    }

    const char * getType(){ return LEDRESET; };
    const char * getMD5(){ return "38b5eca572637ea335996a677152f609"; };

  };

  class LedResetResponse : public ros::Msg
  {
    public:
      typedef const char* _message_type;
      _message_type message;

    LedResetResponse():
      message("")
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_message = strlen(this->message);
      varToArr(outbuffer + offset, length_message);
      offset += 4;
      memcpy(outbuffer + offset, this->message, length_message);
      offset += length_message;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_message;
      arrToVar(length_message, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_message; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_message-1]=0;
      this->message = (char *)(inbuffer + offset-1);
      offset += length_message;
     return offset;
    }

    const char * getType(){ return LEDRESET; };
    const char * getMD5(){ return "5f003d6bcc824cbd51361d66d8e4f76c"; };

  };

  class LedReset {
    public:
    typedef LedResetRequest Request;
    typedef LedResetResponse Response;
  };

}
#endif
