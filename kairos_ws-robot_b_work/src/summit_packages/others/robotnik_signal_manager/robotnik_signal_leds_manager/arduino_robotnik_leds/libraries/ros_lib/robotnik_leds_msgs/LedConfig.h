#ifndef _ROS_SERVICE_LedConfig_h
#define _ROS_SERVICE_LedConfig_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace robotnik_leds_msgs
{

static const char LEDCONFIG[] = "robotnik_leds_msgs/LedConfig";

  class LedConfigRequest : public ros::Msg
  {
    public:
      typedef const char* _password_type;
      _password_type password;
      typedef const char* _state_type;
      _state_type state;
      typedef const char* _mode_type;
      _mode_type mode;
      typedef uint8_t _color_R_type;
      _color_R_type color_R;
      typedef uint8_t _color_G_type;
      _color_G_type color_G;
      typedef uint8_t _color_B_type;
      _color_B_type color_B;
      typedef uint8_t _color_W_type;
      _color_W_type color_W;
      typedef uint16_t _start_led_type;
      _start_led_type start_led;
      typedef uint16_t _end_led_type;
      _end_led_type end_led;
      typedef uint16_t _ms_on_type;
      _ms_on_type ms_on;
      typedef uint16_t _ms_off_type;
      _ms_off_type ms_off;
      typedef const char* _direction_type;
      _direction_type direction;
      typedef uint16_t _speed_type;
      _speed_type speed;

    LedConfigRequest():
      password(""),
      state(""),
      mode(""),
      color_R(0),
      color_G(0),
      color_B(0),
      color_W(0),
      start_led(0),
      end_led(0),
      ms_on(0),
      ms_off(0),
      direction(""),
      speed(0)
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
      uint32_t length_state = strlen(this->state);
      varToArr(outbuffer + offset, length_state);
      offset += 4;
      memcpy(outbuffer + offset, this->state, length_state);
      offset += length_state;
      uint32_t length_mode = strlen(this->mode);
      varToArr(outbuffer + offset, length_mode);
      offset += 4;
      memcpy(outbuffer + offset, this->mode, length_mode);
      offset += length_mode;
      *(outbuffer + offset + 0) = (this->color_R >> (8 * 0)) & 0xFF;
      offset += sizeof(this->color_R);
      *(outbuffer + offset + 0) = (this->color_G >> (8 * 0)) & 0xFF;
      offset += sizeof(this->color_G);
      *(outbuffer + offset + 0) = (this->color_B >> (8 * 0)) & 0xFF;
      offset += sizeof(this->color_B);
      *(outbuffer + offset + 0) = (this->color_W >> (8 * 0)) & 0xFF;
      offset += sizeof(this->color_W);
      *(outbuffer + offset + 0) = (this->start_led >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->start_led >> (8 * 1)) & 0xFF;
      offset += sizeof(this->start_led);
      *(outbuffer + offset + 0) = (this->end_led >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->end_led >> (8 * 1)) & 0xFF;
      offset += sizeof(this->end_led);
      *(outbuffer + offset + 0) = (this->ms_on >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->ms_on >> (8 * 1)) & 0xFF;
      offset += sizeof(this->ms_on);
      *(outbuffer + offset + 0) = (this->ms_off >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->ms_off >> (8 * 1)) & 0xFF;
      offset += sizeof(this->ms_off);
      uint32_t length_direction = strlen(this->direction);
      varToArr(outbuffer + offset, length_direction);
      offset += 4;
      memcpy(outbuffer + offset, this->direction, length_direction);
      offset += length_direction;
      *(outbuffer + offset + 0) = (this->speed >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->speed >> (8 * 1)) & 0xFF;
      offset += sizeof(this->speed);
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
      uint32_t length_state;
      arrToVar(length_state, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_state; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_state-1]=0;
      this->state = (char *)(inbuffer + offset-1);
      offset += length_state;
      uint32_t length_mode;
      arrToVar(length_mode, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_mode; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_mode-1]=0;
      this->mode = (char *)(inbuffer + offset-1);
      offset += length_mode;
      this->color_R =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->color_R);
      this->color_G =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->color_G);
      this->color_B =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->color_B);
      this->color_W =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->color_W);
      this->start_led =  ((uint16_t) (*(inbuffer + offset)));
      this->start_led |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->start_led);
      this->end_led =  ((uint16_t) (*(inbuffer + offset)));
      this->end_led |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->end_led);
      this->ms_on =  ((uint16_t) (*(inbuffer + offset)));
      this->ms_on |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->ms_on);
      this->ms_off =  ((uint16_t) (*(inbuffer + offset)));
      this->ms_off |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->ms_off);
      uint32_t length_direction;
      arrToVar(length_direction, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_direction; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_direction-1]=0;
      this->direction = (char *)(inbuffer + offset-1);
      offset += length_direction;
      this->speed =  ((uint16_t) (*(inbuffer + offset)));
      this->speed |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->speed);
     return offset;
    }

    const char * getType(){ return LEDCONFIG; };
    const char * getMD5(){ return "0ab3f7f591b547b85c672944179b13ea"; };

  };

  class LedConfigResponse : public ros::Msg
  {
    public:
      typedef const char* _message_type;
      _message_type message;

    LedConfigResponse():
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

    const char * getType(){ return LEDCONFIG; };
    const char * getMD5(){ return "5f003d6bcc824cbd51361d66d8e4f76c"; };

  };

  class LedConfig {
    public:
    typedef LedConfigRequest Request;
    typedef LedConfigResponse Response;
  };

}
#endif
