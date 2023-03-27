#ifndef _ROS_SERVICE_LedEffects_h
#define _ROS_SERVICE_LedEffects_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace robotnik_leds_msgs
{

static const char LEDEFFECTS[] = "robotnik_leds_msgs/LedEffects";

  class LedEffectsRequest : public ros::Msg
  {
    public:
      typedef const char* _id_type;
      _id_type id;
      typedef const char* _mode_type;
      _mode_type mode;
      typedef uint8_t _channel_type;
      _channel_type channel;
      typedef const char* _type_type;
      _type_type type;
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
      typedef uint16_t _fade_in_type;
      _fade_in_type fade_in;
      typedef uint16_t _fade_out_type;
      _fade_out_type fade_out;
      typedef uint8_t _background_R_type;
      _background_R_type background_R;
      typedef uint8_t _background_G_type;
      _background_G_type background_G;
      typedef uint8_t _background_B_type;
      _background_B_type background_B;
      typedef uint8_t _background_W_type;
      _background_W_type background_W;
      typedef const char* _direction_type;
      _direction_type direction;
      typedef uint16_t _speed_type;
      _speed_type speed;
      typedef uint16_t _sleep_type;
      _sleep_type sleep;
      typedef uint16_t _led_increment_type;
      _led_increment_type led_increment;
      typedef uint8_t _layer_type;
      _layer_type layer;
      typedef bool _enabled_type;
      _enabled_type enabled;

    LedEffectsRequest():
      id(""),
      mode(""),
      channel(0),
      type(""),
      color_R(0),
      color_G(0),
      color_B(0),
      color_W(0),
      start_led(0),
      end_led(0),
      ms_on(0),
      ms_off(0),
      fade_in(0),
      fade_out(0),
      background_R(0),
      background_G(0),
      background_B(0),
      background_W(0),
      direction(""),
      speed(0),
      sleep(0),
      led_increment(0),
      layer(0),
      enabled(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      uint32_t length_id = strlen(this->id);
      varToArr(outbuffer + offset, length_id);
      offset += 4;
      memcpy(outbuffer + offset, this->id, length_id);
      offset += length_id;
      uint32_t length_mode = strlen(this->mode);
      varToArr(outbuffer + offset, length_mode);
      offset += 4;
      memcpy(outbuffer + offset, this->mode, length_mode);
      offset += length_mode;
      *(outbuffer + offset + 0) = (this->channel >> (8 * 0)) & 0xFF;
      offset += sizeof(this->channel);
      uint32_t length_type = strlen(this->type);
      varToArr(outbuffer + offset, length_type);
      offset += 4;
      memcpy(outbuffer + offset, this->type, length_type);
      offset += length_type;
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
      *(outbuffer + offset + 0) = (this->fade_in >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->fade_in >> (8 * 1)) & 0xFF;
      offset += sizeof(this->fade_in);
      *(outbuffer + offset + 0) = (this->fade_out >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->fade_out >> (8 * 1)) & 0xFF;
      offset += sizeof(this->fade_out);
      *(outbuffer + offset + 0) = (this->background_R >> (8 * 0)) & 0xFF;
      offset += sizeof(this->background_R);
      *(outbuffer + offset + 0) = (this->background_G >> (8 * 0)) & 0xFF;
      offset += sizeof(this->background_G);
      *(outbuffer + offset + 0) = (this->background_B >> (8 * 0)) & 0xFF;
      offset += sizeof(this->background_B);
      *(outbuffer + offset + 0) = (this->background_W >> (8 * 0)) & 0xFF;
      offset += sizeof(this->background_W);
      uint32_t length_direction = strlen(this->direction);
      varToArr(outbuffer + offset, length_direction);
      offset += 4;
      memcpy(outbuffer + offset, this->direction, length_direction);
      offset += length_direction;
      *(outbuffer + offset + 0) = (this->speed >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->speed >> (8 * 1)) & 0xFF;
      offset += sizeof(this->speed);
      *(outbuffer + offset + 0) = (this->sleep >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->sleep >> (8 * 1)) & 0xFF;
      offset += sizeof(this->sleep);
      *(outbuffer + offset + 0) = (this->led_increment >> (8 * 0)) & 0xFF;
      *(outbuffer + offset + 1) = (this->led_increment >> (8 * 1)) & 0xFF;
      offset += sizeof(this->led_increment);
      *(outbuffer + offset + 0) = (this->layer >> (8 * 0)) & 0xFF;
      offset += sizeof(this->layer);
      union {
        bool real;
        uint8_t base;
      } u_enabled;
      u_enabled.real = this->enabled;
      *(outbuffer + offset + 0) = (u_enabled.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->enabled);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      uint32_t length_id;
      arrToVar(length_id, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_id; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_id-1]=0;
      this->id = (char *)(inbuffer + offset-1);
      offset += length_id;
      uint32_t length_mode;
      arrToVar(length_mode, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_mode; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_mode-1]=0;
      this->mode = (char *)(inbuffer + offset-1);
      offset += length_mode;
      this->channel =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->channel);
      uint32_t length_type;
      arrToVar(length_type, (inbuffer + offset));
      offset += 4;
      for(unsigned int k= offset; k< offset+length_type; ++k){
          inbuffer[k-1]=inbuffer[k];
      }
      inbuffer[offset+length_type-1]=0;
      this->type = (char *)(inbuffer + offset-1);
      offset += length_type;
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
      this->fade_in =  ((uint16_t) (*(inbuffer + offset)));
      this->fade_in |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->fade_in);
      this->fade_out =  ((uint16_t) (*(inbuffer + offset)));
      this->fade_out |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->fade_out);
      this->background_R =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->background_R);
      this->background_G =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->background_G);
      this->background_B =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->background_B);
      this->background_W =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->background_W);
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
      this->sleep =  ((uint16_t) (*(inbuffer + offset)));
      this->sleep |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->sleep);
      this->led_increment =  ((uint16_t) (*(inbuffer + offset)));
      this->led_increment |= ((uint16_t) (*(inbuffer + offset + 1))) << (8 * 1);
      offset += sizeof(this->led_increment);
      this->layer =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->layer);
      union {
        bool real;
        uint8_t base;
      } u_enabled;
      u_enabled.base = 0;
      u_enabled.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->enabled = u_enabled.real;
      offset += sizeof(this->enabled);
     return offset;
    }

    const char * getType(){ return LEDEFFECTS; };
    const char * getMD5(){ return "a4971f1ee1e211ef4482773fdcd58d02"; };

  };

  class LedEffectsResponse : public ros::Msg
  {
    public:
      typedef uint8_t _state_type;
      _state_type state;

    LedEffectsResponse():
      state(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const
    {
      int offset = 0;
      *(outbuffer + offset + 0) = (this->state >> (8 * 0)) & 0xFF;
      offset += sizeof(this->state);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer)
    {
      int offset = 0;
      this->state =  ((uint8_t) (*(inbuffer + offset)));
      offset += sizeof(this->state);
     return offset;
    }

    const char * getType(){ return LEDEFFECTS; };
    const char * getMD5(){ return "800f34bc468def1d86e2d42bea5648c0"; };

  };

  class LedEffects {
    public:
    typedef LedEffectsRequest Request;
    typedef LedEffectsResponse Response;
  };

}
#endif
