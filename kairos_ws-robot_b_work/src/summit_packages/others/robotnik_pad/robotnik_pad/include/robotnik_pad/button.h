#ifndef _BUTTON_
#define _BUTTON_

//! Class to save the state of the buttons
class Button
{
  int is_pressed_;
  bool is_released_;

public:
  Button()
  {
    is_pressed_ = 0;
    is_released_ = false;
  }

  //! Set the button as 'pressed'/'released'
  void press(int value)
  {
    if (is_pressed_ and !value)
    {
      is_released_ = true;
    }
    else if (is_released_ and value)
      is_released_ = false;

    is_pressed_ = value;
  }

  int isPressed() const
  {
    return is_pressed_;
  }

  bool isReleased() const
  {
    return is_released_;
  }

  void resetReleased()
  {
    if (is_released_)
      is_released_ = false;
  }
};

#endif  // _BUTTON_