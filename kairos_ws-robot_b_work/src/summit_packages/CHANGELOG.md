# v2.3.0 (12-11-2021)
- Updates rbkairos_common (Álex)([2394322](https://github.com/RobotnikAutomation/summit_packages/commit/2394322))
  - control: updates speed scaling type controller
  - moveit:
    - Updates urdf file references
    - Increase arm z position to avoid autocollision in moveit 

- Updates command_manager msgs to latestes version (Álex)([c2e62b6](https://github.com/RobotnikAutomation/summit_packages/commit/c2e62b6))
- Updates debs (Álex)([23f3318](https://github.com/RobotnikAutomation/summit_packages/commit/23f3318))
- Move build_system to the root of the project (Álex)([20716fe](https://github.com/RobotnikAutomation/summit_packages/commit/20716fe))
- Updates robot_bringup submodule (Álex)([0c33595](https://github.com/RobotnikAutomation/summit_packages/commit/0c33595))
  - Adds sleep before arm bringup
  - cmd_vel topic as arg in control yaml 
  - Updates priorities for twist mux 
  - Base hw from auto_restart to auto_recovery 
  - Adds new commands to handlers.yaml
  - Update sequences in sequences.yaml
  - Adds yamls for command manager family configuration
  - Adds config ps5 pad
  - Adds intensity filters for tim561 and tim 571 
  - Adds controller to rlc robot_status 
  - Adds HMI envs variables
  - Adds twistmux cmd env variable
  - Adds complex command sequencer and command scheduler
  - Updates hmi launch files to work with new features 
- Updates submodules (Álex)([b13e559](https://github.com/RobotnikAutomation/summit_packages/commit/b13e559))
- robot_bringup: Adds sick_tim dependency (Álex)([c5b4dbb](https://github.com/RobotnikAutomation/summit_packages/commit/c5b4dbb))
- Adds roboticsgroup submodule (Álex)([ccf6d87](https://github.com/RobotnikAutomation/summit_packages/commit/ccf6d87))
- Update Changelog for version v2.2.6 (RomanRobotnik)([0966be1](https://github.com/RobotnikAutomation/summit_packages/commit/0966be1))
- Update robot_bringup (Alejandro Arnal)([cfb66df](https://github.com/RobotnikAutomation/summit_packages/commit/cfb66df))
  - sensors: improve myahrs_imu launch file 
  - Set default env var for gearbox for 48V motors 
- Updates robot_bringup, adds myhars launch file (Alejandro Arnal)([6c0d1a1](https://github.com/RobotnikAutomation/summit_packages/commit/6c0d1a1))

# v2.2.6 (26-10-2021)
- Update robot_bringup (Alejandro Arnal)([cfb66df](https://github.com/RobotnikAutomation/summit_packages/commit/cfb66df))
  - sensors: improve myahrs_imu launch file
  - Set default env var for gearbox for 48V motors 
- Updates robot_bringup, adds myhars launch file (Alejandro Arnal)([6c0d1a1](https://github.com/RobotnikAutomation/summit_packages/commit/6c0d1a1))
- Updates CHANGELOG to 2.2.5 (Álex)([3708f32](https://github.com/RobotnikAutomation/summit_packages/commit/3708f32))

# v2.2.5 (22-10-2021)
- Updates rbkairos_sim repos list dependencies (Álex)([088a21d](https://github.com/RobotnikAutomation/summit_packages/commit/088a21d))
- command_manager: updates command manager, adds subscribers (Álex)([befa5cd](https://github.com/RobotnikAutomation/summit_packages/commit/befa5cd))
- manipulation: Upadtes ur repositories (Álex)([7160e81](https://github.com/RobotnikAutomation/summit_packages/commit/7160e81))

# v2.2.4 (13-10-2021)
- robot_bringup: Fix typo when launching arm_bringup (Álex)([31777ca](https://github.com/RobotnikAutomation/summit_packages/commit/31777ca))
- command_manager: Adds RLC localization handlers to devel (Álex)([e115a35](https://github.com/RobotnikAutomation/summit_packages/commit/e115a35))
- Updates command_manager and rcomponent submodules ([7743d4f](https://github.com/RobotnikAutomation/summit_packages/commit/7743d4f))
- Updates rbkairos_sim ([5797195](https://github.com/RobotnikAutomation/summit_packages/commit/5797195))
- robot_bringup: Screen manipulation from arm_complete to manipulation_complete ([d7ecc94](https://github.com/RobotnikAutomation/summit_packages/commit/d7ecc94))
- sim: Updates summit sim ([1d21e0e](https://github.com/RobotnikAutomation/summit_packages/commit/1d21e0e))
- Fixes benewake issues ([8097021](https://github.com/RobotnikAutomation/summit_packages/commit/8097021))
- Adds benewake to robot_bringup ([f533bab](https://github.com/RobotnikAutomation/summit_packages/commit/f533bab))

# v2.2.3 (23-09-2021)
- Adds benewwake to robot_bringup
- Add interactive_marker_proxy ([979626b](https://github.com/RobotnikAutomation/summit_packages/commit/979626b))
- Updates debs ([379a4ac](https://github.com/RobotnikAutomation/summit_packages/commit/379a4ac))
- Updates robot_bringup hmi launch ([c0465d8](https://github.com/RobotnikAutomation/summit_packages/commit/c0465d8))
- navigation: poi manager to rcomponent branch ([6168336](https://github.com/RobotnikAutomation/summit_packages/commit/6168336))
- sensors: Updates zed URDF ([ba9183e](https://github.com/RobotnikAutomation/summit_packages/commit/ba9183e))
- privater: robotnik_locator -> Comments unnecesary calls to rosReadParams() ([e8f5bb1](https://github.com/RobotnikAutomation/summit_packages/commit/e8f5bb1))
- robot: Adds xacro arm pedestal configurable by macro params ([5e941b3](https://github.com/RobotnikAutomation/summit_packages/commit/5e941b3))

# v2.2.2 (13-09-2021)
- web: Update hmi, fixes bugs and improves envrionment selection ([d8e9670](https://github.com/RobotnikAutomation/summit_packages/commit/d8e9670))
- Updates robot_bringup ([11de169](https://github.com/RobotnikAutomation/summit_packages/commit/11de169))

# v2.2.1 (09-09-2021)
- sensors: deltes ros_rslidar module ([fad9601](https://github.com/RobotnikAutomation/summit_packages/commit/fad9601))
- robot_bringup: Updates typo rslidar launch ([3267d94](https://github.com/RobotnikAutomation/summit_packages/commit/3267d94))
- robot_bringup: Updates gps configs and launch ([1698d9d](https://github.com/RobotnikAutomation/summit_packages/commit/1698d9d))

# v2.2.0 (07-09-2021)
- Updates robot_bringup env variables ([f0316b7](https://github.com/RobotnikAutomation/summit_packages/commit/f0316b7))
- (origin/melodic-devel) robot: Updates RB-Kairos description ([84ab234](https://github.com/RobotnikAutomation/summit_packages/commit/84ab234))
- Updates debs ([4b5234e](https://github.com/RobotnikAutomation/summit_packages/commit/4b5234e))
- robot_bringup: Updates submodule and adds vectornav ([e8dfb72](https://github.com/RobotnikAutomation/summit_packages/commit/e8dfb72))
- robot: Updates robots description to work with latest UR repo version ([9743d6f](https://github.com/RobotnikAutomation/summit_packages/commit/9743d6f))
- web: Adds symbolic links to meshes ([e262864](https://github.com/RobotnikAutomation/summit_packages/commit/e262864))
- localization: Deletes navsat nodes ([0e71f9e](https://github.com/RobotnikAutomation/summit_packages/commit/0e71f9e))
- Update xmlrpc_server: - Add commands (commented) to work with grippers ([93138d5](https://github.com/RobotnikAutomation/summit_packages/commit/93138d5))
- Update robotnik_sensors: - Add commands (commented) to work with grippers ([25405ca](https://github.com/RobotnikAutomation/summit_packages/commit/25405ca))
- Update robot_bringup: - Add hard_brake (true by def) param in robot_control.yaml ([c7fdcb9](https://github.com/RobotnikAutomation/summit_packages/commit/c7fdcb9))
- Update summit_xl_robot: - controller: apply acceleration if watchdog and hard_brake disabled - bringup: Lift added - Using robotnik_pad instead of summit_xl_pa ([65573df](https://github.com/RobotnikAutomation/summit_packages/commit/65573df))
- Update rbkairos_common - description: add run dep for egh gripper - description: add egh as an option in ur5plus - control: add default config for u5e calibration ([b56ff97](https://github.com/RobotnikAutomation/summit_packages/commit/b56ff97))
- Merge pull request #21 from RobotnikAutomation/new_structure_production (Román Navarro García)([6719485](https://github.com/RobotnikAutomation/summit_packages/commit/6719485))
- Fixes issues tested in production ([b1aaab8](https://github.com/RobotnikAutomation/summit_packages/commit/b1aaab8))
- Adds moveit ur5 egh ([0fa2d89](https://github.com/RobotnikAutomation/summit_packages/commit/0fa2d89))
- Adds iiwa_stack ([1d22d63](https://github.com/RobotnikAutomation/summit_packages/commit/1d22d63))
- Updates sumoudles ([c962700](https://github.com/RobotnikAutomation/summit_packages/commit/c962700))
- Updates UR submodules ([e56967f](https://github.com/RobotnikAutomation/summit_packages/commit/e56967f))
- Add ur gripper controller ([1852077](https://github.com/RobotnikAutomation/summit_packages/commit/1852077))
- Update robot_bringup ([3c6d044](https://github.com/RobotnikAutomation/summit_packages/commit/3c6d044))
- robot_bringup: Update default env variables ([c3d958d](https://github.com/RobotnikAutomation/summit_packages/commit/c3d958d))
- Update modules ([969359e](https://github.com/RobotnikAutomation/summit_packages/commit/969359e))
- robot: Update rbkairos_common to latest melodic version ([dc6b2ae](https://github.com/RobotnikAutomation/summit_packages/commit/dc6b2ae))
- Add arm launch to robot_bringup ([04937e4](https://github.com/RobotnikAutomation/summit_packages/commit/04937e4))
- Add astra submodules ([0b2f58c](https://github.com/RobotnikAutomation/summit_packages/commit/0b2f58c))
- Update robot_bringup ([5c558d8](https://github.com/RobotnikAutomation/summit_packages/commit/5c558d8))

# v2.1 (24-04-2020)
* Update to submodules
* Updated build_system (9618da45d8303a34bdaa16c97ce55dd922f0147c)
* Updated imu_tools (0d2a0cdcb8888267663f2ab9cde6a7d171a3a983)
* Updated vectornav (49aef4bc709a0a5af8a845719f47dd029f5027c3)
* Updated eband_local_planner (827c75f94620b8c2797711db77915ac0a75ba1e5)
* Updated marker_mapping (596d55a48778087e0c2b21bca5369983550eaf8f)
* Added robotnik_localization
* Updated rcomponent (27c54b7949519a32974b0b68ea21eed56f4dddca)
* Updated robotnik_modbus_io (8e83f6d12c3ea717e5f76b83b567bc1524cc82f8)
* Updated robotnik_msgs (ec07e4b1b5ab4d93b5c17797aa711b5aed7f4a6a)
* Updated rosmon (9685522a6c5b5933593f4dd46051dffa3edaa424)
* Added robot_local_control
* Added battery_manager
* Updated robotnik_navigation (cdf9a63b4c6ed1567b39f922ca9d4737e6d45cf8)
* Updated rbkairos_common (70e344b7a707fc7f65896857efcb08ceff060be0)
* Updated robotnik_base_hw (ad0366bb0097f2d960d808778c19e8836e2be64b)
* Updated summit_xl_common (8e31345c879a1c795ac6082940300d397714e81f)
* Updated summit_xl_robot (840b3b332bcb266717a65a261a9f50c033f23373)
* Added costmap_prohibition_layer
* Added robot_pose_publisher

# V2.0 (24-02-2020)
* Added new battery_estimation package
* Added robot_local_control (master_procedures branch) with minor improvements in procedure_component
* Added marker_mapping package
* Updated robotnik_base_hw_lib with split battery estimation
* Add rostful
* Added rosmon to workspace
* Add rbkairos packages
* Add simulation packages
* Organize packages in different folders/categories
* Add new gps ublox config
* Add compatibility with zed_camera
* Debs generated with new build_system package

# V1.9 (22-10-2019)
* Updated robotnik_base_hw (1b67efe9cde37eacf846fb9a4ff2f7d9140a04f6)
* Updated robotnik_base_hw_lib (v9.17)
* Updated axis_camera (68fa35778273bd2e2dba295678f776d8e62050aa)
* Added battery_manager (03164705eb765160932df7d7f35af495c753d287)
* Updated poi_manager (37679f7972a121b2bf6746c5e1cd811fb272dba2)
* Updated rcomponent (c88a10658b94fbdf59d1affcbe956f66c886c773)
* Updated robotnik_msgs (8232ba96f72492d9c118f9ab16be148f4b0608ba)
* Updated robotnik_navigation (e2d9cb1d498358e3b8e23775a310a4434a6e61be)
* Updated robotnik_sensors (eb66306a69c419517a498525a42eed3930a1bd6c)
* Added ros_rslidar (2565211637b233c17784121035ba6518da17a23e)
* Updated vectornav (not pushed commit)

# V1.8 (25-09-2019)
* Changed robot prefix to "robot"
* Updated robotnik_base_hw (feaa3b2f48886b203f0f315b2bc0ecb31183d38f)
* Updated robotnik_base_hw_lib (v9.16)
* Updated summit_xl_common (bdefe48f2e82c71b4f1c4ec268e62e75012f2f30)
* Updated summit_xl_robot (5850df0365a4b47af3ac25b2e01fb24437e558d5)

# V1.7(13-05-2019)
* Updated imu_manager (060155a0d4eb1454a5a1324c705565f6d61d1211)
* Updated robotnik_base_hw (da22fe3cdf0d0daa22671b01c2a91c258cf0939f)
* Updated robotnik_base_hw_lib (ce387899abdad4e84772ce8cfbfe27c6cfbc7f79)
* Updated robotnik_msgs (93ef35f5322fa55523344ee7f91ef050bbd6e8a6)
* Updated robotnik_navigation (8db01f9c1388fc07da6cfe66ae2a018c9ca77fa2)
* Updated summit_xl_common (b92df677df5345dee4e98d18a9898b1963ed72db)
* Updated summit_xl_robot (8be24d5232d77b3c336a43cc328420e3494025aa)

# V1.6(07-03-2019)
* Added support for docking with reflectors

# V1.5(06-03-2019)
* Added imu_manager (and update px4_pluginlists.yaml)
* Update robotnik_base_hw to 0.9.14 version
* New sources version with libraries without compile

# V1.4(05-02-2019)
* Update summit_xl_robot to include:
  * New environment variables
  * Scripts to auto config env variables and autorun
* Update robotnik_base_hw to 0.9.12 version
  * Offset in voltage
  * Bugs correction in get_time and state machine
  * ignore invalid joint commands (infinite or nan)
* Updated summit_xl_common (5bc482f26b750a91a10f13465c9c32bd5caf3038)
* Update summit_xl_robot (5f1abf94b0c4bb30fab473c9d458f592c338a9d4)
* Update map_nav_manager (e0e2b12b4bea6239f49cec5c0d80d9ade2d4ec8b)
* Add Changelog file


# V1.3 (21-11-2018)
* Added robot_localization_utils
* Updating summit_xl_robot
* Updating summit_xl_common (removed rl_utils)
* Updating robotnik_msgs:
* New messages for safety
* New battery docking msg
* BaseHw:
  * New battery estimation curve
  * Management of docking station charging through internal I/O
  * Feedback of the current from the battery and feedback of the charging process

# V1.2 (25-10-2018)
* summit_xl_description: added new steel template
* summit_xl_description: added poi_manager to bringup
* poi_manager: deleted namespace from launch file
* map_nav_manager (summit_xl branch): added output to the nodes in launch file. poi_marker: updated namespace of poi_manager services and fixed rear camera topics in javascript files.

# V1.1 (04-10-2018)
* Deleted robotnik_navigation from sources. It is a non-deliverable repository.
* Updated robotnik_base_hw to kinetic-0.9.5 tag.
* Created the script to auto-generate the libraries of the workspace (summit_generate_debs.sh)
* Add the libraries folder to the workspace.

# V1.0
* First version of the sources
