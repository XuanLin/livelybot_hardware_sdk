// pinocchio/fwd.hpp MUST come before any Eigen header
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>       // <-- NEW: for Jacobian

#include "ros/ros.h"
#ifndef RELEASE
#include "robot.h"
#else
#include "livelybot_serial/hardware/robot.h"
#endif
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <csignal>
#include <chrono>
#include <cmath>
#include <geometry_msgs/WrenchStamped.h>
#include <mutex>

static volatile bool g_running = true;
void sigint_handler(int sig) { g_running = false; }

// FT sensor callback
static std::mutex g_ft_mutex;
static Eigen::Vector3d g_force_meas = Eigen::Vector3d::Zero();
static bool g_ft_received = false;

void ftCallback(const geometry_msgs::WrenchStamped::ConstPtr &msg)
{
    std::lock_guard<std::mutex> lock(g_ft_mutex);
    g_force_meas[0] = msg->wrench.force.x;
    g_force_meas[1] = msg->wrench.force.y;
    g_force_meas[2] = msg->wrench.force.z;
    g_ft_received = true;
}

// ---- Right arm joint names in the URDF (order must match CANport 4) ----
static const std::vector<std::string> ARM_JOINT_NAMES = {
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_arm_yaw_joint",
    "r_arm_roll_joint",
    "r_wrist_yaw_joint",
};

// Frame names for FK logging (base -> hand chain)
static const std::vector<std::string> ARM_FRAME_NAMES = {
    "base_link",
    "r_shoulder_pitch_link",
    "r_shoulder_roll_link",
    "r_arm_yaw_link",
    "r_arm_roll_link",
    "r_wrist_yaw_link",
    "r_hand_box_link",
};

static const int NUM_ARM_JOINTS = 5;

// ---- Pseudoinverse for fat matrix (rows < cols): J+ = Jt (J Jt)^-1 ----
Eigen::MatrixXd pinv_right(const Eigen::MatrixXd &J)
{
    Eigen::MatrixXd JJt = J * J.transpose();
    return J.transpose() * JJt.inverse();
}

// ---- Damped pseudoinverse for singularity robustness ----
Eigen::MatrixXd pinv_damped(const Eigen::MatrixXd &J, double lambda = 0.01)
{
    Eigen::MatrixXd JJt = J * J.transpose();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(JJt.rows(), JJt.cols());
    return J.transpose() * (JJt + lambda * lambda * I).inverse();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_task_impedance", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle n("~");

    // ---- Parameters ----
    double kp_task, kd_task, knd, loop_rate_hz, grav_scale;
    double f_ref_y, kf_force;
    std::string csv_path, fk_path, urdf_path;

    n.param<double>("kp_task",     kp_task,     10.0);  // task-space stiffness [N/m]
    n.param<double>("kd_task",     kd_task,     2.0);   // task-space damping   [N·s/m]
    n.param<double>("knd",         knd,         1.0);    // null-space damping   [Nm·s/rad]
    n.param<double>("f_ref_y",     f_ref_y,     0.0);    // desired force along y [N]
    n.param<double>("kf_force",    kf_force,    0.0);    // force feedback gain (start low!)
    n.param<double>("grav_scale",  grav_scale,  1.0);
    n.param<double>("loop_rate",   loop_rate_hz, 200.0);
    n.param<std::string>("csv_path", csv_path,
                         "/home/hightorque/arm_task_impedance.csv");
    n.param<std::string>("fk_path", fk_path,
                         "/home/hightorque/arm_fk_snapshot.csv");
    n.param<std::string>("urdf_path", urdf_path, "");

    // Subscribe to ft sensor wrench topic
    std::string ft_topic;
    n.param<std::string>("ft_topic", ft_topic,
                        "/bus0/ft_sensor0/ft_sensor_readings/wrench");
    ros::Subscriber ft_sub = n.subscribe(ft_topic, 1, ftCallback);
    ROS_INFO("Subscribing to F/T sensor: %s", ft_topic.c_str());

    if (urdf_path.empty())
    {
        ROS_ERROR("urdf_path param is required!");
        return -1;
    }

    // ---- Motor-space home offsets ----
    double motor_home[NUM_ARM_JOINTS] = {-0.26, 0.0, 0.0, -1.0, 0.0};
    double motor_sign[NUM_ARM_JOINTS] = {-1.0, 1.0, -1.0, -1.0, 1.0};

    // Desired motor positions in pinocchio angles
    // This should be pinocchio zero, not hardware zero
    // Recall the hardware installation has that weird offset issue
    // The last motor receive 35deg desired angle to align the FT sensor X with base X
    double q_des_motor_pin[NUM_ARM_JOINTS] = {0.0, -3.14/4, -3.14/2, 3.14/4, 0.0};

    // Per-joint gains
    float ls_kp[NUM_ARM_JOINTS] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float ls_kd[NUM_ARM_JOINTS] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};

    const double dt = 1.0 / loop_rate_hz;
    ros::Rate rate(loop_rate_hz);

    // ==== Load Pinocchio model ====
    ROS_INFO("Loading URDF: %s", urdf_path.c_str());
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    ROS_INFO("Pinocchio model: nq=%d, nv=%d, njoints=%d",
             model.nq, model.nv, (int)model.njoints);

    // Find arm joint indices in the Pinocchio model
    int arm_q_idx[NUM_ARM_JOINTS];
    int arm_v_idx[NUM_ARM_JOINTS];

    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        if (!model.existJointName(ARM_JOINT_NAMES[i]))
        {
            ROS_ERROR("Joint '%s' not found in URDF!", ARM_JOINT_NAMES[i].c_str());
            return -1;
        }
        auto jid = model.getJointId(ARM_JOINT_NAMES[i]);
        arm_q_idx[i] = model.joints[jid].idx_q();
        arm_v_idx[i] = model.joints[jid].idx_v();
        ROS_INFO("  %s -> q_idx=%d, v_idx=%d",
                 ARM_JOINT_NAMES[i].c_str(), arm_q_idx[i], arm_v_idx[i]);
    }

    double joint_damping[NUM_ARM_JOINTS];
    double joint_friction[NUM_ARM_JOINTS];
    bool has_friction = false;

    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        joint_damping[i]  = model.damping[arm_v_idx[i]];
        joint_friction[i] = model.friction[arm_v_idx[i]];
        if (joint_damping[i] != 0.0 || joint_friction[i] != 0.0)
            has_friction = true;
        ROS_INFO("  %s -> damping=%.6f, friction=%.6f",
                ARM_JOINT_NAMES[i].c_str(), joint_damping[i], joint_friction[i]);
    }

    if (has_friction)
        ROS_INFO("\033[1;32mFriction/damping found in URDF — will compensate.\033[0m");
    else
        ROS_INFO("\033[1;33mNo friction/damping in URDF — skipping friction FF.\033[0m");

    // Find end-effector frame ID
    const std::string ee_frame_name = "r_hand_box_link";
    if (!model.existFrame(ee_frame_name))
    {
        ROS_ERROR("End-effector frame '%s' not found in URDF!", ee_frame_name.c_str());
        return -1;
    }
    pinocchio::FrameIndex ee_frame_id = model.getFrameId(ee_frame_name);
    ROS_INFO("End-effector frame: %s (id=%d)", ee_frame_name.c_str(), (int)ee_frame_id);

    // Find arm frame IDs for FK logging
    std::vector<pinocchio::FrameIndex> arm_frame_ids;
    for (const auto &fname : ARM_FRAME_NAMES)
    {
        if (!model.existFrame(fname))
        {
            ROS_WARN("Frame '%s' not found, will skip FK logging for it", fname.c_str());
            arm_frame_ids.push_back(0);
        }
        else
        {
            arm_frame_ids.push_back(model.getFrameId(fname));
        }
    }

    // ==== Initialize robot hardware ====
    livelybot_serial::robot rb;

    struct ArmMotor {
        motor* m;
        std::string name;
    };
    std::vector<ArmMotor> arm;

    for (motor *m : rb.Motors)
    {
        if (m->get_motor_belong_canport() == 4)
        {
            arm.push_back({m, m->get_motor_name()});
            ROS_INFO("Motor %d: %s (id=%d)",
                     (int)arm.size() - 1, m->get_motor_name().c_str(),
                     m->get_motor_id());
        }
    }

    if ((int)arm.size() != NUM_ARM_JOINTS)
    {
        ROS_ERROR("Expected %d arm motors, found %d!", NUM_ARM_JOINTS, (int)arm.size());
        return -1;
    }

    // ==== Open CSV log ====
    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        ROS_ERROR("Cannot open CSV: %s", csv_path.c_str());
        return -1;
    }

    // CSV header
    csv << "time";
    // Task-space columns
    csv << ",ee_x,ee_y,ee_z"
        << ",ee_x_des,ee_y_des,ee_z_des"
        << ",err_x,err_y,err_z"
        << ",vel_x,vel_y,vel_z"
        << ",f_des_world_x,f_des_world_y,f_des_world_z"
        << ",f_meas_env_world_x,f_meas_env_world_y,f_meas_env_world_z"
        << ",f_err_world_x,f_err_world_y,f_err_world_z"
        << ",ft_raw_x,ft_raw_y,ft_raw_z";

    // Per-joint columns
    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        csv << "," << arm[i].name << "_motor_pos"
            << "," << arm[i].name << "_pin_pos"
            << "," << arm[i].name << "_vel"
            << "," << arm[i].name << "_tau_task"
            << "," << arm[i].name << "_tau_null"
            << "," << arm[i].name << "_tau_bias"
            << "," << arm[i].name << "_tau_cmd"
            << "," << arm[i].name << "_tau_measured";
    }
    // FK frame positions
    for (const auto &fname : ARM_FRAME_NAMES)
        csv << "," << fname << "_x," << fname << "_y," << fname << "_z";
    csv << std::endl;

    // ==== Wait for initial feedback ====
    ROS_INFO("Waiting for motor feedback...");
    rb.send_get_motor_state_cmd();
    ros::Duration(0.5).sleep();

    // ---- Pinocchio state vectors (full model) ----
    Eigen::VectorXd q_full   = Eigen::VectorXd::Zero(model.nq);
    Eigen::VectorXd v_full   = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd a_zero   = Eigen::VectorXd::Zero(model.nv);

    // Read initial position
    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        float pos = arm[i].m->get_current_motor_state()->position;
        double q_pin = motor_sign[i] * pos - motor_home[i];
        q_full[arm_q_idx[i]] = q_pin;
        ROS_INFO("  %s motor_pos=%.4f  pin_pos=%.4f",
                 arm[i].name.c_str(), pos, q_pin);
    }

    // ---- Compute initial EE position as the task-space target ----
    pinocchio::forwardKinematics(model, data, q_full);
    pinocchio::updateFramePlacements(model, data);

    // Verify if the base frame and world frame aligns up
    Eigen::Vector3d base_pos = data.oMf[model.getFrameId("base_link")].translation();
    ROS_INFO("base_link in world: [%.4f, %.4f, %.4f]", base_pos[0], base_pos[1], base_pos[2]);
    if (base_pos.norm() > 1e-6)
    {
        ROS_WARN("base_link is offset from world frame by %.4f m! "
                "Task-space directions (e.g. y for wall pushing) may not align with expectations.",
                base_pos.norm());
    }

    Eigen::Vector3d x_des = data.oMf[ee_frame_id].translation();

    ROS_INFO("Initial EE position (= x_des): [%.4f, %.4f, %.4f]",
             x_des[0], x_des[1], x_des[2]);

    // Wait for first F/T reading and capture bias
    ROS_INFO("Waiting for F/T sensor data...");
    while (ros::ok() && g_running && !g_ft_received)
    {
        ros::spinOnce();
        ros::Duration(0.01).sleep();
    }
    Eigen::Vector3d f_bias;
    {
        std::lock_guard<std::mutex> lock(g_ft_mutex);
        f_bias = g_force_meas;
    }
    ROS_INFO("F/T bias captured: [%.2f, %.2f, %.2f]", f_bias[0], f_bias[1], f_bias[2]);

    // ---- Selection matrix: S selects position-controlled axes ----
    //   S = diag(1, 0, 1)  -> position control in x, z
    //   (I-S) = diag(0, 1, 0)  -> force control in y
    Eigen::Vector3d S_diag(1.0, 0.0, 1.0);   // position-controlled axes
    Eigen::DiagonalMatrix<double, 3> S_mat(S_diag);
    Eigen::DiagonalMatrix<double, 3> Sf_mat(Eigen::Vector3d(1.0, 1.0, 1.0) - S_diag); // force axes








    // ==== Move to start position before engaging impedance ====
    ROS_INFO("Waiting for motor feedback...");
    rb.send_get_motor_state_cmd();
    ros::Duration(0.5).sleep();

    {
        ROS_INFO("\033[1;33m--- Moving to start position over 3s ---\033[0m");
        double move_duration = 3.0;
        int move_steps = (int)(move_duration * loop_rate_hz);

        // Capture current motor positions
        double q_motor_start[NUM_ARM_JOINTS];
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
            q_motor_start[i] = arm[i].m->get_current_motor_state()->position;

        // Target motor positions — must match what the main loop sends as pos cmd
        double q_motor_target[NUM_ARM_JOINTS];
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
            // q_motor_target[i] = motor_sign[i] * q_des_motor_pin[i] - motor_home[i];
            q_motor_target[i] = (q_des_motor_pin[i] + motor_home[i]) / motor_sign[i];

        for (int step = 0; step < move_steps && ros::ok() && g_running; step++)
        {
            rb.detect_motor_limit();
            double alpha = (double)step / (double)move_steps;
            double blend = 0.5 * (1.0 - cos(M_PI * alpha));  // smooth cosine blend

            for (int i = 0; i < NUM_ARM_JOINTS; i++)
            {
                double q_cmd = q_motor_start[i] + blend * (q_motor_target[i] - q_motor_start[i]);
                arm[i].m->pos_vel_tqe_kp_kd(
                    (float)q_cmd, 0.0f, 0.0f,
                    5.0f,   // moderate kp for positioning
                    0.3f    // moderate kd
                );
            }
            rb.motor_send_2();
            ros::spinOnce();
            rate.sleep();
        }
        ROS_INFO("Reached start position.");
        ros::Duration(0.5).sleep();
    }

    // ==== Print EE frame orientation for FT sensor mapping ====
    {
        // Run FK at the start position
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            q_full[arm_q_idx[i]] = q_des_motor_pin[i];  // desired pin-space position
        }
        pinocchio::forwardKinematics(model, data, q_full);
        pinocchio::updateFramePlacements(model, data);

        Eigen::Matrix3d R_ee = data.oMf[ee_frame_id].rotation();
        Eigen::Vector3d p_ee = data.oMf[ee_frame_id].translation();

        ROS_INFO("\033[1;36m=== EE frame at start pose ===\033[0m");
        ROS_INFO("  EE position in world: [%.4f, %.4f, %.4f]", p_ee[0], p_ee[1], p_ee[2]);
        ROS_INFO("  EE local X in world:  [%.4f, %.4f, %.4f]", R_ee(0,0), R_ee(1,0), R_ee(2,0));
        ROS_INFO("  EE local Y in world:  [%.4f, %.4f, %.4f]", R_ee(0,1), R_ee(1,1), R_ee(2,1));
        ROS_INFO("  EE local Z in world:  [%.4f, %.4f, %.4f]", R_ee(0,2), R_ee(1,2), R_ee(2,2));
        ROS_INFO("Use this to figure out FT sensor frame → EE local frame mapping.");
        ROS_INFO("e.g., if EE local Y = [0, -1, 0] in world, then EE local Y points world -Y.");
    }







    ROS_INFO("\033[1;32m--- TASK-SPACE IMPEDANCE CONTROL STARTED ---\033[0m");
    ROS_INFO("kp_task=%.1f N/m  kd_task=%.1f N·s/m  knd=%.2f Nm·s/rad",
             kp_task, kd_task, knd);
    ROS_INFO("Force tracking: f_ref_y=%.1f N  kf=%.2f", f_ref_y, kf_force);
    ROS_INFO("Position control axes: x, z  |  Force control axis: y");
    ROS_INFO("\033[1;33mPress Ctrl+C to stop.\033[0m");

    auto t_start = std::chrono::steady_clock::now();

    // Preallocate working matrices
    Eigen::MatrixXd J_full(6, model.nv);       // full 6×nv Jacobian
    Eigen::MatrixXd Jp_arm(3, NUM_ARM_JOINTS); // 3×5 position Jacobian (arm cols only)
    Eigen::VectorXd bias(model.nv);

    // ==== Main loop ====
    while (ros::ok() && g_running)
    {
        rb.detect_motor_limit();

        auto t_now = std::chrono::steady_clock::now();
        double t_sec = std::chrono::duration<double>(t_now - t_start).count();

        // ---- 1. Read motor states, convert to Pinocchio space ----
        double q_motor[NUM_ARM_JOINTS];
        double q_pin[NUM_ARM_JOINTS];
        double v_motor[NUM_ARM_JOINTS];
        Eigen::VectorXd v_arm(NUM_ARM_JOINTS);  // arm joint velocities in pin space

        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            auto st = arm[i].m->get_current_motor_state();
            q_motor[i] = st->position;
            v_motor[i] = st->velocity;

            q_pin[i] = motor_sign[i] * q_motor[i] - motor_home[i];

            q_full[arm_q_idx[i]] = q_pin[i];
            v_full[arm_v_idx[i]] = motor_sign[i] * v_motor[i];

            v_arm[i] = motor_sign[i] * v_motor[i];
        }

        // ---- 2. FK: get current EE position ----
        pinocchio::forwardKinematics(model, data, q_full);
        pinocchio::updateFramePlacements(model, data);
        Eigen::Vector3d x_cur = data.oMf[ee_frame_id].translation();

        // ---- 3. Jacobian: compute 6×nv in LOCAL frame, extract 3×5 arm block ----
        J_full.setZero();
        pinocchio::computeFrameJacobian(model, data, q_full, ee_frame_id,
                                        pinocchio::LOCAL, J_full);
        // Extract linear (top 3 rows) for arm joint columns only
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
            Jp_arm.col(i) = J_full.block<3, 1>(0, arm_v_idx[i]);

        // ---- 4. Task-space errors ----
        Eigen::Vector3d e_pos = x_des - x_cur;            // position error [3×1]
        // Be careful this x_dot is also in LOCAL frame, which should be ok
        // since we don't have non-zero desired velocity
        Eigen::Vector3d x_dot = Jp_arm * v_arm;           // EE velocity    [3×1]

        // ---- 5. Force measurement ----
        Eigen::Vector3d f_meas;
        {
            std::lock_guard<std::mutex> lock(g_ft_mutex);
            f_meas = g_force_meas - f_bias;
        }
        
        // ---- Transform FT sensor measurement to EE local frame ----
        // FT x = EE x, FT y = EE y, FT z = -EE z
        Eigen::Vector3d f_meas_local;
        // Temporary remove x and y measurements, only z matters
        // Turning x and y on cause some vibration, probably due to tuning + x/y installation off
        f_meas_local[0] =  f_meas[0]*0.0;  
        f_meas_local[1] =  f_meas[1]*0.0;
        f_meas_local[2] = -f_meas[2];

        // ---- Desired force in world, transform to EE local ----
        Eigen::Vector3d f_des_world(0.0, -15.0, 0.0);  // push along the world Y to the wall
        Eigen::Matrix3d R_ee = data.oMf[ee_frame_id].rotation();
        Eigen::Vector3d f_des_local = R_ee.transpose() * f_des_world;

        // ---- Force error in EE local frame ----
        // FT sensor measures the force onto the robot, but f_err is force onto the env
        // We add a negative sign here
        // so we don't need a negative sign when multiplying with Jacobian
        Eigen::Vector3d f_meas_onto_env_local = -f_meas_local;
        Eigen::Vector3d f_err_local = f_des_local - f_meas_onto_env_local;

        // tau = J_local^T * f_err_local (both in EE local frame)
        Eigen::VectorXd tau_task = Jp_arm.transpose() * f_err_local;

        ROS_INFO_THROTTLE(0.5,
            "FT raw=[%.2f, %.2f, %.2f] "
            "f_meas_onto_env_local=[%.2f, %.2f, %.2f] "
            "f_des_local=[%.2f, %.2f, %.2f] "
            "f_err_local=[%.2f, %.2f, %.2f] "
            "tau_force=[%.5f, %.5f, %.5f, %.5f, %.5f]",
            f_meas[0], f_meas[1], f_meas[2],
            f_meas_onto_env_local[0], f_meas_onto_env_local[1], f_meas_onto_env_local[2],
            f_des_local[0], f_des_local[1], f_des_local[2],
            f_err_local[0], f_err_local[1], f_err_local[2],
            tau_task[0], tau_task[1], tau_task[2], tau_task[3], tau_task[4]);

        // // ---- 6. Hybrid position/force task-space command ----
        // //   F = S * (Kp * e_pos - Kd * x_dot)                   <-- position axes
        // //     + (I-S) * (F_ref + Kf * (F_ref - F_meas))         <-- force axes
        // Eigen::Vector3d F_position = S_mat * (kp_task * e_pos - kd_task * x_dot);

        // Eigen::Vector3d F_ref_vec(0.0, f_ref_y, 0.0);
        // Eigen::Vector3d F_force = Sf_mat * (F_ref_vec + kf_force * (F_ref_vec - f_meas));

        // Eigen::Vector3d F_cmd = F_position;  // + F_force;

        // ---- 7. Map to joint torques: tau_task = Jp^T * F ----
        // Eigen::VectorXd tau_task = Jp_arm.transpose() * F_cmd;  // [5×1]

        // ---- 8. Null-space damping: tau_null = -(I - Jp+ Jp) * Knd * v_arm ----
        Eigen::MatrixXd Jp_pinv = pinv_damped(Jp_arm);   // [5×3]
        Eigen::MatrixXd N = Eigen::MatrixXd::Identity(NUM_ARM_JOINTS, NUM_ARM_JOINTS)
                            - Jp_pinv * Jp_arm;           // [5×5] null-space projector
        Eigen::VectorXd tau_null = -N * knd * v_arm;      // [5×1]

        // ---- 9. Bias (gravity + Coriolis): h(q, v) via RNEA(q, v, 0) ----
        pinocchio::rnea(model, data, q_full, v_full, a_zero);
        bias = data.tau;

        // ---- 10. Send commands ----
        csv << t_sec;

        Eigen::Vector3d f_meas_env_world = R_ee * f_meas_onto_env_local;
        Eigen::Vector3d f_err_world = f_des_world - f_meas_env_world;

        // Log task-space data
        csv << "," << x_cur[0] << "," << x_cur[1] << "," << x_cur[2]
            << "," << x_des[0] << "," << x_des[1] << "," << x_des[2]
            << "," << e_pos[0] << "," << e_pos[1] << "," << e_pos[2]
            << "," << x_dot[0] << "," << x_dot[1] << "," << x_dot[2]
            << "," << f_des_world[0] << "," << f_des_world[1] << "," << f_des_world[2]
            << "," << f_meas_env_world[0] << "," << f_meas_env_world[1] << "," << f_meas_env_world[2]
            << "," << f_err_world[0] << "," << f_err_world[1] << "," << f_err_world[2]
            << "," << f_meas[0] << "," << f_meas[1] << "," << f_meas[2];

        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            int vi = arm_v_idx[i];
            double tau_bias_i = bias[vi];
            double tau_i = tau_bias_i + tau_task[i]; // + tau_null[i];

            // Friction/damping feedforward using actual velocity
            // This friction compensation has some weird vibration issue?
            // if (has_friction)
            // {
            //     double dq_actual = motor_sign[i] * v_motor[i];  // pin-space velocity
            //     tau_i += joint_damping[i] * dq_actual;           // viscous
            //     if (dq_actual > 1e-2)
            //         tau_i += joint_friction[i];                   // Coulomb +
            //     else if (dq_actual < -1e-2)
            //         tau_i -= joint_friction[i];                   // Coulomb -
            // }

            // Pure torque control
            arm[i].m->pos_vel_tqe_kp_kd(
                // This should be pinocchio zero, not hardware zero
                // Recall the hardware installation has that weird offset issue
                // (float) (motor_sign[i] * q_des_motor_pin[i] - motor_home[i]),
                (float) (q_des_motor_pin[i] + motor_home[i]) / motor_sign[i],
                0.0f,
                (float)(motor_sign[i] * tau_i),
                ls_kp[i],
                ls_kd[i]
            );

            auto st = arm[i].m->get_current_motor_state();

            csv << "," << q_motor[i]
                << "," << q_pin[i]
                << "," << v_motor[i]
                << "," << tau_task[i]
                << "," << tau_null[i]
                << "," << tau_bias_i
                << "," << tau_i
                << "," << st->torque;
        }

        // Log frame positions
        for (size_t f = 0; f < arm_frame_ids.size(); f++)
        {
            Eigen::Vector3d pos = data.oMf[arm_frame_ids[f]].translation();
            csv << "," << pos[0] << "," << pos[1] << "," << pos[2];
        }
        csv << std::endl;

        rb.motor_send_2();

        ros::spinOnce();
        rate.sleep();
    }

    // ---- Shutdown: ramp torques to zero ----
    ROS_INFO("Shutting down -- ramping torques to zero...");
    for (int iter = 0; iter < (int)(1.0 * loop_rate_hz) && ros::ok(); iter++)
    {
        rb.detect_motor_limit();
        double fade = 1.0 - (double)iter / (1.0 * loop_rate_hz);

        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            auto st = arm[i].m->get_current_motor_state();
            q_full[arm_q_idx[i]] = motor_sign[i] * st->position - motor_home[i];
            v_full[arm_v_idx[i]] = motor_sign[i] * st->velocity;
        }

        // Gravity only during shutdown (safe ramp-down)
        pinocchio::rnea(model, data, q_full, v_full * 0.0, a_zero);

        for (int j = 0; j < NUM_ARM_JOINTS; j++)
        {
            auto st = arm[j].m->get_current_motor_state();
            double tau_grav = data.tau[arm_v_idx[j]] * grav_scale * fade;
            arm[j].m->pos_vel_tqe_kp_kd(
                st->position, 0.0f, (float)(motor_sign[j] * tau_grav), 0.0f, 0.0f);
        }
        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    csv.close();
    ROS_INFO("\033[1;32m--- TASK-SPACE IMPEDANCE STOPPED ---\033[0m");
    ROS_INFO("Data: %s", csv_path.c_str());

    return 0;
}
