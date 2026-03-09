// pinocchio/fwd.hpp MUST come before any Eigen header
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

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

static volatile bool g_running = true;
void sigint_handler(int sig) { g_running = false; }

// ---- Right arm joint names in the URDF (order must match CANport 4) ----
static const std::vector<std::string> ARM_JOINT_NAMES = {
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_arm_yaw_joint",
    "r_arm_roll_joint",
    "r_wrist_yaw_joint",
};

static const int NUM_ARM_JOINTS = 5;

// ---- Sinusoidal trajectory parameters (per joint) ----
// Center position in Pinocchio (URDF) space
static const double Q_CENTER[NUM_ARM_JOINTS] = {0.0, -M_PI / 2.0, 0.0, M_PI / 2.0, 0.0};

// Amplitude (rad) -- keep small for safety; tune per joint
static const double Q_AMP[NUM_ARM_JOINTS] = {0.0, M_PI / 4.0, 0.0, 0.0, 0.0};

// Frequency (Hz) -- different per joint so we excite a richer set of configs
static const double Q_FREQ[NUM_ARM_JOINTS] = {0.20, 0.20, 0.20, 0.20, 0.20};

// Duration of the sinusoidal test (seconds)
static const double TRAJ_DURATION = 20.0;


// ---- Desired trajectory: q, dq, ddq as functions of time ----
// Pure sinusoid, no envelope. sin(0)=0 so trajectory starts exactly at Q_CENTER.
inline void compute_desired(double t, double q_des[NUM_ARM_JOINTS],
                            double dq_des[NUM_ARM_JOINTS],
                            double ddq_des[NUM_ARM_JOINTS])
{
    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        double w = 2.0 * M_PI * Q_FREQ[i];
        
        // Always starts at sin(0) = 0
        q_des[i]   = Q_CENTER[i] + Q_AMP[i] * sin(w * t);
        dq_des[i]  = Q_AMP[i] * w * cos(w * t);
        ddq_des[i] = -Q_AMP[i] * w * w * sin(w * t);
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_sysid_traj", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle n("~");

    // ---- Parameters ----
    double loop_rate_hz;
    std::string csv_path, urdf_path;

    n.param<double>("loop_rate",     loop_rate_hz, 200.0);
    n.param<std::string>("csv_path", csv_path,
                         "/home/hightorque/arm_sysid_traj.csv");
    n.param<std::string>("urdf_path", urdf_path, "");

    if (urdf_path.empty())
    {
        ROS_ERROR("urdf_path param is required!");
        return -1;
    }

    // ---- Motor-space calibration (same as gravity comp) ----
    double motor_home[NUM_ARM_JOINTS] = {-0.26, 0.0, 0.0, -1.0, 0.0};
    double motor_sign[NUM_ARM_JOINTS] = {-1.0, 1.0, -1.0, -1.0, 1.0};

    ros::Rate rate(loop_rate_hz);

    // ==== Load Pinocchio model ====
    ROS_INFO("Loading URDF: %s", urdf_path.c_str());
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    ROS_INFO("Pinocchio model: nq=%d, nv=%d, njoints=%d",
             model.nq, model.nv, (int)model.njoints);

    // Find arm joint indices in Pinocchio model
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
        auto jid = model.getJointId(ARM_JOINT_NAMES[i]);
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
    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        csv << "," << arm[i].name << "_q_des"
            << "," << arm[i].name << "_q_act"
            << "," << arm[i].name << "_dq_des"
            << "," << arm[i].name << "_dq_act"
            << "," << arm[i].name << "_ddq_des"
            << "," << arm[i].name << "_tau_ff"
            << "," << arm[i].name << "_tau_meas";
    }
    csv << std::endl;

    // ==== Wait for initial feedback ====
    ROS_INFO("Waiting for motor feedback...");
    rb.send_get_motor_state_cmd();
    ros::Duration(0.5).sleep();

    // ==== Phase 0: Move to start position using position control ====
    // sin(0)=0, so trajectory starts at Q_CENTER. Move there first.
    {
        ROS_INFO("\033[1;33m--- Moving to start position (Q_CENTER) over 3s ---\033[0m");
        double move_duration = 3.0;
        int move_steps = (int)(move_duration * loop_rate_hz);

        // Capture current motor positions
        double q_motor_start[NUM_ARM_JOINTS];
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
            q_motor_start[i] = arm[i].m->get_current_motor_state()->position;

        // Target motor positions for Q_CENTER
        double q_motor_target[NUM_ARM_JOINTS];
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
            q_motor_target[i] = (Q_CENTER[i] + motor_home[i]) / motor_sign[i];

        for (int step = 0; step < move_steps && ros::ok() && g_running; step++)
        {
            rb.detect_motor_limit();
            double alpha = (double)step / (double)move_steps;
            double blend = 0.5 * (1.0 - cos(M_PI * alpha));

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

    // ==== Phase 1: Pure feedforward trajectory tracking ====
    ROS_INFO("\033[1;32m--- FEEDFORWARD TRAJECTORY STARTED (%.1fs) ---\033[0m", TRAJ_DURATION);
    ROS_INFO("\033[1;33mPure inverse-dynamics torque, NO PD feedback.\033[0m");
    ROS_INFO("\033[1;33mPress Ctrl+C to abort.\033[0m");

    // Pinocchio state vectors (full model)
    Eigen::VectorXd q_full   = Eigen::VectorXd::Zero(model.nq);
    Eigen::VectorXd dq_full  = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd ddq_full = Eigen::VectorXd::Zero(model.nv);

    auto t_start = std::chrono::steady_clock::now();
    int print_counter = 0;
    const int PRINT_INTERVAL = 200;

    while (ros::ok() && g_running)
    {
        rb.detect_motor_limit();

        auto t_now = std::chrono::steady_clock::now();
        double t_sec = std::chrono::duration<double>(t_now - t_start).count();

        if (t_sec > TRAJ_DURATION)
            break;

        // ---- Desired trajectory ----
        double q_des[NUM_ARM_JOINTS], dq_des[NUM_ARM_JOINTS], ddq_des[NUM_ARM_JOINTS];
        compute_desired(t_sec, q_des, dq_des, ddq_des);

        // ---- Fill Pinocchio vectors with DESIRED values ----
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            q_full[arm_q_idx[i]]   = q_des[i];
            dq_full[arm_v_idx[i]]  = dq_des[i];
            ddq_full[arm_v_idx[i]] = ddq_des[i];
        }

        // ---- Inverse dynamics: tau = RNEA(q_des, dq_des, ddq_des) ----
        pinocchio::rnea(model, data, q_full, dq_full, ddq_full);

        // ---- Read actual motor states ----
        double q_act[NUM_ARM_JOINTS], dq_act[NUM_ARM_JOINTS];
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            auto st = arm[i].m->get_current_motor_state();
            double q_motor_raw = st->position;
            q_act[i]  = motor_sign[i] * q_motor_raw - motor_home[i];
            dq_act[i] = motor_sign[i] * st->velocity;
        }

        // ---- Send pure feedforward torque (kp=0, kd=0) ----
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            double tau_ff = data.tau[arm_v_idx[i]];
            double q_motor_des = (q_des[i] + motor_home[i]) / motor_sign[i];
            double dq_motor_des = dq_des[i] / motor_sign[i];

            float p_gain = 5.0f;
            float d_gain = 0.3f;
            if (i == 1){
                p_gain = 1.0f;  // Only shoulder roll has low gain - benchmark tracking
                // p_gain = 0.0f;
                // d_gain = 0.0f;
                // If you want to test pure feedforward, make this p_gain = 0.0f
            }

            // Add friction/damping compensation using ACTUAL velocity
            if (has_friction)
            {
                double dq_actual = dq_act[i];
                tau_ff += joint_damping[i] * dq_actual;                         // viscous
                if (dq_actual > 1e-2)
                    tau_ff += joint_friction[i];                                 // Coulomb +
                else if (dq_actual < -1e-2)
                    tau_ff -= joint_friction[i];                                 // Coulomb -
                // else: near zero velocity, don't apply Coulomb (avoids chatter)
            }

            arm[i].m->pos_vel_tqe_kp_kd(
                (float)q_motor_des,                 // position target
                (float)dq_motor_des,                // velocity target
                (float)(motor_sign[i] * tau_ff),    // feedforward torque
                p_gain,                               // light kp
                d_gain                                // light kd
            );
        }

        // ---- Log to CSV (every tick) ----
        csv << t_sec;
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            double tau_ff = data.tau[arm_v_idx[i]];

            if (has_friction)
            {
                double dq_actual = dq_act[i];
                tau_ff += joint_damping[i] * dq_actual;
                if (dq_actual > 1e-2)
                    tau_ff += joint_friction[i];
                else if (dq_actual < -1e-2)
                    tau_ff -= joint_friction[i];
            }

            auto st = arm[i].m->get_current_motor_state();
            csv << "," << q_des[i]
                << "," << q_act[i]
                << "," << dq_des[i]
                << "," << dq_act[i]
                << "," << ddq_des[i]
                << "," << tau_ff
                << "," << st->torque;
        }
        csv << std::endl;

        // ---- Periodic print ----
        print_counter++;
        if (print_counter >= PRINT_INTERVAL)
        {
            print_counter = 0;
            ROS_INFO("t=%.1f / %.1f", t_sec, TRAJ_DURATION);
            for (int i = 0; i < NUM_ARM_JOINTS; i++)
            {
                auto st = arm[i].m->get_current_motor_state();
                double err_deg = (q_des[i] - q_act[i]) * 180.0 / M_PI;
                ROS_INFO("  [%d] %s: raw_pos=%.8f raw_tqe=%.4f",
                        i, arm[i].name.c_str(),
                        st->position, st->torque);
            }
        }

        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    // ==== Shutdown: zero torques immediately ====
    ROS_INFO("Trajectory done -- zeroing torques.");
    for (int j = 0; j < NUM_ARM_JOINTS; j++)
    {
        auto st = arm[j].m->get_current_motor_state();
        arm[j].m->pos_vel_tqe_kp_kd(st->position, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    rb.motor_send_2();

    csv.close();
    ROS_INFO("\033[1;32m--- TRAJECTORY TEST COMPLETE ---\033[0m");
    ROS_INFO("CSV log: %s", csv_path.c_str());
    ROS_INFO("Run:  python3 plot_sysid_traj.py --csv %s", csv_path.c_str());

    return 0;
}
