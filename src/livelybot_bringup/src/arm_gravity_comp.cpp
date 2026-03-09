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

// Frame names for FK logging (base → hand chain)
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

int main(int argc, char **argv)
{

    ros::init(argc, argv, "arm_gravity_comp", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle n("~");

    // ---- Parameters ----
    double kp, kd, loop_rate_hz, grav_scale;
    std::string csv_path, fk_path, urdf_path;

    n.param<double>("kp",          kp,          0.0);    // low stiffness (compliant)
    n.param<double>("kd",          kd,          0.0);    // low damping
    n.param<double>("grav_scale",  grav_scale,  1.0);    // scale factor for gravity comp (tune if model is off)
    n.param<double>("loop_rate",   loop_rate_hz, 200.0);
    n.param<std::string>("csv_path", csv_path,
                         "/home/hightorque/arm_gravity_comp.csv");
    n.param<std::string>("fk_path", fk_path,
                         "/home/hightorque/arm_fk_snapshot.csv");
    n.param<std::string>("urdf_path", urdf_path, "");

    if (urdf_path.empty())
    {
        ROS_ERROR("urdf_path param is required!");
        return -1;
    }

    // ---- Motor-space home offsets: motor angles that produce URDF q=0 ----
    // Determined from hardware testing:
    //   motor_home[i] is the motor reading when the arm is in URDF zero (T-pose)
    //   q_pinocchio = q_motor - motor_home
    // double motor_home[NUM_ARM_JOINTS] = {0.0, 0.0, 0.0, -1.57, 0.0};
    // double motor_sign[NUM_ARM_JOINTS] = {-1.0, 1.0, 1.0, 1.0, 1.0};
    double motor_home[NUM_ARM_JOINTS] = {-0.26, 0.0, 0.0, -1.0, 0.0};
    double motor_sign[NUM_ARM_JOINTS] = {-1.0, 1.0, -1.0, -1.0, 1.0};

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
    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        csv << "," << arm[i].name << "_motor_pos"
            << "," << arm[i].name << "_pin_pos"
            << "," << arm[i].name << "_vel"
            << "," << arm[i].name << "_tau_grav"
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

    for (int i = 0; i < NUM_ARM_JOINTS; i++)
    {
        float pos = arm[i].m->get_current_motor_state()->position;
        ROS_INFO("  %s motor_pos=%.4f  pin_pos=%.4f",
                 arm[i].name.c_str(), pos, pos - motor_home[i]);
    }

    ROS_INFO("\033[1;32m--- GRAVITY COMPENSATION STARTED ---\033[0m");
    ROS_INFO("kp=%.2f  kd=%.2f  grav_scale=%.2f", kp, kd, grav_scale);
    ROS_INFO("\033[1;33mThe arm should feel weightless. Backdrive it to test.\033[0m");
    ROS_INFO("\033[1;33mPress Ctrl+C to stop.\033[0m");

    auto t_start = std::chrono::steady_clock::now();
    int fk_snapshot_counter = 0;
    const int FK_SNAPSHOT_INTERVAL = 200;  // save FK snapshot every N iterations (~1s at 200Hz)

    int print_counter = 0;
    const int PRINT_INTERVAL = 200;  // every ~1s at 200Hz

    // Pinocchio state vectors (full model)
    Eigen::VectorXd q_full = Eigen::VectorXd::Zero(model.nq);
    Eigen::VectorXd v_full = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd a_zero = Eigen::VectorXd::Zero(model.nv);

    // ==== Main loop ====
    while (ros::ok() && g_running)
    {
        rb.detect_motor_limit();

        auto t_now = std::chrono::steady_clock::now();
        double t_sec = std::chrono::duration<double>(t_now - t_start).count();

        // ---- Read motor states, convert to Pinocchio space ----
        double q_motor[NUM_ARM_JOINTS];
        double q_pin[NUM_ARM_JOINTS];
        double v_motor[NUM_ARM_JOINTS];

        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            auto st = arm[i].m->get_current_motor_state();
            q_motor[i] = st->position;
            v_motor[i] = st->velocity;

            q_pin[i] = motor_sign[i] * q_motor[i] - motor_home[i];

            // Set in full Pinocchio configuration
            q_full[arm_q_idx[i]] = q_pin[i];
            v_full[arm_v_idx[i]] = v_motor[i];
        }



        // Periodic debug print
        print_counter++;
        if (print_counter >= PRINT_INTERVAL)
        {
            print_counter = 0;
            ROS_INFO("---- Motor State (t=%.1f) ----", t_sec);
            for (int i = 0; i < NUM_ARM_JOINTS; i++)
            {
                auto st = arm[i].m->get_current_motor_state();
                ROS_INFO("  [%d] %s: motor=%.4f(%.1f°)  home=%.4f(%.1f°)  pin=%.4f(%.1f°)  vel=%.4f  torque=%.4f",
                 i, arm[i].name.c_str(),
                 q_motor[i], q_motor[i] * 180.0 / M_PI,
                 motor_home[i], motor_home[i] * 180.0 / M_PI,
                 q_pin[i], q_pin[i] * 180.0 / M_PI,
                 v_motor[i], st->torque);
            }
        }



        // ---- Compute gravity torques via RNEA(q, 0, 0) ----
        pinocchio::rnea(model, data, q_full, v_full * 0.0, a_zero);
        // data.tau now contains gravity compensation torques


        // Periodic debug: show Pinocchio inputs & outputs
        print_counter++;
        if (print_counter >= PRINT_INTERVAL)
        {
            print_counter = 0;
            ROS_INFO("---- Debug (t=%.1f) ----", t_sec);
            for (int i = 0; i < NUM_ARM_JOINTS; i++)
            {
                auto st = arm[i].m->get_current_motor_state();
                ROS_INFO("  [%d] %s: motor=%.4f(%.1f°)  home=%.4f(%.1f°)  pin=%.4f(%.1f°)  vel=%.4f  meas_torque=%.4f",
                        i, arm[i].name.c_str(),
                        q_motor[i], q_motor[i] * 180.0 / M_PI,
                        motor_home[i], motor_home[i] * 180.0 / M_PI,
                        q_pin[i], q_pin[i] * 180.0 / M_PI,
                        v_motor[i], st->torque);
            }
            ROS_INFO("  q_full (arm slots): [%.4f, %.4f, %.4f, %.4f, %.4f]",
                    q_full[arm_q_idx[0]], q_full[arm_q_idx[1]], q_full[arm_q_idx[2]],
                    q_full[arm_q_idx[3]], q_full[arm_q_idx[4]]);
            ROS_INFO("  tau_grav (arm slots): [%.4f, %.4f, %.4f, %.4f, %.4f]",
                    data.tau[arm_v_idx[0]] * grav_scale,
                    data.tau[arm_v_idx[1]] * grav_scale,
                    data.tau[arm_v_idx[2]] * grav_scale,
                    data.tau[arm_v_idx[3]] * grav_scale,
                    data.tau[arm_v_idx[4]] * grav_scale);
        }


        // ---- Compute FK for logging ----
        pinocchio::forwardKinematics(model, data, q_full);
        pinocchio::updateFramePlacements(model, data);

        // ---- Send commands ----
        csv << t_sec;

        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            double tau_grav = data.tau[arm_v_idx[i]] * grav_scale;

            // Command: hold current position with low impedance + gravity feedforward
            arm[i].m->pos_vel_tqe_kp_kd(
                (float)q_motor[i],    // hold current motor position
                0.0f,                 // zero velocity ff
                (float)(motor_sign[i] * tau_grav),      // gravity compensation torque
                0.0f,
                0.0f
            );

            auto st = arm[i].m->get_current_motor_state();
            csv << "," << q_motor[i]
                << "," << q_pin[i]
                << "," << v_motor[i]
                << "," << tau_grav
                << "," << tau_grav  // tau_cmd = tau_grav (for now)
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

        // ---- Periodically save FK snapshot for visualization ----
        fk_snapshot_counter++;
        if (fk_snapshot_counter >= FK_SNAPSHOT_INTERVAL)
        {
            fk_snapshot_counter = 0;

            std::ofstream fk_out(fk_path, std::ios::trunc);
            if (fk_out.is_open())
            {
                fk_out << "frame,x,y,z" << std::endl;
                for (size_t f = 0; f < arm_frame_ids.size(); f++)
                {
                    Eigen::Vector3d pos = data.oMf[arm_frame_ids[f]].translation();
                    fk_out << ARM_FRAME_NAMES[f]
                           << "," << pos[0] << "," << pos[1] << "," << pos[2]
                           << std::endl;
                }
                // Also write motor/pinocchio angles and gravity torques
                fk_out << "---" << std::endl;
                fk_out << "joint,q_motor,q_pin,tau_grav" << std::endl;
                for (int i = 0; i < NUM_ARM_JOINTS; i++)
                {
                    fk_out << ARM_JOINT_NAMES[i]
                           << "," << q_motor[i]
                           << "," << q_pin[i]
                           << "," << data.tau[arm_v_idx[i]] * grav_scale
                           << std::endl;
                }
                fk_out.close();
            }
        }

        ros::spinOnce();
        rate.sleep();
    }

    // ---- Shutdown: ramp torques to zero ----
    ROS_INFO("Shutting down — ramping torques to zero...");
    for (int i = 0; i < (int)(1.0 * loop_rate_hz) && ros::ok(); i++)
    {
        rb.detect_motor_limit();
        double fade = 1.0 - (double)i / (1.0 * loop_rate_hz);
        for (int j = 0; j < NUM_ARM_JOINTS; j++)
        {
            auto st = arm[j].m->get_current_motor_state();
            double tau_grav = data.tau[arm_v_idx[j]] * grav_scale * fade;
            arm[j].m->pos_vel_tqe_kp_kd(
                st->position, 0.0f, (float)(motor_sign[i] * tau_grav), (float)kp, (float)kd);
        }
        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    csv.close();
    ROS_INFO("\033[1;32m--- GRAVITY COMP STOPPED ---\033[0m");
    ROS_INFO("Data: %s", csv_path.c_str());
    ROS_INFO("FK snapshot: %s", fk_path.c_str());
    ROS_INFO("Run:  python3 plot_arm_state.py --csv %s --fk %s", csv_path.c_str(), fk_path.c_str());

    return 0;
}
