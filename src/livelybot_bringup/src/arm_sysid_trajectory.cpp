// ============================================================================
// arm_sysid_trajectory.cpp
//
// Executes a pre-computed optimal excitation trajectory on the right arm
// hardware, with position control.
// Logs joint positions, velocities, accelerations, and MEASURED torques
// for offline system identification.
//
// CONTROL MODE: Pure PD position tracking
//   - Position setpoint: q_des(t) from trajectory
//   - Velocity setpoint: dq_des(t) from trajectory  
//   - Torque feedforward: 0.0
//   - Motor runs in PD mode only
//
// DATA LOGGING: MEASURED values only (from motor sensors)
//   - Measured position, velocity, torque from motor feedback
//   - Desired trajectory (q_des, dq_des, ddq_des) for reference
//
// Workflow:
//   Phase 0 (HOLD):    Hold current position for 1s, let sensors settle
//   Phase 1 (MOVE):    Slowly interpolate from current pose to q_traj(0)
//   Phase 2 (EXECUTE): Play the excitation trajectory with pure PD control
//   Phase 3 (RETURN):  Slowly return to start pose and ramp down
//
// ============================================================================

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
#include <sstream>
#include <string>
#include <vector>
#include <csignal>
#include <chrono>
#include <cmath>

static volatile bool g_running = true;
void sigint_handler(int sig) { g_running = false; }

// ---- Right arm configuration (must match gravity comp code) ----
static const std::vector<std::string> ARM_JOINT_NAMES = {
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_arm_yaw_joint",
    "r_arm_roll_joint",
    "r_wrist_yaw_joint",
};
static const int NUM_ARM_JOINTS = 5;

// ---- Trajectory data loaded from CSV ----
struct TrajectoryData {
    std::vector<double> time;                          // (N,)
    std::vector<std::array<double, 5>> q;              // (N, 5)
    std::vector<std::array<double, 5>> dq;             // (N, 5)
    std::vector<std::array<double, 5>> ddq;            // (N, 5)
    int N;
    double duration;
};

// ---- Load trajectory from CSV ----
bool load_trajectory(const std::string &path, TrajectoryData &traj)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        ROS_ERROR("Cannot open trajectory CSV: %s", path.c_str());
        return false;
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<double> vals;

        while (std::getline(ss, token, ',')) {
            vals.push_back(std::stod(token));
        }

        // Expected: t, q0..q4, dq0..dq4, ddq0..ddq4 = 16 columns
        if ((int)vals.size() < 1 + 3 * NUM_ARM_JOINTS) {
            ROS_WARN("Skipping malformed CSV row (got %d cols)", (int)vals.size());
            continue;
        }

        traj.time.push_back(vals[0]);

        std::array<double, 5> qi, dqi, ddqi;
        for (int i = 0; i < NUM_ARM_JOINTS; i++) {
            qi[i]   = vals[1 + i];
            dqi[i]  = vals[1 + NUM_ARM_JOINTS + i];
            ddqi[i] = vals[1 + 2 * NUM_ARM_JOINTS + i];
        }
        traj.q.push_back(qi);
        traj.dq.push_back(dqi);
        traj.ddq.push_back(ddqi);
    }

    traj.N = (int)traj.time.size();
    traj.duration = traj.time.back();

    file.close();
    return traj.N > 0;
}

// ---- Smooth interpolation (minimum jerk) between two poses ----
void min_jerk_interp(const double q_start[5], const double q_end[5],
                     double alpha, double q_out[5])
{
    // alpha in [0, 1]; minimum jerk profile: s = 10a^3 - 15a^4 + 6a^5
    double s = alpha * alpha * alpha * (10.0 - 15.0 * alpha + 6.0 * alpha * alpha);
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        q_out[i] = q_start[i] + s * (q_end[i] - q_start[i]);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_sysid_trajectory", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle n("~");

    // ---- Parameters ----
    double kp, kd, loop_rate_hz, move_time;
    std::string csv_path, traj_path, urdf_path;
    int num_repeats;

    n.param<double>("kp",          kp,          8.0);
    n.param<double>("kd",          kd,          0.5);
    n.param<double>("loop_rate",   loop_rate_hz, 200.0);
    n.param<double>("move_time",   move_time,   3.0);     // seconds to move to start
    n.param<int>("num_repeats",    num_repeats,  1);      // how many times to repeat trajectory
    n.param<std::string>("traj_path", traj_path, "");
    n.param<std::string>("urdf_path", urdf_path, "");
    n.param<std::string>("csv_path",  csv_path,
                         "/home/hightorque/sysid_data.csv");

    if (urdf_path.empty() || traj_path.empty()) {
        ROS_ERROR("urdf_path and traj_path params are required!");
        return -1;
    }

    // ---- Motor-space home offsets (must match gravity comp code!) ----
    double motor_home[NUM_ARM_JOINTS] = {-0.26, 0.0, 0.0, -1.0, 0.0};
    double motor_sign[NUM_ARM_JOINTS] = {-1.0, 1.0, -1.0, -1.0, 1.0};

    const double dt = 1.0 / loop_rate_hz;
    ros::Rate rate(loop_rate_hz);

    // ==== Load trajectory ====
    TrajectoryData traj;
    ROS_INFO("Loading trajectory: %s", traj_path.c_str());
    if (!load_trajectory(traj_path, traj)) {
        ROS_ERROR("Failed to load trajectory!");
        return -1;
    }
    ROS_INFO("  Loaded %d samples, duration=%.2fs", traj.N, traj.duration);

    // ==== Load Pinocchio model ====
    ROS_INFO("Loading URDF: %s", urdf_path.c_str());
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    int arm_q_idx[NUM_ARM_JOINTS];
    int arm_v_idx[NUM_ARM_JOINTS];
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        if (!model.existJointName(ARM_JOINT_NAMES[i])) {
            ROS_ERROR("Joint '%s' not found in URDF!", ARM_JOINT_NAMES[i].c_str());
            return -1;
        }
        auto jid = model.getJointId(ARM_JOINT_NAMES[i]);
        arm_q_idx[i] = model.joints[jid].idx_q();
        arm_v_idx[i] = model.joints[jid].idx_v();
    }

    // ==== Initialize robot hardware ====
    livelybot_serial::robot rb;

    struct ArmMotor { motor* m; std::string name; };
    std::vector<ArmMotor> arm;
    for (motor *m : rb.Motors) {
        if (m->get_motor_belong_canport() == 4) {
            arm.push_back({m, m->get_motor_name()});
        }
    }
    if ((int)arm.size() != NUM_ARM_JOINTS) {
        ROS_ERROR("Expected %d arm motors on CAN4, found %d!",
                  NUM_ARM_JOINTS, (int)arm.size());
        return -1;
    }

    // ==== Open data log CSV ====
    std::ofstream csv(csv_path);
    if (!csv.is_open()) {
        ROS_ERROR("Cannot open output CSV: %s", csv_path.c_str());
        return -1;
    }
    
    // ---- CSV Header ----
    // Columns:
    //   - time: elapsed time (s)
    //   - phase: current phase (0=HOLD, 1=MOVE, 2=EXECUTE, 3=RETURN)
    //   - traj_idx: index into trajectory array (-1 if not in EXECUTE phase)
    //   For each joint i (0..4):
    //     - q_des_i:   desired position (rad) from trajectory
    //     - dq_des_i:  desired velocity (rad/s) from trajectory
    //     - ddq_des_i: desired acceleration (rad/s^2) from trajectory
    //     - q_meas_i:  MEASURED position (rad) in Pinocchio frame
    //     - q_motor_i: MEASURED position (rad) in motor frame (for debugging)
    //     - dq_meas_i: MEASURED velocity (rad/s) from motor sensor
    //     - tau_meas_i: MEASURED torque (Nm) from motor sensor
    csv << "time,phase,traj_idx";
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        csv << ",q_des_" << i
            << ",dq_des_" << i
            << ",ddq_des_" << i
            << ",q_meas_" << i
            << ",q_motor_" << i
            << ",dq_meas_" << i
            << ",tau_meas_" << i;
    }
    csv << std::endl;

    // ==== Wait for initial feedback ====
    ROS_INFO("Waiting for motor feedback...");
    rb.send_get_motor_state_cmd();
    ros::Duration(0.5).sleep();

    // Read current motor positions
    double q_current_motor[NUM_ARM_JOINTS];
    double q_current_pin[NUM_ARM_JOINTS];
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        q_current_motor[i] = arm[i].m->get_current_motor_state()->position;
        q_current_pin[i] = motor_sign[i] * q_current_motor[i] - motor_home[i];
        ROS_INFO("  [%d] %s: motor=%.4f  pin=%.4f (%.1f deg)",
                 i, arm[i].name.c_str(),
                 q_current_motor[i], q_current_pin[i],
                 q_current_pin[i] * 180.0 / M_PI);
    }

    // Target start position = q_traj(0)
    double q_start_pin[NUM_ARM_JOINTS];
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        q_start_pin[i] = traj.q[0][i];
    }

    ROS_INFO("\033[1;33m=== SYSTEM IDENTIFICATION TRAJECTORY ===\033[0m");
    ROS_INFO("Control mode: POSITION CONTROL");
    ROS_INFO("PD gains: kp=%.1f  kd=%.2f", kp, kd);
    ROS_INFO("Repeats: %d", num_repeats);
    ROS_INFO("");
    ROS_INFO("Phase 0: HOLD   (1.0s) - settle sensors");
    ROS_INFO("Phase 1: MOVE   (%.1fs) - interpolate to trajectory start", move_time);
    ROS_INFO("Phase 2: EXECUTE (%.1fs x %d) - run excitation trajectory", traj.duration, num_repeats);
    ROS_INFO("Phase 3: RETURN (%.1fs) - return to initial pose", move_time);
    ROS_INFO("");
    ROS_INFO("\033[1;33mPress Ctrl+C to abort at any time.\033[0m");

    auto t_start = std::chrono::steady_clock::now();
    int print_counter = 0;

    // Phase timing
    const double T_HOLD   = 1.0;
    const double T_MOVE   = move_time;
    const double T_EXEC   = traj.duration * num_repeats;
    const double T_RETURN = move_time;

    // Save initial pose for return
    double q_init_pin[NUM_ARM_JOINTS];
    for (int i = 0; i < NUM_ARM_JOINTS; i++)
        q_init_pin[i] = q_current_pin[i];

    // ==== Main control loop ====
    while (ros::ok() && g_running) {
        rb.detect_motor_limit();

        auto t_now = std::chrono::steady_clock::now();
        double t_sec = std::chrono::duration<double>(t_now - t_start).count();

        // ---- Read MEASURED motor states ----
        double q_motor_meas[NUM_ARM_JOINTS];     // Measured position (motor frame)
        double q_pin_meas[NUM_ARM_JOINTS];       // Measured position (Pinocchio frame)
        double v_meas[NUM_ARM_JOINTS];           // Measured velocity
        double tau_meas[NUM_ARM_JOINTS];         // Measured torque

        for (int i = 0; i < NUM_ARM_JOINTS; i++) {
            auto st = arm[i].m->get_current_motor_state();
            q_motor_meas[i] = st->position;
            v_meas[i] = st->velocity;
            tau_meas[i] = st->torque;
            
            // Convert to Pinocchio frame
            q_pin_meas[i] = motor_sign[i] * q_motor_meas[i] - motor_home[i];
        }

        // ---- Determine phase and DESIRED trajectory setpoints ----
        double q_des[NUM_ARM_JOINTS]   = {};     // Desired position (for PD control)
        double dq_des[NUM_ARM_JOINTS]  = {};     // Desired velocity (for PD control)
        double ddq_des[NUM_ARM_JOINTS] = {};     // Desired acceleration (logged for reference)
        int phase = -1;
        int traj_idx = -1;

        if (t_sec < T_HOLD) {
            // ---- Phase 0: Hold current position ----
            phase = 0;
            for (int i = 0; i < NUM_ARM_JOINTS; i++) {
                q_des[i] = q_init_pin[i];
                dq_des[i] = 0.0;
                ddq_des[i] = 0.0;
            }
        }
        else if (t_sec < T_HOLD + T_MOVE) {
            // ---- Phase 1: Smooth move to trajectory start ----
            phase = 1;
            double alpha = (t_sec - T_HOLD) / T_MOVE;
            alpha = std::max(0.0, std::min(1.0, alpha));
            min_jerk_interp(q_init_pin, q_start_pin, alpha, q_des);
            // Velocity and acceleration are zero (let controller handle it)
            for (int i = 0; i < NUM_ARM_JOINTS; i++) {
                dq_des[i] = 0.0;
                ddq_des[i] = 0.0;
            }
        }
        else if (t_sec < T_HOLD + T_MOVE + T_EXEC) {
            // ---- Phase 2: Execute excitation trajectory ----
            phase = 2;
            double t_traj = t_sec - T_HOLD - T_MOVE;

            // Handle repeats: wrap around trajectory
            double t_in_period = std::fmod(t_traj, traj.duration);

            // Find closest trajectory index
            traj_idx = (int)(t_in_period / (traj.duration / (traj.N - 1)));
            traj_idx = std::max(0, std::min(traj.N - 1, traj_idx));

            // Extract desired trajectory values
            for (int i = 0; i < NUM_ARM_JOINTS; i++) {
                q_des[i]   = traj.q[traj_idx][i];
                dq_des[i]  = traj.dq[traj_idx][i];
                ddq_des[i] = traj.ddq[traj_idx][i];
            }
        }
        else if (t_sec < T_HOLD + T_MOVE + T_EXEC + T_RETURN) {
            // ---- Phase 3: Return to initial pose ----
            phase = 3;
            double alpha = (t_sec - T_HOLD - T_MOVE - T_EXEC) / T_RETURN;
            alpha = std::max(0.0, std::min(1.0, alpha));
            
            // Interpolate from last trajectory point back to initial pose
            double q_traj_end[NUM_ARM_JOINTS];
            for (int i = 0; i < NUM_ARM_JOINTS; i++)
                q_traj_end[i] = traj.q[traj.N - 1][i];
            
            min_jerk_interp(q_traj_end, q_init_pin, alpha, q_des);
            
            for (int i = 0; i < NUM_ARM_JOINTS; i++) {
                dq_des[i] = 0.0;
                ddq_des[i] = 0.0;
            }
        }
        else {
            // ---- All phases complete ----
            ROS_INFO("\033[1;32mTrajectory complete!\033[0m");
            break;
        }

        // ---- Send motor commands ----
        // Command structure: pos_vel_tqe_kp_kd(position, velocity, torque_ff, kp, kd)
        // We set torque_ff = 0.0
        for (int i = 0; i < NUM_ARM_JOINTS; i++) {
            // Convert desired position and velocity to motor frame
            double q_des_motor = motor_sign[i] * (q_des[i] + motor_home[i]);
            double dq_des_motor = motor_sign[i] * dq_des[i];
            
            // Command: PD control
            arm[i].m->pos_vel_tqe_kp_kd(
                (float)q_des_motor,      // Position setpoint (motor frame)
                (float)dq_des_motor,     // Velocity setpoint (motor frame)
                0.0f,                    // Torque feedforward = 0.0
                (float)kp,               // Position gain
                (float)kd                // Velocity gain
            );
        }

        // ---- Log data to CSV ----
        // Log: time, phase, trajectory index, desired values, MEASURED values
        csv << t_sec << "," << phase << "," << traj_idx;
        for (int i = 0; i < NUM_ARM_JOINTS; i++) {
            csv << "," << q_des[i]           // Desired position (reference)
                << "," << dq_des[i]          // Desired velocity (reference)
                << "," << ddq_des[i]         // Desired acceleration (reference)
                << "," << q_pin_meas[i]      // MEASURED position (Pinocchio frame)
                << "," << q_motor_meas[i]    // MEASURED position (motor frame, for debug)
                << "," << v_meas[i]          // MEASURED velocity
                << "," << tau_meas[i];       // MEASURED torque
        }
        csv << std::endl;

        // ---- Periodic console output ----
        print_counter++;
        if (print_counter >= (int)loop_rate_hz) {
            print_counter = 0;
            const char* phase_names[] = {"HOLD", "MOVE", "EXEC", "RETURN"};
            ROS_INFO("[t=%.1f] Phase %d (%s) traj_idx=%d",
                     t_sec, phase, phase_names[phase], traj_idx);
            for (int i = 0; i < NUM_ARM_JOINTS; i++) {
                ROS_INFO("  [%d] q_des=%.3f q_meas=%.3f err=%.4f v_meas=%.3f tau_meas=%.3f",
                         i, q_des[i], q_pin_meas[i], q_des[i] - q_pin_meas[i],
                         v_meas[i], tau_meas[i]);
            }
        }

        // Send commands and maintain loop rate
        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    // ==== Shutdown: ramp gains to zero for safe stop ====
    ROS_INFO("Shutting down - ramping gains to zero...");
    for (int i = 0; i < (int)(1.0 * loop_rate_hz) && ros::ok(); i++) {
        rb.detect_motor_limit();
        
        // Linear fade of gains over 1 second
        double fade = 1.0 - (double)i / (1.0 * loop_rate_hz);
        
        for (int j = 0; j < NUM_ARM_JOINTS; j++) {
            auto st = arm[j].m->get_current_motor_state();
            
            // Hold current position, but with decreasing gains
            arm[j].m->pos_vel_tqe_kp_kd(
                st->position,              // Hold current position
                0.0f,                      // Zero velocity
                0.0f,                      // Zero torque feedforward
                (float)(kp * fade),        // Fade position gain
                (float)(kd * fade)         // Fade velocity gain
            );
        }
        
        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    csv.close();
    
    ROS_INFO("\033[1;32m=== SYSID TRAJECTORY COMPLETE ===\033[0m");
    ROS_INFO("Data saved: %s", csv_path.c_str());
    ROS_INFO("");
    ROS_INFO("CSV columns:");
    ROS_INFO("  - time, phase, traj_idx");
    ROS_INFO("  - For each joint: q_des, dq_des, ddq_des, q_meas, q_motor, dq_meas, tau_meas");
    ROS_INFO("");
    ROS_INFO("Next step:");
    ROS_INFO("  python3 solve_sysid.py --urdf <urdf> --csv %s", csv_path.c_str());

    return 0;
}
