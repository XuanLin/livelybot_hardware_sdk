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

// Global flag for clean shutdown
static volatile bool g_running = true;
void sigint_handler(int sig) { g_running = false; }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_joint_survey", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle n("~");

    // ---- Tunable parameters (override via launch file) ----
    double kp, kd, loop_rate_hz;
    double sweep_amplitude;   // how far to move each joint (rad)
    double hold_time;         // seconds to hold at each position
    double slew_rate;         // rad/s — max speed to ramp toward target
    std::string csv_path;

    n.param<double>("kp",              kp,              20.0);
    n.param<double>("kd",              kd,              1.0);
    n.param<double>("loop_rate",       loop_rate_hz,    200.0);
    n.param<double>("sweep_amplitude", sweep_amplitude, 0.3);   // ~17 deg
    n.param<double>("hold_time",       hold_time,       3.0);
    n.param<double>("slew_rate",       slew_rate,       0.2);   // rad/s ramp
    n.param<std::string>("csv_path",   csv_path,
                         "/home/hightorque/arm_joint_survey.csv");

    const double dt = 1.0 / loop_rate_hz;
    ros::Rate r(loop_rate_hz);
    livelybot_serial::robot rb;

    // ---- Identify right arm motors (CANport 4) ----
    struct ArmMotor {
        motor* m;
        std::string name;
        double cmd_pos;   // current commanded position (slewed)
    };
    std::vector<ArmMotor> arm;

    // Expected URDF joint mapping (for the printout)
    const char* urdf_names[] = {
        "r_shoulder_pitch (axis Y)",
        "r_shoulder_roll  (axis X)",
        "r_arm_yaw        (axis Z)",
        "r_arm_roll       (axis X)",
        "r_wrist_yaw      (axis Z)"
    };

    for (motor *m : rb.Motors)
    {
        if (m->get_motor_belong_canport() == 4)
        {
            int idx = (int)arm.size();
            const char* guess = (idx < 5) ? urdf_names[idx] : "???";
            arm.push_back({m, m->get_motor_name(), 0.0});
            ROS_INFO("Motor %d: %-12s  (type=%d, id=%d)  ->  URDF: %s",
                     idx, m->get_motor_name().c_str(),
                     m->get_motor_type(), m->get_motor_id(), guess);
        }
    }

    if (arm.empty())
    {
        ROS_ERROR("No right arm motors found on CANport 4!");
        return -1;
    }

    ROS_INFO("Found %d right arm motors", (int)arm.size());
    ROS_INFO("Params: kp=%.1f  kd=%.1f  amplitude=%.2f rad (%.1f deg)  "
             "hold=%.1fs  slew=%.2f rad/s",
             kp, kd, sweep_amplitude, sweep_amplitude * 180.0 / M_PI,
             hold_time, slew_rate);

    // ---- Open CSV ----
    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
        ROS_ERROR("Cannot open CSV: %s", csv_path.c_str());
        return -1;
    }
    csv << "time,phase,active_motor";
    for (size_t i = 0; i < arm.size(); i++)
        csv << "," << arm[i].name << "_cmd"
            << "," << arm[i].name << "_pos"
            << "," << arm[i].name << "_vel"
            << "," << arm[i].name << "_torque";
    csv << std::endl;

    // ---- Wait for feedback, capture current positions ----
    ROS_INFO("Waiting for motor feedback...");
    rb.send_get_motor_state_cmd();
    ros::Duration(0.5).sleep();

    for (auto &am : arm)
    {
        am.cmd_pos = am.m->get_current_motor_state()->position;
        ROS_INFO("  %s current pos: %.4f rad", am.name.c_str(), am.cmd_pos);
    }

    // Home position = URDF zero in motor space
    // double home[] = {0.0, -0.78, 1.57, 0.78, 0.0};
    double home[] = {0.0, -0.78, 1.57, 0.5, 0.0};

    // ---- Build phase schedule ----
    // Phase 0: ramp all joints to 0 and hold
    // Phase 1..N: for each joint i, ramp to +amp, hold, ramp to -amp, hold,
    //             ramp back to 0, hold
    struct Phase {
        std::string label;
        int    active_motor;        // -1 = all
        double target;              // target for the active motor
        double duration;            // seconds
    };

    std::vector<Phase> phases;

    // Phase 0: go to zero
    phases.push_back({"ALL_TO_ZERO", -1, 0.0, 6.0});

    // Per-joint sweeps
    for (size_t i = 0; i < arm.size(); i++)
    {
        std::string base = arm[i].name;
        phases.push_back({base + "_POS",  (int)i,  home[i] + sweep_amplitude, hold_time});
        phases.push_back({base + "_HOME", (int)i,  home[i],                   hold_time});
        phases.push_back({base + "_NEG",  (int)i,  home[i] - sweep_amplitude, hold_time});
        phases.push_back({base + "_HOME", (int)i,  home[i],                   hold_time});

        // phases.push_back({base + "_POS",  (int)i,  sweep_amplitude, hold_time});
        // phases.push_back({base + "_ZERO", (int)i,  0.0,             hold_time});
        // phases.push_back({base + "_NEG",  (int)i, -sweep_amplitude, hold_time});
        // phases.push_back({base + "_ZERO", (int)i,  0.0,             hold_time});
    }

    ROS_INFO("\033[1;32m--- ARM JOINT SURVEY STARTED ---\033[0m");
    ROS_INFO("Total phases: %d.  Press Ctrl+C to abort at any time.",
             (int)phases.size());

    auto t_start = std::chrono::steady_clock::now();

    // ---- Helper: slew a value toward a target ----
    auto slew = [&](double current, double target) -> double {
        double step = slew_rate * dt;
        if (target > current)
            return std::min(current + step, target);
        else
            return std::max(current - step, target);
    };

    // ---- Main loop: iterate through phases ----
    size_t phase_idx = 0;
    double phase_start_time = 0.0;

    // Set all initial targets to 0 for phase 0
    // std::vector<double> targets(arm.size(), 0.0);
    std::vector<double> targets(home, home + arm.size());

    while (ros::ok() && g_running && phase_idx < phases.size())
    {
        rb.detect_motor_limit();

        auto t_now = std::chrono::steady_clock::now();
        double t_sec = std::chrono::duration<double>(t_now - t_start).count();

        const Phase& ph = phases[phase_idx];

        // Update target for current phase
        if (ph.active_motor >= 0)
            targets[ph.active_motor] = ph.target;
        else  // all to zero
            // std::fill(targets.begin(), targets.end(), 0.0);
            for (size_t i = 0; i < arm.size(); i++) targets[i] = home[i];

        // Slew each motor's commanded position toward its target
        for (size_t i = 0; i < arm.size(); i++)
            arm[i].cmd_pos = slew(arm[i].cmd_pos, targets[i]);

        // Send commands and log
        csv << t_sec << "," << ph.label << "," << ph.active_motor;

        for (size_t i = 0; i < arm.size(); i++)
        {
            arm[i].m->pos_vel_tqe_kp_kd(
                (float)arm[i].cmd_pos,  // position target
                0.0f,                   // velocity ff
                0.0f,                   // torque ff
                (float)kp,
                (float)kd
            );

            auto st = arm[i].m->get_current_motor_state();
            csv << "," << arm[i].cmd_pos
                << "," << st->position
                << "," << st->velocity
                << "," << st->torque;
        }
        csv << std::endl;

        rb.motor_send_2();

        // Check phase completion: all motors near target + min time elapsed
        double phase_elapsed = t_sec - phase_start_time;
        bool all_near = true;
        for (size_t i = 0; i < arm.size(); i++)
        {
            if (std::fabs(arm[i].cmd_pos - targets[i]) > 1e-3)
            {
                all_near = false;
                break;
            }
        }

        if (all_near && phase_elapsed >= ph.duration)
        {
            ROS_INFO("Phase %2zu/%zu done: %-20s  (%.1fs)",
                     phase_idx + 1, phases.size(),
                     ph.label.c_str(), phase_elapsed);
            phase_idx++;
            phase_start_time = t_sec;
        }

        ros::spinOnce();
        r.sleep();
    }

    // ---- Return to zero on exit ----
    ROS_INFO("Returning all joints to zero...");
    for (int i = 0; i < (int)(3.0 * loop_rate_hz) && ros::ok(); i++)
    {
        rb.detect_motor_limit();

        for (size_t i = 0; i < arm.size(); i++)
        {
            arm[i].cmd_pos = slew(arm[i].cmd_pos, home[i]);
            arm[i].m->pos_vel_tqe_kp_kd(
                (float)arm[i].cmd_pos, 0.0f, 0.0f, (float)kp, (float)kd);
        }

        // for (auto &am : arm)
        // {
        //     am.cmd_pos = slew(am.cmd_pos, 0.0);
        //     am.m->pos_vel_tqe_kp_kd(
        //         (float)am.cmd_pos, 0.0f, 0.0f, (float)kp, (float)kd);
        // }
        rb.motor_send_2();
        ros::spinOnce();
        r.sleep();
    }

    csv.close();
    ROS_INFO("\033[1;32m--- SURVEY COMPLETE ---\033[0m");
    ROS_INFO("Data saved to: %s", csv_path.c_str());
    ROS_INFO(" ");
    ROS_INFO("=== WHAT TO LOOK FOR ===");
    ROS_INFO("1) Zero position: where did the arm sit after ALL_TO_ZERO?");
    ROS_INFO("2) For each joint, +amplitude moved which direction?");
    ROS_INFO("   URDF expects positive = right-hand rule around joint axis:");
    for (int i = 0; i < std::min((int)arm.size(), 5); i++)
        ROS_INFO("     Motor %d (%s) -> %s", i, arm[i].name.c_str(), urdf_names[i]);
    ROS_INFO("3) If a motor moved the WRONG way, we need a sign flip for that joint.");

    return 0;
}
