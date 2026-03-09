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

// ============================================================
// CONFIGURE THESE:
// ============================================================
// Which motor index (0-4) to test
static const int TEST_MOTOR_IDX = 4;

// Torque magnitude to apply (Nm) — keep small for safety!
static const double TEST_TORQUE = 0.3;

// How long to apply torque in each direction (seconds)
static const double TORQUE_DURATION = 2.0;

// How long to hold still before saving snapshot (seconds)
// (lets the arm settle after torque is applied)
static const double SETTLE_DURATION = 0.5;
// ============================================================

enum TestPhase {
    PHASE_INIT,          // hold all motors, read baseline
    PHASE_POS_TORQUE,    // apply +torque to test motor
    PHASE_POS_SETTLE,    // hold position, let arm settle
    PHASE_POS_SNAPSHOT,  // save FK snapshot
    PHASE_RETURN_1,      // return to start position
    PHASE_NEG_TORQUE,    // apply -torque to test motor
    PHASE_NEG_SETTLE,    // hold position, let arm settle
    PHASE_NEG_SNAPSHOT,  // save FK snapshot
    PHASE_RETURN_2,      // return to start position
    PHASE_DONE,
};

static const char* phase_name(TestPhase p) {
    switch(p) {
        case PHASE_INIT:         return "INIT";
        case PHASE_POS_TORQUE:   return "POS_TORQUE";
        case PHASE_POS_SETTLE:   return "POS_SETTLE";
        case PHASE_POS_SNAPSHOT: return "POS_SNAPSHOT";
        case PHASE_RETURN_1:     return "RETURN_1";
        case PHASE_NEG_TORQUE:   return "NEG_TORQUE";
        case PHASE_NEG_SETTLE:   return "NEG_SETTLE";
        case PHASE_NEG_SNAPSHOT: return "NEG_SNAPSHOT";
        case PHASE_RETURN_2:     return "RETURN_2";
        case PHASE_DONE:         return "DONE";
        default:                 return "???";
    }
}

void save_fk_snapshot(const std::string &path,
                      const std::string &label,
                      int test_idx,
                      double applied_torque,
                      const pinocchio::Model &model,
                      pinocchio::Data &data,
                      const Eigen::VectorXd &q_full,
                      const std::vector<pinocchio::FrameIndex> &arm_frame_ids,
                      const double q_motor[],
                      const double q_pin[],
                      const int arm_v_idx[])
{
    // Run FK
    pinocchio::forwardKinematics(model, data, q_full);
    pinocchio::updateFramePlacements(model, data);

    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) {
        ROS_ERROR("Cannot open %s for writing!", path.c_str());
        return;
    }

    out << "# Motor sign test: " << label << std::endl;
    out << "# Test motor index: " << test_idx
        << " (" << ARM_JOINT_NAMES[test_idx] << ")" << std::endl;
    out << "# Applied torque (motor space): " << applied_torque << " Nm" << std::endl;
    out << std::endl;

    // Frame positions
    out << "frame,x,y,z" << std::endl;
    for (size_t f = 0; f < arm_frame_ids.size(); f++) {
        Eigen::Vector3d pos = data.oMf[arm_frame_ids[f]].translation();
        out << ARM_FRAME_NAMES[f]
            << "," << pos[0] << "," << pos[1] << "," << pos[2]
            << std::endl;
    }

    out << std::endl;
    out << "---" << std::endl;
    out << "joint,q_motor,q_pin,q_pin_deg" << std::endl;
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        out << ARM_JOINT_NAMES[i]
            << "," << q_motor[i]
            << "," << q_pin[i]
            << "," << q_pin[i] * 180.0 / M_PI
            << std::endl;
    }
    out.close();
    ROS_INFO("Saved FK snapshot: %s", path.c_str());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_motor_sign_test", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle n("~");

    // ---- Parameters ----
    double loop_rate_hz;
    std::string urdf_path, output_dir;

    n.param<double>("loop_rate", loop_rate_hz, 200.0);
    n.param<std::string>("urdf_path", urdf_path, "");
    n.param<std::string>("output_dir", output_dir, "/home/hightorque/");

    if (urdf_path.empty()) {
        ROS_ERROR("urdf_path param is required!");
        return -1;
    }

    // ---- Motor-space mappings (same as gravity comp) ----
    // double motor_home[NUM_ARM_JOINTS] = {-0.26, 0.0, 0.0, -1.57, 0.0};
    double motor_home[NUM_ARM_JOINTS] = {-0.26, 0.0, 0.0, -1.0, 0.0};
    double motor_sign[NUM_ARM_JOINTS] = {-1.0, 1.0, -1.0, -1.0, 1.0};

    const double dt = 1.0 / loop_rate_hz;
    ros::Rate rate(loop_rate_hz);

    // Validate test motor index
    if (TEST_MOTOR_IDX < 0 || TEST_MOTOR_IDX >= NUM_ARM_JOINTS) {
        ROS_ERROR("TEST_MOTOR_IDX=%d out of range [0,%d)!", TEST_MOTOR_IDX, NUM_ARM_JOINTS);
        return -1;
    }

    ROS_INFO("============================================");
    ROS_INFO("  MOTOR SIGN VERIFICATION TEST");
    ROS_INFO("  Testing motor %d: %s", TEST_MOTOR_IDX, ARM_JOINT_NAMES[TEST_MOTOR_IDX].c_str());
    ROS_INFO("  Torque: +/- %.2f Nm", TEST_TORQUE);
    ROS_INFO("  Duration: %.1f s each direction", TORQUE_DURATION);
    ROS_INFO("  motor_sign[%d] = %.1f", TEST_MOTOR_IDX, motor_sign[TEST_MOTOR_IDX]);
    ROS_INFO("============================================");

    // ==== Load Pinocchio model ====
    ROS_INFO("Loading URDF: %s", urdf_path.c_str());
    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::Data data(model);

    // Find arm joint indices in the Pinocchio model
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

    // Find arm frame IDs for FK
    std::vector<pinocchio::FrameIndex> arm_frame_ids;
    for (const auto &fname : ARM_FRAME_NAMES) {
        if (!model.existFrame(fname)) {
            ROS_WARN("Frame '%s' not found, using 0", fname.c_str());
            arm_frame_ids.push_back(0);
        } else {
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

    for (motor *m : rb.Motors) {
        if (m->get_motor_belong_canport() == 4) {
            arm.push_back({m, m->get_motor_name()});
        }
    }

    if ((int)arm.size() != NUM_ARM_JOINTS) {
        ROS_ERROR("Expected %d arm motors, found %d!", NUM_ARM_JOINTS, (int)arm.size());
        return -1;
    }

    // ==== Wait for initial feedback ====
    ROS_INFO("Waiting for motor feedback...");
    rb.send_get_motor_state_cmd();
    ros::Duration(0.5).sleep();

    // Record starting positions (we'll hold non-test motors here)
    double start_motor_pos[NUM_ARM_JOINTS];
    for (int i = 0; i < NUM_ARM_JOINTS; i++) {
        start_motor_pos[i] = arm[i].m->get_current_motor_state()->position;
        ROS_INFO("  Motor %d (%s): start_pos=%.4f", i, arm[i].name.c_str(), start_motor_pos[i]);
    }

    // Pinocchio state vectors
    Eigen::VectorXd q_full = Eigen::VectorXd::Zero(model.nq);

    // State machine
    TestPhase phase = PHASE_INIT;
    auto phase_start = std::chrono::steady_clock::now();
    auto t_start     = std::chrono::steady_clock::now();

    // Hold position for test motor after torque phase
    double hold_pos_after_torque = start_motor_pos[TEST_MOTOR_IDX];

    // Stiffness for holding non-test motors in place
    const float HOLD_KP = 3.0f;
    const float HOLD_KD = 1.0f;

    ROS_INFO("\033[1;33mStarting test. Press Ctrl+C to abort.\033[0m");

    // Open CSV log for the full test timeline
    std::string csv_path = output_dir + "motor_sign_test_timeline.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "time,phase,test_motor_torque";
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
            csv << "," << arm[i].name << "_motor_pos"
                << "," << arm[i].name << "_pin_pos";
        for (const auto &fname : ARM_FRAME_NAMES)
            csv << "," << fname << "_x," << fname << "_y," << fname << "_z";
        csv << std::endl;
    }

    while (ros::ok() && g_running && phase != PHASE_DONE)
    {
        rb.detect_motor_limit();

        auto t_now = std::chrono::steady_clock::now();
        double t_sec = std::chrono::duration<double>(t_now - t_start).count();
        double phase_elapsed = std::chrono::duration<double>(t_now - phase_start).count();

        // ---- Read motor states ----
        double q_motor[NUM_ARM_JOINTS];
        double q_pin[NUM_ARM_JOINTS];

        for (int i = 0; i < NUM_ARM_JOINTS; i++) {
            auto st = arm[i].m->get_current_motor_state();
            q_motor[i] = st->position;
            q_pin[i] = motor_sign[i] * q_motor[i] - motor_home[i];
            q_full[arm_q_idx[i]] = q_pin[i];
        }

        // Compute FK for logging
        pinocchio::forwardKinematics(model, data, q_full);
        pinocchio::updateFramePlacements(model, data);

        // ---- State machine ----
        double test_torque_cmd = 0.0;

        switch (phase)
        {
        case PHASE_INIT:
            // Hold everything still for 1 second to establish baseline
            if (phase_elapsed >= 1.0) {
                ROS_INFO("Baseline established. Starting positive torque...");
                phase = PHASE_POS_TORQUE;
                phase_start = t_now;
            }
            break;

        case PHASE_POS_TORQUE:
            test_torque_cmd = +TEST_TORQUE;
            if (phase_elapsed >= TORQUE_DURATION) {
                // Record where the motor ended up
                hold_pos_after_torque = q_motor[TEST_MOTOR_IDX];
                ROS_INFO("Positive torque done. Motor moved to %.4f (started at %.4f, delta=%.4f)",
                         hold_pos_after_torque, start_motor_pos[TEST_MOTOR_IDX],
                         hold_pos_after_torque - start_motor_pos[TEST_MOTOR_IDX]);
                phase = PHASE_POS_SETTLE;
                phase_start = t_now;
            }
            break;

        case PHASE_POS_SETTLE:
            // Hold at current position, let vibrations die
            if (phase_elapsed >= SETTLE_DURATION) {
                phase = PHASE_POS_SNAPSHOT;
                phase_start = t_now;
            }
            break;

        case PHASE_POS_SNAPSHOT:
        {
            std::string path = output_dir + "fk_positive_torque.csv";
            save_fk_snapshot(path, "POSITIVE TORQUE (+" + std::to_string(TEST_TORQUE) + " Nm)",
                             TEST_MOTOR_IDX, +TEST_TORQUE,
                             model, data, q_full, arm_frame_ids,
                             q_motor, q_pin, arm_v_idx);
            ROS_INFO("\033[1;32mPositive torque snapshot saved!\033[0m");
            ROS_INFO("Returning to start position...");
            phase = PHASE_RETURN_1;
            phase_start = t_now;
            break;
        }

        case PHASE_RETURN_1:
            // Hold at start position for 2 seconds
            if (phase_elapsed >= 2.0) {
                ROS_INFO("Starting negative torque...");
                phase = PHASE_NEG_TORQUE;
                phase_start = t_now;
            }
            break;

        case PHASE_NEG_TORQUE:
            test_torque_cmd = -TEST_TORQUE;
            if (phase_elapsed >= TORQUE_DURATION) {
                hold_pos_after_torque = q_motor[TEST_MOTOR_IDX];
                ROS_INFO("Negative torque done. Motor moved to %.4f (started at %.4f, delta=%.4f)",
                         hold_pos_after_torque, start_motor_pos[TEST_MOTOR_IDX],
                         hold_pos_after_torque - start_motor_pos[TEST_MOTOR_IDX]);
                phase = PHASE_NEG_SETTLE;
                phase_start = t_now;
            }
            break;

        case PHASE_NEG_SETTLE:
            if (phase_elapsed >= SETTLE_DURATION) {
                phase = PHASE_NEG_SNAPSHOT;
                phase_start = t_now;
            }
            break;

        case PHASE_NEG_SNAPSHOT:
        {
            std::string path = output_dir + "fk_negative_torque.csv";
            save_fk_snapshot(path, "NEGATIVE TORQUE (-" + std::to_string(TEST_TORQUE) + " Nm)",
                             TEST_MOTOR_IDX, -TEST_TORQUE,
                             model, data, q_full, arm_frame_ids,
                             q_motor, q_pin, arm_v_idx);
            ROS_INFO("\033[1;32mNegative torque snapshot saved!\033[0m");
            ROS_INFO("Returning to start...");
            phase = PHASE_RETURN_2;
            phase_start = t_now;
            break;
        }

        case PHASE_RETURN_2:
            if (phase_elapsed >= 2.0) {
                phase = PHASE_DONE;
            }
            break;

        case PHASE_DONE:
            break;
        }

        // ---- Send motor commands ----
        for (int i = 0; i < NUM_ARM_JOINTS; i++)
        {
            if (i == TEST_MOTOR_IDX)
            {
                if (phase == PHASE_POS_TORQUE || phase == PHASE_NEG_TORQUE)
                {
                    // Apply pure torque, no position control
                    // kp=0, kd=small for some damping
                    arm[i].m->pos_vel_tqe_kp_kd(
                        (float)q_motor[i],   // doesn't matter with kp=0
                        0.0f,
                        (float)test_torque_cmd,
                        0.0f,                // no position stiffness
                        0.3f                 // light damping so it doesn't fly
                    );
                }
                else if (phase == PHASE_POS_SETTLE || phase == PHASE_NEG_SETTLE ||
                         phase == PHASE_POS_SNAPSHOT || phase == PHASE_NEG_SNAPSHOT)
                {
                    // Hold at wherever it ended up
                    arm[i].m->pos_vel_tqe_kp_kd(
                        (float)hold_pos_after_torque,
                        0.0f, 0.0f,
                        HOLD_KP, HOLD_KD
                    );
                }
                else
                {
                    // Hold at start position (INIT, RETURN phases)
                    arm[i].m->pos_vel_tqe_kp_kd(
                        (float)start_motor_pos[i],
                        0.0f, 0.0f,
                        HOLD_KP, HOLD_KD
                    );
                }
            }
            else
            {
                // Non-test motors: always hold at start position
                arm[i].m->pos_vel_tqe_kp_kd(
                    (float)start_motor_pos[i],
                    0.0f, 0.0f,
                    HOLD_KP, HOLD_KD
                );
            }
        }

        // ---- Log to CSV ----
        if (csv.is_open()) {
            csv << t_sec << "," << phase_name(phase) << "," << test_torque_cmd;
            for (int i = 0; i < NUM_ARM_JOINTS; i++)
                csv << "," << q_motor[i] << "," << q_pin[i];
            for (size_t f = 0; f < arm_frame_ids.size(); f++) {
                Eigen::Vector3d pos = data.oMf[arm_frame_ids[f]].translation();
                csv << "," << pos[0] << "," << pos[1] << "," << pos[2];
            }
            csv << std::endl;
        }

        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    // ---- Shutdown: release all motors gently ----
    ROS_INFO("Releasing motors...");
    for (int i = 0; i < (int)(1.0 * loop_rate_hz) && ros::ok(); i++) {
        rb.detect_motor_limit();
        double fade = 1.0 - (double)i / (1.0 * loop_rate_hz);
        for (int j = 0; j < NUM_ARM_JOINTS; j++) {
            auto st = arm[j].m->get_current_motor_state();
            arm[j].m->pos_vel_tqe_kp_kd(
                st->position, 0.0f, 0.0f,
                (float)(HOLD_KP * fade), (float)(HOLD_KD * fade));
        }
        rb.motor_send_2();
        ros::spinOnce();
        rate.sleep();
    }

    csv.close();

    ROS_INFO("============================================");
    ROS_INFO("  TEST COMPLETE for motor %d (%s)", TEST_MOTOR_IDX, ARM_JOINT_NAMES[TEST_MOTOR_IDX].c_str());
    ROS_INFO("  motor_sign[%d] = %.1f", TEST_MOTOR_IDX, motor_sign[TEST_MOTOR_IDX]);
    ROS_INFO("  Outputs:");
    ROS_INFO("    %sfk_positive_torque.csv", output_dir.c_str());
    ROS_INFO("    %sfk_negative_torque.csv", output_dir.c_str());
    ROS_INFO("    %smotor_sign_test_timeline.csv", output_dir.c_str());
    ROS_INFO("");
    ROS_INFO("  HOW TO CHECK:");
    ROS_INFO("  Compare the FK frame positions in both files.");
    ROS_INFO("  If +torque in motor space moves the arm in the SAME");
    ROS_INFO("  direction as Pinocchio's +q for this joint,");
    ROS_INFO("  then motor_sign = +1 is correct.");
    ROS_INFO("  If it moves OPPOSITE, motor_sign should be -1.");
    ROS_INFO("============================================");

    return 0;
}
