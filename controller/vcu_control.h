// Launch / traction control, written to run unchanged on the VCU and inside the
// Python accel sim (accel/firmware.py loads it through this C ABI).
//
// Rules so it stays portable to the Teensy:
//   - no heap, no Arduino, no printf; time is passed in, never read
//   - inputs are what the VCU actually receives: wheel speed sensor edges and
//     decoded CAN signals, flagged when a new frame arrived
//   - everything the controller "believes" about the car lives in vcu_config_t,
//     so the sim can give the plant different (true) values
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum vcu_mode {
  VCU_MODE_PASSTHROUGH = 0,  // driver request straight to the inverter
  VCU_MODE_SLIP_PI = 1,      // slip-target PI scaling the driver request
  VCU_MODE_MU_ADAPTIVE = 2,  // grip map from estimated mu + slip PI trim + mu learning
};

enum { VCU_FL = 0, VCU_FR = 1, VCU_RL = 2, VCU_RR = 3 };

typedef struct {
  int32_t mode;

  // car model (the controller's belief, not necessarily the truth)
  float mass;              // kg
  float cg_height;         // m
  float wheelbase;         // m
  float cg_to_front;       // m, front axle to CG
  float tire_radius;       // m
  float final_drive;       // motor rev / wheel rev
  float driveline_eff;
  float rear_inertia;      // kg m^2 at the rear axle (wheels + rotor * gr^2)
  float cla, cda, rho, crr, aero_front;
  float torque_peak;       // Nm, never command more

  // sensing
  int32_t ws_teeth;            // wheel speed sensor teeth per rev
  uint32_t ws_min_window_us;   // average edges over at least this long
  uint32_t ws_timeout_us;      // no edge for this long -> wheel stopped
  float ws_valid_speed;        // m/s, front wheel speed trusted above this
  float vn_vel_gain;           // 0..1 blend toward VectorNav velocity per frame
  float ws_vel_gain;           // 0..1 blend toward front wheel speed per control step
  float accel_filter_tau;      // s, low-pass on VectorNav accel

  // control
  uint32_t control_period_us;  // torque command period
  float slip_target;
  float slip_speed_floor;      // m/s, slip denominator floor at low speed
  float kp, ki;                // on slip error normalized by target
  float recover_gain;          // ki multiplier while under target
  float scale_max;             // MU_ADAPTIVE: PI may push this far above the map
  float mu_init;               // MU_ADAPTIVE: starting grip estimate (be conservative)
  float mu_min, mu_max;
  float mu_adapt_rate;         // 1/s, how fast mu_hat tracks observed grip
} vcu_config_t;

typedef struct {
  uint32_t now_us;

  // wheel speed sensors: input-capture edges since the last step
  uint16_t ws_edges[4];
  uint32_t ws_last_edge_us[4];   // time of the most recent of those edges

  // CAN: latest decoded values; *_new = 1 if a frame arrived since last step
  uint8_t motor_rpm_new;
  float motor_rpm;
  uint8_t vn_new;
  float vn_vel_x;                // m/s, body forward velocity
  float vn_accel_x;              // m/s^2, body forward acceleration

  float driver_torque_nm;        // from the pedal map
} vcu_inputs_t;

typedef struct {
  float torque_cmd_nm;           // what goes in the inverter command frame

  // diagnostics (log these on the car too)
  float wheel_omega[4];          // rad/s per wheel sensor
  float v_est;                   // m/s ground speed estimate
  float accel_est;               // m/s^2 filtered
  float slip;                    // measured slip ratio fed to the controller
  float scale;                   // controller multiplier
  float mu_hat;                  // MU_ADAPTIVE grip estimate
  float torque_ff_nm;            // MU_ADAPTIVE map torque
} vcu_outputs_t;

size_t vcu_state_size(void);
void vcu_init(void *state, const vcu_config_t *config, uint32_t now_us);
void vcu_step(void *state, const vcu_inputs_t *in, vcu_outputs_t *out);

#ifdef __cplusplus
}
#endif
