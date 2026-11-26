#include "vcu_control.h"

#include <math.h>
#include <new>

namespace {

constexpr float G = 9.81f;
constexpr float TWO_PI = 6.28318530718f;

float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// Wheel speed from input-capture edge times. Averages over at least
// ws_min_window_us (several teeth at speed, one tooth period at crawl). Between
// edges the speed can't be higher than one tooth per time-since-last-edge, so
// the estimate decays toward 0 smoothly instead of holding and then timing out.
struct WheelSpeed {
  bool have_ref = false;
  uint32_t ref_us = 0;
  uint32_t last_us = 0;
  uint32_t edges = 0;
  float omega = 0.0f;

  void update(uint16_t new_edges, uint32_t last_edge_us, uint32_t now_us,
              const vcu_config_t &c) {
    const float tooth = TWO_PI / (float)c.ws_teeth;
    if (new_edges > 0) {
      if (!have_ref) {
        have_ref = true;
        ref_us = last_edge_us;
        edges = 0;
      } else {
        edges += new_edges;
        uint32_t span = last_edge_us - ref_us;
        if (span >= c.ws_min_window_us && span > 0) {
          omega = (float)edges * tooth / ((float)span * 1e-6f);
          ref_us = last_edge_us;
          edges = 0;
        }
      }
      last_us = last_edge_us;
    }
    if (have_ref) {
      uint32_t since = now_us - last_us;
      if (since > c.ws_timeout_us) {
        have_ref = false;
        omega = 0.0f;
      } else if (since > 0) {
        float bound = tooth / ((float)since * 1e-6f);
        if (bound < omega) omega = bound;
      }
    }
  }
};

struct State {
  vcu_config_t cfg;
  WheelSpeed ws[4];
  uint32_t last_us;
  uint32_t next_control_us;

  float motor_rpm;
  float v_est;
  float accel_raw;
  float accel_est;

  float integral;
  float scale;
  float mu_hat;
  float torque_cmd;
  float torque_ff;
  float slip;

  // rear load (N) at a given acceleration and speed, from the configured model
  float rear_load(float accel, float v) const {
    const vcu_config_t &c = cfg;
    float statics = c.mass * G * c.cg_to_front / c.wheelbase;
    float transfer = c.mass * accel * c.cg_height / c.wheelbase;
    float downforce = 0.5f * c.rho * c.cla * v * v * (1.0f - c.aero_front);
    return statics + transfer + downforce;
  }

  // PI on normalized slip error. Integral ranges [-1, hi - 1] so the output
  // scale is in [0, hi]. Stops winding down while pinned at full cut.
  void run_pi(float hi, float dt) {
    const vcu_config_t &c = cfg;
    float err = (c.slip_target - slip) / c.slip_target;
    float ki = c.ki * (err > 0 ? c.recover_gain : 1.0f);
    float trial = clampf(integral + ki * err * dt, -1.0f, hi - 1.0f);
    if (!(c.kp * err + trial < -1.0f && err < 0)) integral = trial;
    scale = clampf(1.0f + c.kp * err + integral, 0.0f, hi);
  }

  void control(float driver, float dt) {
    const vcu_config_t &c = cfg;
    switch (c.mode) {
      case VCU_MODE_SLIP_PI:
        run_pi(1.0f, dt);
        torque_cmd = driver * scale;
        break;

      case VCU_MODE_MU_ADAPTIVE: {
        float fz_r = rear_load(accel_est, v_est);
        // learn grip from what the car is actually doing, but only while slip sits
        // near target and the PI isn't saturated: then the tires are working near
        // peak and accel reflects grip, not a transient or a torque limit
        bool near_target = fabsf(slip - c.slip_target) < 0.5f * c.slip_target;
        bool pi_active = scale > 0.05f && scale < c.scale_max - 0.05f;
        if (near_target && pi_active && fz_r > 0) {
          float drag = 0.5f * c.rho * c.cda * v_est * v_est;
          float fx = c.mass * accel_est + drag + c.crr * c.mass * G;
          float mu_obs = fx / fz_r;
          mu_hat += c.mu_adapt_rate * dt * (mu_obs - mu_hat);
          mu_hat = clampf(mu_hat, c.mu_min, c.mu_max);
        }
        // torque that holds the rear tires at mu_hat, plus spinning up the driveline
        float r = c.tire_radius;
        torque_ff = (mu_hat * fz_r * r + c.rear_inertia * accel_est / r) /
                    (c.final_drive * c.driveline_eff);
        run_pi(c.scale_max, dt);
        float cap = torque_ff * scale;
        torque_cmd = driver < cap ? driver : cap;
        break;
      }

      default:
        scale = 1.0f;
        torque_cmd = driver;
        break;
    }
    torque_cmd = clampf(torque_cmd, 0.0f, c.torque_peak);
  }

  void step(const vcu_inputs_t &in, vcu_outputs_t &out) {
    const vcu_config_t &c = cfg;
    float dt = (float)(in.now_us - last_us) * 1e-6f;
    last_us = in.now_us;

    for (int i = 0; i < 4; i++)
      ws[i].update(in.ws_edges[i], in.ws_last_edge_us[i], in.now_us, c);
    if (in.motor_rpm_new) motor_rpm = in.motor_rpm;

    // ground speed: integrate accel, pull toward VectorNav velocity and (when the
    // sensors have enough resolution) the front wheels
    if (in.vn_new) accel_raw = in.vn_accel_x;
    if (c.accel_filter_tau > 0)
      accel_est += dt * (accel_raw - accel_est) / c.accel_filter_tau;
    else
      accel_est = accel_raw;
    v_est += accel_raw * dt;
    if (in.vn_new) v_est += c.vn_vel_gain * (in.vn_vel_x - v_est);
    float v_front = 0.5f * (ws[VCU_FL].omega + ws[VCU_FR].omega) * c.tire_radius;
    if (v_front > c.ws_valid_speed) v_est += c.ws_vel_gain * (v_front - v_est);
    if (v_est < 0) v_est = 0;

    // rear wheel speed from the motor: far finer resolution than the rear sensors
    float v_rear = motor_rpm * (TWO_PI / 60.0f) / c.final_drive * c.tire_radius;
    float denom = v_est > c.slip_speed_floor ? v_est : c.slip_speed_floor;
    slip = (v_rear - v_est) / denom;

    if ((int32_t)(in.now_us - next_control_us) >= 0) {
      float period = (float)c.control_period_us * 1e-6f;
      next_control_us = in.now_us + c.control_period_us;
      control(in.driver_torque_nm, period);
    }

    out.torque_cmd_nm = torque_cmd;
    for (int i = 0; i < 4; i++) out.wheel_omega[i] = ws[i].omega;
    out.v_est = v_est;
    out.accel_est = accel_est;
    out.slip = slip;
    out.scale = scale;
    out.mu_hat = mu_hat;
    out.torque_ff_nm = torque_ff;
  }
};

}  // namespace

extern "C" {

size_t vcu_state_size(void) { return sizeof(State); }

void vcu_init(void *state, const vcu_config_t *config, uint32_t now_us) {
  State *s = new (state) State();
  s->cfg = *config;
  s->last_us = now_us;
  s->next_control_us = now_us;
  s->motor_rpm = 0;
  s->v_est = 0;
  s->accel_raw = 0;
  s->accel_est = 0;
  s->integral = 0;
  s->scale = 1.0f;
  s->mu_hat = config->mu_init;
  s->torque_cmd = 0;
  s->torque_ff = 0;
  s->slip = 0;
}

void vcu_step(void *state, const vcu_inputs_t *in, vcu_outputs_t *out) {
  static_cast<State *>(state)->step(*in, *out);
}

}  // extern "C"
