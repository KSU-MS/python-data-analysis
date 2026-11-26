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
  uint32_t last_vn_us;

  float motor_rpm;
  float torque_fb;
  bool have_torque_fb;
  float v_est;
  float accel_raw;
  float accel_est;

  float integral;
  float scale;
  float torque_cmd;
  float torque_ff;
  float slip;
  vcu_learned_t learned;

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

  void learn_mu(float fz_r, float dt, bool cap_binding) {
    const vcu_config_t &c = cfg;
    // only trust what we see while the inverter delivers what we ask for: when it's
    // motor/power limited, low slip says nothing about grip
    bool tracking = have_torque_fb && fabsf(torque_fb - torque_cmd) < c.torque_track_tol;
    if (!tracking || fz_r <= 0) return;

    // fast: grip from measured accel, while slip sits near target and the PI isn't
    // saturated (tires working near peak, not a transient)
    bool near_target = fabsf(slip - c.slip_target) < 0.5f * c.slip_target;
    bool pi_active = scale > 0.05f && scale < c.scale_max - 0.05f;
    if (near_target && pi_active) {
      float drag = 0.5f * c.rho * c.cda * v_est * v_est;
      float fx = c.mass * accel_est + drag + c.crr * c.mass * G;
      learned.mu += c.mu_adapt_rate * dt * (fx / fz_r - learned.mu);
    }
    // slow: absorb the PI's steady correction into the map, so a map that's too
    // low (PI pinned high, slip under target) or too high can't stay that way
    if (cap_binding) learned.mu += c.mu_transfer_rate * dt * (scale - 1.0f) * learned.mu;
    learned.mu = clampf(learned.mu, c.mu_min, c.mu_max);
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
        float r = learned.r_rear;
        learn_mu(fz_r, dt, driver >= torque_ff * scale);
        // torque that holds the rear tires at mu, plus spinning up the driveline
        torque_ff = (learned.mu * fz_r * r + c.rear_inertia * accel_est / r) /
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

  void calibrate(const vcu_inputs_t &in, float dt, float v_front, float omega_rear) {
    const vcu_config_t &c = cfg;
    bool stopped = ws[0].omega == 0 && ws[1].omega == 0 && ws[2].omega == 0 &&
                   ws[3].omega == 0 && fabsf(motor_rpm) < 1.0f && in.driver_torque_nm < 1.0f;
    if (stopped) {
      // staged on the line: VectorNav accel should read 0, and so should speed
      learned.accel_bias += c.accel_bias_rate * dt * (accel_raw - learned.accel_bias);
      v_est = 0;
      return;
    }
    bool steady = fabsf(accel_est) < 2.0f && v_est > c.radius_cal_speed;
    if (!steady) return;
    float lo = c.tire_radius * (1 - c.radius_tolerance);
    float hi = c.tire_radius * (1 + c.radius_tolerance);
    // front wheels roll free: VectorNav speed / wheel speed is their radius
    if (in.vn_new && v_front > 0) {
      float dt_vn = (float)(in.now_us - last_vn_us) * 1e-6f;
      float omega_front = v_front / learned.r_front;
      float sample = in.vn_vel_x / omega_front;
      learned.r_front += clampf(c.radius_cal_rate * dt_vn, 0.0f, 1.0f) * (sample - learned.r_front);
      learned.r_front = clampf(learned.r_front, lo, hi);
    }
    // rear only rolls free while coasting
    if (have_torque_fb && fabsf(torque_fb) < c.coast_torque && omega_rear > 0) {
      float sample = v_est / omega_rear;
      learned.r_rear += c.radius_cal_rate * dt * (sample - learned.r_rear);
      learned.r_rear = clampf(learned.r_rear, lo, hi);
    }
  }

  void step(const vcu_inputs_t &in, vcu_outputs_t &out) {
    const vcu_config_t &c = cfg;
    float dt = (float)(in.now_us - last_us) * 1e-6f;
    last_us = in.now_us;

    for (int i = 0; i < 4; i++)
      ws[i].update(in.ws_edges[i], in.ws_last_edge_us[i], in.now_us, c);
    if (in.motor_rpm_new) motor_rpm = in.motor_rpm;
    if (in.motor_torque_new) {
      torque_fb = in.motor_torque_fb;
      have_torque_fb = true;
    }

    // ground speed: integrate (bias-corrected) accel, pull toward VectorNav
    // velocity and, when the sensors have enough resolution, the front wheels
    if (in.vn_new) accel_raw = in.vn_accel_x;
    float accel = accel_raw - learned.accel_bias;
    if (c.accel_filter_tau > 0)
      accel_est += dt * (accel - accel_est) / c.accel_filter_tau;
    else
      accel_est = accel;
    v_est += accel * dt;
    if (in.vn_new) v_est += c.vn_vel_gain * (in.vn_vel_x - v_est);
    float v_front = 0.5f * (ws[VCU_FL].omega + ws[VCU_FR].omega) * learned.r_front;
    if (v_front > c.ws_valid_speed) v_est += c.ws_vel_gain * (v_front - v_est);
    if (v_est < 0) v_est = 0;

    // rear wheel speed from the motor: far finer resolution than the rear sensors
    float omega_rear = motor_rpm * (TWO_PI / 60.0f) / c.final_drive;
    calibrate(in, dt, v_front, omega_rear);
    if (in.vn_new) last_vn_us = in.now_us;
    float v_rear = omega_rear * learned.r_rear;
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
    out.torque_ff_nm = torque_ff;
    out.learned = learned;
  }
};

}  // namespace

extern "C" {

size_t vcu_state_size(void) { return sizeof(State); }

size_t vcu_struct_size(int which) {
  switch (which) {
    case 0: return sizeof(vcu_config_t);
    case 1: return sizeof(vcu_learned_t);
    case 2: return sizeof(vcu_inputs_t);
    case 3: return sizeof(vcu_outputs_t);
    case 4: return sizeof(vcu_log_record_t);
    default: return 0;
  }
}

void vcu_init(void *state, const vcu_config_t *config, const vcu_learned_t *learned,
              uint32_t now_us) {
  State *s = new (state) State();
  s->cfg = *config;
  s->last_us = now_us;
  s->next_control_us = now_us;
  s->last_vn_us = now_us;
  s->motor_rpm = 0;
  s->torque_fb = 0;
  s->have_torque_fb = false;
  s->v_est = 0;
  s->accel_raw = 0;
  s->accel_est = 0;
  s->integral = 0;
  s->scale = 1.0f;
  s->torque_cmd = 0;
  s->torque_ff = 0;
  s->slip = 0;
  vcu_learned_t l = {config->tire_radius, config->tire_radius, 0.0f, config->mu_init};
  if (learned) {
    if (learned->r_front > 0) l.r_front = learned->r_front;
    if (learned->r_rear > 0) l.r_rear = learned->r_rear;
    l.accel_bias = learned->accel_bias;
    if (learned->mu > 0) l.mu = learned->mu;
  }
  s->learned = l;
}

void vcu_step(void *state, const vcu_inputs_t *in, vcu_outputs_t *out) {
  static_cast<State *>(state)->step(*in, *out);
}

void vcu_make_log_record(const void *state, const vcu_inputs_t *in, const vcu_outputs_t *out,
                         vcu_log_record_t *r) {
  const State *s = static_cast<const State *>(state);
  r->version = VCU_LOG_VERSION;
  r->mode = (uint16_t)s->cfg.mode;
  r->now_us = in->now_us;
  for (int i = 0; i < 4; i++) {
    r->ws_edges[i] = in->ws_edges[i];
    r->ws_last_edge_us[i] = in->ws_last_edge_us[i];
  }
  r->motor_rpm_new = in->motor_rpm_new;
  r->motor_torque_new = in->motor_torque_new;
  r->vn_new = in->vn_new;
  r->reserved = 0;
  r->motor_rpm = in->motor_rpm;
  r->motor_torque_fb = in->motor_torque_fb;
  r->vn_vel_x = in->vn_vel_x;
  r->vn_accel_x = in->vn_accel_x;
  r->driver_torque_nm = in->driver_torque_nm;
  r->torque_cmd_nm = out->torque_cmd_nm;
  r->v_est = out->v_est;
  r->slip = out->slip;
  r->scale = out->scale;
  r->mu = out->learned.mu;
}

}  // extern "C"
