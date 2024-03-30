class Autopid:
    def __init__(self, input_var, setpoint_var, output_var, output_min, output_max, Kp, Ki, Kd):
        self.input_var = input_var
        self.setpoint_var = setpoint_var
        self.output_var = output_min
        self.output_min = output_min
        self.output_max = output_max
        self.set_gains(Kp, Ki, Kd)
        self.time_step = 5 #inverter control loop is 3ms
        self.last_step = 0
        self.stopped = False
        self.integral = 0
        self.previous_error = 0
        self.bang_on = 0
        self.bang_off = 0

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def set_bang_bang(self, bang_on, bang_off):
        self.bang_on = bang_on
        self.bang_off = bang_off

    def set_output_range(self, output_min, output_max):
        self.output_min = output_min
        self.output_max = output_max

    def set_time_step(self, time_step):
        self.time_step = time_step

    def at_set_point(self, threshold):
        return abs(self.setpoint_var - self.input_var) <= threshold
        
    def clamp(self,n, minn, maxn):
        return min(max(n, minn), maxn)
    
    def run(self, systime, input):
        """run the PID

        Args:
            systime (int): current time (elapsed time) in milliseconds
            input (the input being controlled): yes

        Returns:
            float: the output
        """
        if self.stopped:
            self.stopped = False
            self.reset(systime)
            
        self.input_var = input
        dT = systime - self.last_step
        pid = 0
        dT = systime - self.last_step
        if dT >= self.time_step:
            self.last_step = systime
            error = self.setpoint_var - self.input_var
            self.integral += (error + self.previous_error) / 2 * dT / 1000.0
            if self.Ki > 0:
                self.integral = self.clamp(self.integral,self.output_min/self.Ki,self.output_max/self.Ki)
            d_error = (error - self.previous_error) / dT / 1000.0
            self.previous_error = error
            # print(f"error: {error} integral: {self.integral} d_error: {d_error}")
            pid = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * d_error)
            if pid < self.output_min:
                pid = self.output_min
            elif pid > self.output_max:
                pid = self.output_max
            else:
                pid = pid
            # print(f"input: {input} error: {error} pid: {pid}")
            self.output_var = pid
            return pid


    def stop(self, systime):
        self.stopped = True
        self.reset(systime)

    def reset(self, systime):
        self.last_step = systime
        self.integral = 0
        self.previous_error = 0

    def is_stopped(self):
        return self.stopped

    def get_integral(self):
        return self.integral

    def set_integral(self, integral):
        self.integral = integral
