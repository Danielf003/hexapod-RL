import numpy as np
import mujoco
from Param_traj import param_traj
from Traj_thetas import thetas_traj
from mecanum_gen import generate_scene
from mecanum_sim import SensorOutput, SimHandler

class InputHolder:
    def __init__(self, timestep, input_func):
        self.timestep = timestep
        self.input_func = input_func
    def update(self, action, current_time):
        if current_time == 0:
            self.Tupd = 2*action[0]
            self.nsteps = int(self.Tupd/self.timestep)
            self.cnt = 0
            self.memory = self.input_func(action)
        if self.cnt >= self.nsteps:
            self.memory = self.input_func(action)
            self.cnt = 0
            self.Tupd = 2*action[0]
            self.nsteps = int(self.Tupd/self.timestep)
        else:
            self.cnt += 1

    def get_output(self):
        # print(f'cnt: {self.cnt}')
        return self.memory

def func(action):
    T_f, T_b, L, alpha = action[:4]
    delta_thetas = np.asarray(action[4:]).reshape((4,))
    print(delta_thetas.shape)
    C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alpha, delta_thetas)

    return T_f, T_b, C_x, C_y, C_z, a

if __name__ == '__main__':
    spec = generate_scene()
    # spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)

    spec.compile()
    model_xml = spec.to_xml()

    simtime = 20

    # prepare data logger
    # simout = SensorOutput(sensor_names=[sen.name for sen in spec.sensors],
    #                       sensor_dims=[3])
    simout = None
    # prepare sim params
    simh = SimHandler(model_xml, None, simlength=simtime, simout=simout)  
    memory = InputHolder(simh.timestep, func)
    
    # define control function
    def ctrl_f(t, model, data, holder: InputHolder):
        legdofs=model.jnt_dofadr[1:]
        legqpos=model.jnt_qposadr[1:]

        use_traj = 1
        use_memory = 1

        nj = 4
        nlegs = 6
        # qdes = np.array([0, 1.22, 4.01, 5.76])
        qdes = np.zeros(nj*1)
        dqdes = np.zeros(nj*1)
        ddqdes = np.zeros(nj*1)

        # Выходы НС
        T_f = 2
        T_b = 0
        L = 2
        alfa = 0.26
        # H = 1
        delta_T = T_f
        delta_thetas = np.array([0, 0.25,-0.2,0])
        if use_memory:
            holder.update([T_f, T_b, L, alfa, delta_thetas], t)
            T_f, T_b, C_x, C_y, C_z, a = holder.get_output()
        else:
            C_x, C_y, C_z, a = param_traj(T_f, T_b, L, alfa, delta_thetas)        

        if use_traj:
            qdes1, dqdes1, ddqdes1= thetas_traj(t, T_f, T_b, 0, C_x, C_y, C_z, a)
            qdes1[2] = -qdes1[2]
            qdes2, dqdes2, ddqdes2= thetas_traj(t, T_f, T_b, delta_T, C_x, C_y, C_z, a)
            qdes2[2] = -qdes2[2]

            q0 = [0, 1.22, 4.01-2*np.pi, 5.76-2*np.pi]
            qdes1 = qdes1 - np.array(q0)
            qdes2 = qdes2 - np.array(q0)

        # kp, kd = np.diag([50,40,30,40]*nlegs), np.diag([2,5,2,2]*nlegs)
        kp, kd = np.diag([5000,4000,3000,13000]*1), np.diag([90,300,200,200]*1)
        u = np.zeros(model.nv)
        e = np.zeros(nj*nlegs)
        de = np.zeros(nj*nlegs)
        print(f'time: {data.time}')
        for i in range(nlegs):
            if use_traj:
                if i in (0, 2, 4):
                        qdes = qdes1
                        # qdes[2] = qdes1[2] - 2*np.pi
                        # qdes[3] = qdes1[3] - 2*np.pi
                        dqdes = dqdes1
                        ddqdes = ddqdes1
                        # print(f'i = {i}, qdes = {qdes}')
                else:
                        qdes = qdes2
                        # qdes[2] = -qdes2[2] - 2*np.pi
                        # qdes[3] = -qdes2[3] - 2*np.pi
                        dqdes = dqdes2
                        ddqdes = ddqdes2
                        # print(f'i = {i}, qdes = {qdes}')

            e[0+i*4:4+i*4] = data.qpos[legqpos][0+i*4:4+i*4]-qdes
            de[0+i*4:4+i*4] = data.qvel[legdofs][0+i*4:4+i*4]-dqdes
            
            u[legdofs[0+i*4:4+i*4]] = ddqdes - kp@e[0+i*4:4+i*4] - kd@de[0+i*4:4+i*4]
        
        # u[legdofs] = np.array([1,1,1,1])
        # print(model.jnt_dofadr)
        Mu = np.empty(model.nv)
        mujoco.mj_mulM(model, data, Mu, u)#+c)
        tau = Mu + data.qfrc_bias
        tau = tau[legdofs]
        # print(tau)
        # print(data.qpos[:4])
        return tau

    # run MuJoCo simulation
    fin_dur = simh.simulate(is_slowed=0, control_func=ctrl_f, control_func_args=(memory,))

    # simout.plot(fin_dur, ['Скорость центра робота [м/с]'], [['v_x','v_y','v_z']])

    # print out xml
    # print(model_xml)
