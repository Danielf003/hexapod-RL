import numpy as np
import mujoco

from mecanum_gen import generate_scene
from mecanum_sim import SensorOutput, SimHandler


if __name__ == '__main__':
    spec = generate_scene()
    # spec.add_sensor(name='vel_c', type=mujoco.mjtSensor.mjSENS_VELOCIMETER, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)

    spec.compile()
    model_xml = spec.to_xml()

    simtime = 100

    # prepare data logger
    # simout = SensorOutput(sensor_names=[sen.name for sen in spec.sensors],
    #                       sensor_dims=[3])
    simout = None
    # prepare sim params
    simh = SimHandler(model_xml, None, simlength=simtime, simout=simout)  

    # define control function
    def ctrl_f(t, model, data):
        # print(model.jnt_dofadr)
        legdofs=model.jnt_dofadr[1:]
        legqpos=model.jnt_qposadr[1:]
        # print(legqpos)

        nj = 4
        nlegs = 6

        qdes = np.zeros(nj*nlegs)
        dqdes = np.zeros(nj*nlegs)
        ddqdes = np.zeros(nj*nlegs)
        e = data.qpos[legqpos]-qdes
        de = data.qvel[legdofs]-dqdes

        # kp, kd = np.diag([50,40,30,40]*nlegs), np.diag([2,5,2,2]*nlegs)
        kp, kd = np.diag([5000,4000,3000,13000]*nlegs), np.diag([90,300,200,200]*nlegs)
        u = np.zeros(model.nv)
        u[legdofs] = ddqdes - kp@e - kd@de
        
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
    fin_dur = simh.simulate(is_slowed=True, control_func=ctrl_f)

    # simout.plot(fin_dur, ['Скорость центра робота [м/с]'], [['v_x','v_y','v_z']])

    # print out xml
    # print(model_xml)
