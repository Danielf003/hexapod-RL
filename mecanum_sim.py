import time
import os
from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
import mujoco
from mujoco import viewer


class SimHandler:
    """A class to prepare and run MuJoCo simulation from existing XML model."""
    def __init__(self, xml, xml_path, simlength, simout = None, timestep=None, actuator_callback=None, recorder=None):
        """
        Prepare MuJoCo simulation from an existing XML model.

        Args:
            xml (str): XML of a MuJoCo model (taken from xml_path if None).
            xml_path (str): path to XML-file of a MuJoCo model.
            simlength (float): end time of the simulation.
            simout (SimOutput): object for simulation data collection.
            timestep (float): simulation timestep (taken from a model if None).
            actuator_callback: custom actuator dynamics.
        """
        if xml:
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:

            self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        if timestep:
            self.model.opt.timestep = timestep
            self.timestep = timestep
        else:
            self.timestep = self.model.opt.timestep

        self.actuator_callback = actuator_callback

        self.n_steps = int(simlength / self.model.opt.timestep + 0.5)
        self.simout = simout
        if self.simout:
            self.simout.prepare(self.model, self.data, self.n_steps)

        self.recorder = recorder
        if self.recorder:
            self.recorder.prepare_folder()
            self.recorder.finish_init(self.timestep, max_source_duration=simlength)
            # getattr(self.model.vis, 'global_').offwidth = recorder.w
            # getattr(self.model.vis, 'global_').offheight = recorder.h

    def simulate(self, const_control=None, control_func=None, is_slowed=True, control_func_args=(), qpos0=None):
        """
        Run prepared MuJoCo simulation.

        Args:
            const_control: list of constant input values (or a single value) for all actuators in a model.
            is_slowed (bool): if True, slows down the simulation to realtime (in case computations are faster).

        Returns:
            float: the time when the simulation has stopped.
        """
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        if qpos0 is not None:
            self.model.qpos0 = qpos0
        self.conrol_func = control_func
        self.conrol_func_args = control_func_args
        if const_control:
            self.data.ctrl = const_control

        if self.actuator_callback:
            mujoco.set_mjcb_act_dyn(self.actuator_callback)
        self._run_main_loop(is_slowed)
        mujoco.set_mjcb_act_dyn(None)

        return self.data.time

    def _run_main_loop(self, is_slowed):
        with viewer.launch_passive(self.model, self.data, show_left_ui=False) as viewr:
            viewr.cam.elevation = self.model.vis.global_.elevation
            viewr.cam.azimuth = self.model.vis.global_.azimuth
            viewr.cam.distance = self.model.stat.extent * 1.5
            viewr.cam.lookat = self.model.stat.center
            viewr.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewr.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True

            current_step = 0
            start = time.time()
            while viewr.is_running() and current_step < self.n_steps:

                # Write current values to output
                if self.simout:
                    self.simout.update(current_step)

                # Render and store certain frames
                if self.recorder:
                    if self.recorder.is_frame_to_save(self.data.time, self.timestep):
                        self.recorder.store_frame(None, is_file_saved=False)
                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewr.sync()

                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                if self.conrol_func is not None:
                    self.data.ctrl = self.conrol_func(current_step * self.model.opt.timestep, self.model, self.data, *self.conrol_func_args)
                mujoco.mj_step(self.model, self.data)

                # Progressive time-keeping
                if is_slowed:
                    desired_time = current_step * self.model.opt.timestep
                    wait_time = desired_time - (time.time()-start)
                    if wait_time > 0:
                        time.sleep(wait_time)

                current_step += 1
        print(f'Simulation lasted {time.time()-start} s.')

class SimOutput:
    """
    An abstract class for data collection from the simulation.

    Includes methods to prepare, update and save data. Implementation
    depends on a specific MuJoCo model and the variables of interest."""
    @abstractmethod
    def prepare(self, model, data, n_steps):
        """
        Initialize variables to read desired data and store it.

        Args:
            model (mujoco.MjModel): MuJoCo model instance to get special info (like actuator indexes).
            data (mujoco.MjData): MuJoCo data instance to read the data from.
            n_steps (int): max number of steps in a simulation (needed to init arrays for data storage).
        """
        pass

    @abstractmethod
    def update(self, step):
        """
        Store data from current iteration.

        Args:
            step (int): current iteration/step number.
        """
        pass

    @abstractmethod
    def trim_data(self):
        """Shrink arrays with collected data if the simulation ended prematurely."""
        pass

    @abstractmethod
    def save(self):
        """Save collected data to a file."""
        pass

    @abstractmethod
    def plot(self, simlength):
        """
        Plot collected data using matplotlib.

        Args:
            simlength (float): time axis limit for plots.
        """
        pass

    @staticmethod
    def save_csv(data, filename, save_dir=''):
        """
        Save data to a .csv file.

        Args:
            data: numpy array or list to save.
            filename (str): name of the CSV file.
            save_dir (str): path to save the CSV file (taken from definitions if None).
        """
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, filename+".csv"), data, delimiter=",", fmt='%s')


class SensorOutput(SimOutput):
    """A class for collecting output of sensors defined in the model."""
    def __init__(self, sensor_names, sensor_dims):
        """
        Args:
            sensor_names (iterable): list of sensor names
            sensor_dims (iterable): list of dimensions of sensor output signals
        """
        self.sensor_names = sensor_names
        self.sensor_dims = sensor_dims

    def prepare(self, model, data, n_steps):
        self.model = model
        self.data = data
        self.times = np.full(n_steps, None)
        self.sensordata = {name: np.full([n_steps, dim], None) for name, dim in zip(self.sensor_names, self.sensor_dims)}

    def update(self, step):
        for name in self.sensor_names:
            self.sensordata[name][step] = self.data.sensor(name).data.copy()
        self.times[step] = self.data.time

    def trim_data(self):
        idxs = np.where(self.times != None)[0]
        self.times = self.times[idxs]
        for name in self.sensor_names:
            self.sensordata[name] = self.sensordata[name][idxs]

    def save(self):
        self.trim_data()
        res = np.h_stack((self.times, np.h_stack((self.sensordata[name] for name in self.sensor_names))))
        self.save_csv(res, "sensors_out")

    def plot(self, simlength, ylabels, leglabels):
        self.trim_data()

        for name, ylab, leglab in zip(self.sensor_names, ylabels, leglabels):
            plt.figure(1)
            ax = plt.gca()
            ax.set_xlim([0, simlength])
            ax.grid(True)
            ax.plot(self.times, self.sensordata[name], label=leglab)
            ax.set_title(f'Показания датчика {name}')
            ax.set_ylabel(ylab)
            ax.set_xlabel('Время [с]')
            ax.legend(frameon=True)#, loc='lower right')
            plt.tight_layout()

        plt.show()


# class MotorsOutput(SimOutput):
#     """A class for collecting states, velocities and efforts of actuators defined in the model."""
#     def prepare(self, model, data, n_steps):
#         self.model = model
#         self.data = data
#         self.times = np.full(n_steps, None)
#         self.mot_num = len(self.data.ctrl)
#         self.state_num = len(self.data.act)

#         self.speed = np.full([n_steps, self.mot_num], None)
#         self.torque = np.full([n_steps, self.mot_num], None)
#         self.current = np.full([n_steps, self.mot_num], None)

#     def update(self, step):
#         for i in range(self.mot_num):
#             # Actuated joint velocity (output motor speed)
#             self.speed[step][i] = self.data.qvel[self.model.actuator_trnid[i][0]]
#             # Actuated joint effort from the actuator (output motor torque)
#             self.torque[step][i] = self.data.qfrc_actuator[self.model.actuator_trnid[i][0]]
#             # Current in motor windings, 0 for non-BLDCs (actuators without state)
#             if self.state_num > 0:
#                 self.current[step][i] = self.data.act[i]
#             else:
#                 self.current[step][i] = 0
#         self.times[step] = self.data.time

#     def trim_data(self):
#         idxs = np.where(self.times != None)[0]
#         self.times = self.times[idxs]
#         self.current = self.current[idxs]
#         self.torque = self.torque[idxs]
#         self.speed = self.speed[idxs]

#     def save(self):
#         self.trim_data()
#         res = np.column_stack((self.times, self.current,
#                                self.torque, self.speed*30/np.pi))
#         self.save_csv(res, "motors_out")

#     def plot(self, simlength):
#         self.trim_data()

#         fig, axs = plt.subplots(3, sharex=True)
#         axs[0].set_xlim([0, simlength])
#         # fig.suptitle(r'$Motor\;dynamics$')
#         fig.suptitle('BLDC dynamics')
#         axs[0].plot(self.times, self.current)
#         # axs[0].set_title('$Current\;[A]$')
#         axs[0].set_ylabel('Current [A]')
#         axs[0].grid(True)
#         # plt.figlegend([r'$Right\;motor$', r'$Left\;motor$'])
#         plt.figlegend(['Right motor', 'Left motor'])
#         axs[1].plot(self.times, self.torque)
#         # axs[1].set_title(r'$Output\;torque\;[Nm]$')
#         axs[1].set_ylabel('Output torque [Nm]')
#         axs[1].grid(True)
#         axs[2].plot(self.times, self.speed*30/np.pi)
#         # axs[2].set_title('$Speed\;[rpm]$')
#         axs[2].set_ylabel('Speed [rpm]')
#         axs[2].set_xlabel('Time [s]')
#         axs[2].grid(True)
#         plt.tight_layout()
#         plt.show()

#     def plot_ideal(self, simlength):
#         self.trim_data()

#         fig, axs = plt.subplots(2, sharex=True)
#         axs[0].set_xlim([0, simlength])
#         # fig.suptitle(r'$Motor\;dynamics$')
#         # fig.suptitle('Motor dynamics')
#         fig.suptitle('Динамика приводов')
#         axs[0].plot(self.times, self.torque)
#         # axs[0].set_title(r'$Output\;torque\;[Nm]$')
#         # axs[0].set_ylabel('Output torque [Nm]')
#         axs[0].set_ylabel('Выходной момент [Н*м]')

#         axs[0].grid(True)
#         # plt.figlegend([r'$Upper\;motor$', r'$Lower\;motor$'])
#         # plt.figlegend(['Upper motor', 'Lower motor'])
#         # plt.figlegend(['Верхний мотор', 'Нижний мотор'])
#         # plt.figlegend(['Right motor', 'Left motor'])
#         plt.figlegend(['Правый мотор', 'Левый мотор'])
#         axs[1].plot(self.times, self.speed * 30 / np.pi)
#         # axs[1].set_title('$Speed\;[rpm]$')
#         # axs[1].set_ylabel('Speed [rpm]')
#         # axs[1].set_xlabel('Time [s]')
#         axs[1].set_ylabel('Частота вращения [об/мин]')
#         axs[1].set_xlabel('Время [с]')
#         axs[1].grid(True)
#         plt.tight_layout()
#         plt.show()
