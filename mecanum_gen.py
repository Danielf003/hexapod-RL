import itertools
from os.path import join

import numpy as np
import mujoco


def disable_parts_contact(spec, parts_lists):
    exclusion_set = set()
    p_sets = []
    for parts_list in parts_lists:
        cur = set(parts_list)
        p_sets.append(cur)

    for parts_set in p_sets:
        local_combs = list(itertools.combinations(parts_set, 2))
        for comb in local_combs:
            exclusion_set.add(comb)

    for e in sorted(exclusion_set, key=lambda x: x[0].name + x[1].name):
        b1 = e[0]
        b2 = e[1]
        spec.add_exclude(name=f'exclude_{b1.name}_{b2.name}',
                               bodyname1=b1.name, bodyname2=b2.name)

def generate_scene():
    spec = mujoco.MjSpec()

    # spec.option.timestep = 0.05
    # getattr(spec.visual, 'global').azimuth = 45
    # getattr(spec.visual, 'global').elevation = 0
    spec.visual.scale.jointlength = 3.6
    spec.visual.scale.jointwidth = 0.12

    # does not work in mujoco 3.2.6, fixed in an unreleased version
    # j_color = [47 / 255, 142 / 255, 189 / 255, 1]
    # mjcf_model.visual.rgba.joint = j_color
    # spec.visual.rgba.constraint = j_color

    spec.stat.extent = 0.6
    spec.stat.center = [0,0,.3]

    spec.add_texture(name="//unnamed_texture_0", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1, 1, 1], rgb2=[1, 1, 1], 
                     width=512, height=3072)
    tex = spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4], 
                     rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name='groundplane', texrepeat=[5, 5], reflectance=.2, texuniform=True).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

    spec.worldbody.add_light(name="//unnamed_light_0", directional=True, castshadow=False, pos=[0, 0, 3], dir=[0, 0.8, -1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, condim=1, size=[0, 0, 0.125], material="groundplane")

    # mesh_filenames = ['board1.stl', 'board2.stl', 'box1.stl', 'box2.stl', 'chair1.stl', 
    #              'chair2.stl', 'chair3.stl', 'chair4.stl', 'filter.stl', 'puff.stl', 
    #              'puff2.stl', 'shelf1.stl', 'shelf2.stl', 'sofa.stl', 'table1.stl', 
    #              'table2.stl', 'table_r1.stl', 'table_r2.stl', 'trash1.stl', 
    #              'trash2.stl', 'trash3.stl', 'wall1.stl', 'wall2.stl']
    # mesh_path = "meshes"
    # for fn in mesh_filenames:
    #     mesh_name = fn[:-4] # cutoff the file extension
    #     spec.add_mesh(file=join(mesh_path,fn))
    #     spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name)

    # wheel_filename = 'mecanum.stl'
    # mecanum1_visual = spec.add_mesh(name='mesh1', file=join(mesh_path,wheel_filename), scale=[.78,.78,.78])
    # mecanum2_visual = spec.add_mesh(name='mesh2', file=join(mesh_path,wheel_filename), scale=[.78,.78,-.78])
    
    
    def create_leg(wtype, R, hub_thickness, n_roller=8, roller_angle=np.pi/4):
        lpos = [0,0,0]

        # hub_r = .9*R
        # roller_r = .8*R

        # roller_damping = .0001
        # hub_mass = 0.02
        # hub_damping = .001

        # step = (2*np.pi) / n_roller
        # chord = 2*(R - roller_r)*np.sin(step/2) #distance between neighbor pins
        # psi = roller_angle
        # # for desired roller angle this must hold: chord/h == np.tan(psi)
        # h = chord/np.tan(psi)

        lspec = mujoco.MjSpec()

        link_name='link'
        jaxis = ([0,0,1],[0,1,0],[0,1,0],[0,1,0])
        # cfg = 
        for i in range(len(cfg)):
            link1 = lspec.worldbody.add_body(name=f'{link_name}_1', pos=lpos, 
                                        #    mass=hub_mass, ipos=[0,0,0], 
                                        #    iquat=[0.707107, 0, 0, 0.707107], 
                                        #    inertia=[0.0524193, 0.0303095, 0.0303095]
                                        )
            hub.add_joint(name=hub_name, axis=[0,1,0], damping=hub_damping)
            hub.add_geom(size=[hub_r,hub_thickness/2,0], quat=[0.707107, 0.707107, 0, 0], 
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER, group=1
                        #  contype=1, conaffinity=0,
                        #  rgba=[0.2, 0.2, 0.2, 0.5]
                        )

        # for i in range(n_roller):
        #     roller_name = 'roller_' + str(i)
        #     joint_name = 'slip_' + str(i)

        #     pin_1 = np.array([(R - roller_r)*np.cos(step*i), -h/2, (R - roller_r)*np.sin(step*i)])

        #     if wtype == 0:
        #         if i == n_roller-1:
        #             pin_2 = np.array([(R - roller_r)*np.cos(step*0), h/2, (R - roller_r)*np.sin(step*0)])
        #         else:
        #             pin_2 = np.array([(R - roller_r)*np.cos(step*(i+1)), h/2, (R - roller_r)*np.sin(step*(i+1))])
        #     else:
        #         if i == 0:
        #             pin_2 = np.array([(R - roller_r)*np.cos(step*(n_roller-1)), h/2, (R - roller_r)*np.sin(step*(n_roller-1))])
        #         else:
        #             pin_2 = np.array([(R - roller_r)*np.cos(step*(i-1)), h/2, (R - roller_r)*np.sin(step*(i-1))])
        #     axis = pin_2 - pin_1
        #     pos = pin_1 + axis/2

        #     roller = hub.add_body(name=roller_name, pos=pos,
        #                           ipos=[0,0,0], #iquat=[0.711549, 0.711549, 0, 0], 
        #                           inertia=[.00001, .00001, .00001], mass=.001)
        #     roller.add_joint(name=joint_name, axis=axis, 
        #                      damping=roller_damping, limited=False, actfrclimited=False)
        #     roller.add_geom(size=[roller_r,0,0], quat=[1, 0, 0, 0], group=1
        #                     # contype=1, conaffinity=0, 
        #                     # rgba=[0.2, 0.2, 0.2, 1]
        #                     )
        return link1, lspec

    l, w, h = .4, .2, .05
    wR = 0.04
    hub_thickness = wR
    n_roll = 8

    box = spec.worldbody.add_body(name="box", pos=[0,0,wR+h/2])
    box.add_freejoint()
    box.add_geom(size=[w/2,l/2,h/2], type=mujoco.mjtGeom.mjGEOM_BOX)
    box.add_site(name='box_center')

    dx = w/2 + hub_thickness/2
    dy = .8*l/2
    dz = -h/2

    site1 = box.add_site(pos=[dx,dy,dz], euler=[0,0,-90]) # front right
    site2 = box.add_site(pos=[-dx,dy,dz], euler=[0,0,-90]) # front left
    site3 = box.add_site(pos=[-dx,-dy,dz], euler=[0,0,-90]) # rear left
    site4 = box.add_site(pos=[dx,-dy,dz], euler=[0,0,-90]) # rear right
    
    site0 = spec.worldbody.add_site(pos=[0,0,0.5])
    
    hub_name = 'hub'

    leg_body1, _ = create_leg(0, wR, hub_thickness, n_roll, hub_name=hub_name)

    # wheel_body1, _ = create_wheel(0, wR, hub_thickness, n_roll, hub_name=hub_name)
    # wheel_body2, _ = create_wheel(1, wR, hub_thickness, n_roll, hub_name=hub_name)
    # w1 = site1.attach(wheel_body1, 'w1-', '')
    # w2 = site2.attach(wheel_body2, 'w2-', '')
    # w3 = site3.attach(wheel_body1, 'w3-', '')
    # w4 = site4.attach(wheel_body2, 'w4-', '')

    leg1 = site0.attach(leg_body1, 'leg1-', '')

    # spec.compiler.inertiagrouprange = [0,1]

    # w1.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2)
    # w2.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2)
    # w3.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum1_visual.name, euler=[90,0,0], group=2)
    # w4.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mecanum2_visual.name, euler=[90,0,0], group=2)

    # disable_parts_contact(spec, [(box, w1, *w1.bodies),
    #                              (box, w2, *w2.bodies),
    #                              (box, w3, *w3.bodies),
    #                              (box, w4, *w4.bodies),
    #                              (w1, spec.worldbody),
    #                              (w2, spec.worldbody),
    #                              (w3, spec.worldbody),
    #                              (w4, spec.worldbody)])
    # Collision is enabled only for pairs world-rollers(spheres) and world-chassie(box). 
    # Hub and visual wheel are disabled

    input_saturation = [-.4,.4] # Nm
    # for i in range(4):
    #     spec.add_actuator(name=f'torque{i+1}', target=f'w{i+1}-'+hub_name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
    #                       ctrllimited=True, ctrlrange=input_saturation)

    return spec

