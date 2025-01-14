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
    getattr(spec.visual, 'global').azimuth = -135
    getattr(spec.visual, 'global').elevation = -20
    spec.visual.scale.jointlength = 3.6
    spec.visual.scale.jointwidth = 0.12

    # does not work in mujoco 3.2.6, fixed in an unreleased version
    # j_color = [47 / 255, 142 / 255, 189 / 255, 1]
    # mjcf_model.visual.rgba.joint = j_color
    # spec.visual.rgba.constraint = j_color

    spec.stat.extent = 4.5
    spec.stat.center = [0,0,.3]

    spec.add_texture(name="//unnamed_texture_0", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[1, 1, 1], rgb2=[1, 1, 1], 
                     width=512, height=3072)
    tex = spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D, 
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4], 
                     rgb2=[0.1, 0.2, 0.3], mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name='groundplane', texrepeat=[5, 5], reflectance=.2, texuniform=True).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'groundplane'

    spec.worldbody.add_light(name="//unnamed_light_0", directional=True, castshadow=False, pos=[0, 0, 3], dir=[0, 0.8, -1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.125], material="groundplane",
                            # friction = [0.01, 0.005, 0.0003]
                            # condim=6
                            )

    xr = np.arange(-20,20,.5) #1
    yr = np.arange(-20,20,.5) #1
    # hr = np.random.randint(0,10,5)
    np.random.seed(0)
    hr = np.random.rand(len(xr),len(yr))*.5
    eulr = np.random.randint(0,2,(len(xr),len(yr),3))
    zi = 0
    li, wi = .2, .2
    for i,xi in enumerate(xr):
        for j,yi in enumerate(yr):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[li, wi, li], pos=[xi,yi,zi], euler=np.dot(eulr[i,j,:],45))
            # spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[li, wi, hr[i,j]], pos=[xi,yi,zi])

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
    
    
    def create_leg():
        lpos = [0,0,0]

        lspec = mujoco.MjSpec()

        link_name='l'
        link_r = 0.08
        jaxis = ([0,0,1],[0,-1,0],[0,-1,0],[0,-1,0])
        dampings = (.001,)*4

        initial_q = [0, 1.22, 4.01-2*np.pi, 5.76-2*np.pi] # for 1 leg
        initial_q_glob = [0, initial_q[1], initial_q[1]+initial_q[2], initial_q[1]+initial_q[2]+initial_q[3]]
        RyT = lambda ang: np.array([[np.cos(ang),0,-np.sin(ang)],[0,0,0],[np.sin(ang),0,np.cos(ang)]])
        # ends = ([.3,0,0],[.6,0,.6],[.7,0,-1.],[0,0,-.6])
        l1,l2,l3,l4 = .3, np.linalg.norm(np.array([.6,0,.6])), np.linalg.norm(np.array([.7,0,-1.])), np.linalg.norm(np.array([0,0,-.6]))
        ends = ([l1,0,0],[l2,0,0],[l3,0,0],[l4,0,0]) #local for each link
        ends = tuple(RyT(ang) @ np.asarray(en) for en, ang in zip(ends,initial_q_glob)) #local for each link
        starts = (lpos, *ends[:-1])

        parent = lspec.worldbody
        link_bodies = []
        for i in range(len(jaxis)):
            link = parent.add_body(name=f'{link_name}{i+1}', pos=starts[i], 
                                        #    mass=hub_mass, ipos=[0,0,0], 
                                        #    iquat=[0.707107, 0, 0, 0.707107], 
                                        #    inertia=[0.0524193, 0.0303095, 0.0303095]
                                        )
            link.add_joint(name=f'{link_name}{i+1}', axis=jaxis[i], damping=dampings[i])
            link.add_geom(size=[link_r,0,0], fromto=[0,0,0,*ends[i]],
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE, group=1,
                        # friction = [2, 0.005, 0.0003], condim=6
                        #  contype=1, conaffinity=0,
                        #  rgba=[0.2, 0.2, 0.2, 0.5]
                        )
            parent = link
            link_bodies.append(link)

        return link_bodies[0], lspec

    l, w, h = 4., 2., .5
    n_legs = 6
    # box_pos = [0,0,1+h/2]
    box_pos = [0,0,1.1+h/2] # для неровной поверхности нужно спавнить выше
    box = spec.worldbody.add_body(name="box", pos=box_pos)
    box.add_freejoint()
    box.add_geom(size=[w/2,l/2,h/2], type=mujoco.mjtGeom.mjGEOM_BOX)
    box.add_site(name='box_center')

    spec.add_sensor(name='rob_Y', type=mujoco.mjtSensor.mjSENS_FRAMEYAXIS, objname='box_center', objtype=mujoco.mjtObj.mjOBJ_SITE)

    dx = w/2 
    dy = .8*l/2
    dz = -h/2

    leg_rot = 0
    site1 = box.add_site(pos=[dx,dy,dz], euler=[0,0,0+leg_rot]) # front right
    site2 = box.add_site(pos=[-dx,dy,dz], euler=[0,0,180-leg_rot]) # front left
    site3 = box.add_site(pos=[-dx,0,dz], euler=[0,0,180]) # center left
    site4 = box.add_site(pos=[-dx,-dy,dz], euler=[0,0,180+leg_rot]) # rear left
    site5 = box.add_site(pos=[dx,-dy,dz], euler=[0,0,0-leg_rot]) # rear right
    site6 = box.add_site(pos=[dx,0,dz], euler=[0,0,0]) # center right
    sites = [site1,site2,site3,site4,site5,site6]

    leg_body1, _ = create_leg()
    
    for i in range(n_legs):
        leg1 = sites[i].attach(leg_body1, f'leg{i+1}-', '')

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
    for i in range(n_legs):
        for j in range(4):
            spec.add_actuator(name=f'leg{i+1}-l{j+1}', target=f'leg{i+1}-l{j+1}', trntype=mujoco.mjtTrn.mjTRN_JOINT,
                            #   ctrllimited=True, ctrlrange=input_saturation
                              )
    
    mjmodel = spec.compile()
    # initial_q = [0, 1.22, 4.01, 5.76] # for 1 leg
    initial_q = [0,0,0,0] # for 1 leg
    qpos0 = np.zeros(mjmodel.nq)
    # legqpos = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    legqpos=mjmodel.jnt_qposadr[1:]
    qpos0[legqpos] = np.asarray(initial_q * n_legs)
    qpos0[:3] = np.asarray(box_pos)

    spec.add_key(name='q0', qpos=qpos0)

    return spec

if __name__ == '__main__':
    spec = generate_scene()

    spec.compile()
    model_xml = spec.to_xml()

    with open("hexapod.xml", "w") as text_file:
            text_file.write(model_xml)
