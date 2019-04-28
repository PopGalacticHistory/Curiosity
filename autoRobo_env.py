import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, gearJointDef, prismaticJointDef, contactListener)
import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle

#Add discription of the project.
#Based on the bipedel_walker script from gym

FPS    = 50 #Frames per Second
SCALE  = 30.0   

MOTORS_TORQUE = SCALE*10

HULL_W = 80/SCALE
HULL_H = 20/SCALE
HULL_POLY =[
    (-15,30), (-30,-10),
    (30,-10), (15,30)
    ]

ARM_W, ARM_H = 7/SCALE, 75/SCALE

VIEWPORT_W = 320
VIEWPORT_H = 240

STEP_FACTOR    = 2
TERRAIN_STEP   = 14/STEP_FACTOR/SCALE
TERRAIN_LENGTH = 200*STEP_FACTOR   # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/6
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20*STEP_FACTOR    # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy

ARM_FD = fixtureDef(
                    shape=polygonShape(box=(ARM_W/2, ARM_H)),
                    density=0.05,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

ARM_FD2 = fixtureDef(
                    shape=polygonShape(box=(ARM_W/2, ARM_H/2)),
                    density=0.0001,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)



HAND_FD = fixtureDef(
                    shape=circleShape(radius=HULL_H),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)

NUM_TREES=30

class ContactDetector(contactListener): #Should be important if trying to catch ball
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        for arm in [self.env.arms[0], self.env.arms[0]]:
            if arm in [contact.fixtureA.body, contact.fixtureB.body]:
                arm.ground_contact = True
    def EndContact(self, contact):
        for arm in [self.env.arms[0], self.env.arms[0]]:
            if arm in [contact.fixtureA.body, contact.fixtureB.body]:
                arm.ground_contact = False

class autoRoboEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }



    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None
        self.hull2 = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
                        shape = polygonShape(vertices=
                        [(0, 0),
                         (1, 0),
                         (1, -1),
                         (0, -1)]),
                        friction = FRICTION)

        self.fd_edge = fixtureDef(
                    shape = edgeShape(vertices=
                    [(0, 0),
                     (1, 1)]),
                    friction = FRICTION,
                    categoryBits=0x0001,
                )

        self.reset()

        high = np.array([np.inf] * 24)
        self.action_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        #The state includes - the arm joint angle, arm joint speed, arm joint motor speed, camera angle, camera speed and rgb array

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.wListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for arm in self.arms:
            self.world.DestroyBody(arm)
        self.arms = []
        self.joints = []

    def _generate_terrain(self):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*(TERRAIN_STEP)
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.8, 1.0 if i%2==0 else 0.8, 0.2)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.5+0.25*np.sin(i/(6*np.pi)), 0.3 + 0.25*np.sin(i/(6*np.pi)), 0.0)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP*STEP_FACTOR
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*STEP_FACTOR*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP*STEP_FACTOR),
                 y+ 5*TERRAIN_STEP*STEP_FACTOR*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP*STEP_FACTOR) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def _generate_trees(self):
        numOfTrees = NUM_TREES
        self.trees_poly = []
        self.leaves_poly = []
        for _ in range(numOfTrees):
          TREE_H = random.randrange(round((TERRAIN_HEIGHT*1.5 + ARM_H)/2),round((TERRAIN_HEIGHT*1.5 + ARM_H)*1.3)) #random.randrange(1,ARM_H)
          TREE_W = ARM_W*TREE_H/8
          pos_x = TERRAIN_STEP*TERRAIN_LENGTH/2 
          xt = random.randrange(round(pos_x-80*TERRAIN_STEP*STEP_FACTOR), round(pos_x+80*TERRAIN_STEP))
          yt = TERRAIN_HEIGHT - TERRAIN_STEP
          poly_t = [(xt,TREE_H), (xt+TREE_W,TREE_H), (xt+TREE_W,yt), (xt,yt)]
          self.trees_poly.append(poly_t)
          xl = xt
          yl = TREE_H
          for i in range(3):
            poly_l = [(xt - 5*TREE_W, yl+i*TREE_H/7) , (xt + 5*TREE_W, yl + i*TREE_H/7), (xt, yl + TREE_H/5 + i*TREE_H/7)]
            self.leaves_poly.append(poly_l)
            
    def _generate_background(self):
        self.numx = 45
        self.numy = 28
        self.background_poly = [0] * self.numy
        self.background_poly_left = [0] * self.numy
        for i in range(self.numy):
            self.background_poly[i] = [0]*self.numx
            self.background_poly_left[i] = [0] * self.numx
        pos_x = TERRAIN_STEP*TERRAIN_LENGTH/2 
        pos_y = TERRAIN_HEIGHT - 7*TERRAIN_STEP
        SQUAR_SIZE = TERRAIN_STEP*STEP_FACTOR
        for i in range(self.numy):
            for j in range(self.numx):     
                x = pos_x + j*SQUAR_SIZE
                y = pos_y + i*SQUAR_SIZE
                poly_p = [
                        (x , y), 
                        (x + SQUAR_SIZE, y), 
                        (x + SQUAR_SIZE, y + SQUAR_SIZE),
                        (x, y + SQUAR_SIZE)
                        ]
                xm = pos_x - j*SQUAR_SIZE
                #ym = pos_y - i*SQUAR_SIZE
                poly_m = [
                        (xm , y), 
                        (xm - SQUAR_SIZE, y), 
                        (xm - SQUAR_SIZE, y + SQUAR_SIZE),
                        (xm, y + SQUAR_SIZE)
                        ]
                
                self.background_poly[i][j] = ( poly_p )
                self.background_poly_left[i][j] = ( poly_m )
                #self.background_poly.append( poly_p )
          

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = True
        self.prev_shaping = None
        self.scroll = 0.0
        magnitude = 0.0
        self.D=10
        self.color_vec=np.random.rand(self.D,self.D)
        self.background = mpimg.imread('background.png')
        #W = VIEWPORT_W/SCALE 
        #H = VIEWPORT_H/SCALE Defined but not used

        self._generate_terrain()
        self._generate_clouds()
        self._generate_trees()
        self._generate_background()

        init_x = TERRAIN_STEP*TERRAIN_LENGTH/2 #setting initial x,y coordinates
        init_y = TERRAIN_HEIGHT
        
        self.hull = self.world.CreateStaticBody(
            position = (init_x, init_y + HULL_H/3),
            fixtures = HULL_FD
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (6,0.3,0.5)

        self.hull2 = self.world.CreateStaticBody(
            position = (init_x + 2*HULL_W, init_y + HULL_H/2),
            fixtures = HULL_FD
                )
        self.hull2.color1 = (0.5,0.4,0.9)
        self.hull2.color2 = (6,0.3,0.5)


        self.arms = []
        self.joints = []
        self.gears = []
        
        gear = self.world.CreateDynamicBody(
          position = (init_x + 2*HULL_W, init_y + ARM_H/2 + 3*HULL_H/5),
          angle = (np.pi),
          fixtures = ARM_FD2
          )
        gear.color1 = (0.8, 3, 0.5)
        gear.color2 = (0.4, 0.3, 0.7)
        
        arm = self.world.CreateDynamicBody(
                position = (init_x, init_y + ARM_H + 3*HULL_H/4),
                angle = (np.pi),
                fixtures = ARM_FD
                )
        arm.color1 = (1, 1, 0.3)
        arm.color2 = (1, 0.8, 0.2)
        
        rjd = revoluteJointDef( #define revolute joint of the arm and motor control
                bodyA=self.hull,
                bodyB=arm,
                anchor= self.hull.worldCenter,
                enableMotor=True,
                enableLimit=False,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=0.5,
                #lowerAngle = 0,
                #upperAngle = np.pi,
                )

        camera_rjd = revoluteJointDef( #define revolute joint of the camera and motor control, to make it easy I simulated the same joint as the arm and converted it to
                                        # actions on the camera. The joints mass is set to zero so gravity won't be a problem.
                bodyA=self.hull2,
                bodyB=gear,
                anchor= self.hull2.worldCenter,
                enableMotor=True,
                enableLimit=False,
                maxMotorTorque=MOTORS_TORQUE/10,
                motorSpeed=0.5,
                lowerAngle = -np.pi/2,
                upperAngle = +np.pi/2,
                )
        self.joints.append(self.world.CreateJoint(rjd))
        self.joints.append(self.world.CreateJoint(camera_rjd))
        self.arms.append(arm)
        self.gears.append(gear)

           
        self.drawlist = self.terrain + self.arms + [self.hull]


        return self.step(np.array([0,0]))[0]

    def step(self, action):
        for r in range(2):
          if np.abs(action[r])>20:
            self.game_over = False
            bad_act=r
            
        assert self.game_over, "Action should be lower than 2"%(action, type(action))
        
        for i in range(len(action)):   #taking action
          if action[i] is not 0 :
            self.joints[i].motorSpeed     = float((action[i])) #Need to think how it relates to SCALE.
            #self.joints[i].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[i]), 0, 1))
            self.joints[i].lowerAngle = self.joints[0].angle
            self.joints[i].upperAngle = self.joints[0].angle
          
          elif action[i] is 0:
            self.joints[i].motorSpeed = 0  #this is to control the arm from falling when action is 0 (counters gravity)



        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.hull.position
        vel = self.hull.linearVelocity
        
        #normalize the angles so to be from -1 to 1
        Hand_joint = self.joints[0].angle/(np.pi/2)
        Head_joint = self.joints[1].angle/(np.pi/2)

        state = [Hand_joint,
                 Head_joint,
                 ] #The state includes - the arm joint angle, arm joint speed, arm joint motor speed, camera angle, camera speed
        state = [np.array(state) , self.render()[1]] #adding the RGB screen pixels as outputs of the states.
        
        #action[1] is equal to the motor command of the camera, it's range will be from -pi/2 to pi/2 where 0 is the center.
        camera_action = ((self.joints[1].angle)/(np.pi/2))*20

        
        self.scroll = pos.x - 12*TERRAIN_STEP*STEP_FACTOR - TERRAIN_STEP*STEP_FACTOR*camera_action

        reward = 0
        #Need to add reward parameters.
        
        done = False
        #Need to add done indicators.
        
        return state, reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        #self.view = rendering.SimpleImageViewer()
        #self.img = mpimg.imread('ganesh.jpg')
        #self.view.imshow(self.img)

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)
        #self.img = rendering.Image('ganesh.jpg', 983, 853)
        #self.img.render1()
        '''
        self.viewer.draw_polygon( [
                    (self.scroll,                  0),
                    (self.scroll+VIEWPORT_W/SCALE, 0),
                    (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
                    (self.scroll,                  VIEWPORT_H/SCALE),
                    ], color = (0.2,0.0,0.7) )
        '''
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
          
        for i in range(self.numy):
            for j in range(self.numx):
                r = ((i**2)+(j**2))**1/2
                self.viewer.draw_polygon( self.background_poly[i][j], color=(0.8 - r*0.006, r*0.001, r*0.004))
                self.viewer.draw_polygon( self.background_poly_left[i][j], color=(0.8 - r*0.006, r*0.001, r*0.004))

        
        #generate trees:        
        for i in range(NUM_TREES):
          tree_color = (0.6-1/10., 0.3-1/10., 0.5-1/10.)
          self.viewer.draw_polygon(self.trees_poly[i], color=tree_color)
        for j in range(len(self.leaves_poly)):
          leaves_color = (0.4, 1, 0.2)
          self.viewer.draw_polygon(self.leaves_poly[j], color = leaves_color)
        
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*STEP_FACTOR*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )
        
        

        
        return self.viewer.render(return_rgb_array = mode=='rgb_array') , self.viewer.get_array()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
