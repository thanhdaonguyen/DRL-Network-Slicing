import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from environment import NetworkSlicingEnv, UAV, UE, DemandArea
from agents import MADRLAgent
import colorsys
from utils import Configuration

class Network3DVisualizer:
    """3D OpenGL visualizer for UAV network slicing system"""

    def __init__(self, env: NetworkSlicingEnv, agent: Optional[MADRLAgent] = None,
                 window_width: int = 1200, window_height: int = 800):
        pygame.init()
        
        self.env = env
        self.agent = agent
        self.window_width = window_width
        self.window_height = window_height
        
        # Create OpenGL display
        pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("UAV Network Slicing 3D Visualizer")
        
        # OpenGL setup
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (window_width / window_height), 0.1, 5000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        
        # Set clear color (background)
        glClearColor(0.08, 0.08, 0.12, 1.0)
        
        # Camera settings
        self.camera_pos = np.array([env.service_area[0]/2, env.service_area[1]/2, 800])
        self.camera_target = np.array([env.service_area[0]/2, env.service_area[1]/2, 0])
        self.camera_up = np.array([0, 0, 1])
        self.camera_distance = 1500
        self.camera_angle_h = 0  # Horizontal rotation
        self.camera_angle_v = 30  # Vertical rotation
        
        # Mouse control
        self.mouse_sensitivity = 0.3
        self.mouse_pressed = False
        self.last_mouse_pos = (0, 0)
        
        # Colors (RGB values 0-1 for OpenGL)
        self.colors = {
            'background': (0.08, 0.08, 0.12, 1.0),
            'grid': (0.15, 0.15, 0.2, 1.0),
            'ground': (0.1, 0.1, 0.15, 0.8),
            'uav': (1.0, 0.4, 0.4, 1.0),
            'uav_selected': (1.0, 0.8, 0.4, 1.0),
            'ue_embb': (0.4, 0.6, 1.0, 1.0),
            'ue_urllc': (0.4, 1.0, 0.6, 1.0),
            'ue_mmtc': (1.0, 0.7, 0.4, 1.0),
            'coverage': (0.4, 1.0, 0.4, 0.1),
            'connection': (0.6, 0.6, 1.0, 0.3),
            'path': (1.0, 0.4, 0.4, 0.6),
            'interference': (1.0, 0.2, 0.2, 0.4)
        }
        
        # View options
        self.show_coverage = True
        self.show_connections = True
        self.show_paths = True
        self.show_interference = False
        self.show_grid = True
        self.selected_uav = None
        
        # Animation
        self.animation_time = 0
        self.clock = pygame.time.Clock()
        
        # UAV path tracking
        self.uav_paths = {}
        self.max_path_length = 100
        
        # Initialize paths for each UAV
        for uav_id in self.env.uavs.keys():
            self.uav_paths[uav_id] = []
        
        # Performance metrics
        self.metrics_history = {
            'qos': [],
            'energy': [],
            'interference': []
        }
        self.max_history = 100
        self.current_metrics = {'qos': 0.0, 'energy': 0.0, 'interference': 0.0}
        
        # Initialize simulation
        self.observations = self.env.reset()
        
        print("3D Visualizer initialized successfully!")

    def update_camera(self):
        """Update camera position based on angles"""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Convert spherical coordinates to cartesian
        rad_h = math.radians(self.camera_angle_h)
        rad_v = math.radians(self.camera_angle_v)
        
        x = self.camera_distance * math.cos(rad_v) * math.cos(rad_h)
        y = self.camera_distance * math.cos(rad_v) * math.sin(rad_h)
        z = self.camera_distance * math.sin(rad_v)
        
        # Camera position relative to target
        self.camera_pos = self.camera_target + np.array([x, y, z])
        
        # Setup view matrix
        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            self.camera_target[0], self.camera_target[1], self.camera_target[2],
            0, 0, 1  # Up vector
        )

    def draw_grid(self):
        """Draw 3D grid and ground plane"""
        if not self.show_grid:
            return
        
        # Ground plane
        glColor4f(*self.colors['ground'])
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(self.env.service_area[0], 0, 0)
        glVertex3f(self.env.service_area[0], self.env.service_area[1], 0)
        glVertex3f(0, self.env.service_area[1], 0)
        glEnd()
        
        # Grid lines
        glColor4f(*self.colors['grid'])
        glBegin(GL_LINES)
        
        grid_spacing = 100
        
        # Horizontal lines (X direction)
        for y in range(0, int(self.env.service_area[1]) + 1, grid_spacing):
            glVertex3f(0, y, 1)  # Slightly above ground
            glVertex3f(self.env.service_area[0], y, 1)
        
        # Vertical lines (Y direction)
        for x in range(0, int(self.env.service_area[0]) + 1, grid_spacing):
            glVertex3f(x, 0, 1)
            glVertex3f(x, self.env.service_area[1], 1)
        
        glEnd()
        
        # Draw coordinate axes
        glLineWidth(3)
        glBegin(GL_LINES)
        # X axis - Red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(100, 0, 0)
        
        # Y axis - Green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 100, 0)
        
        # Z axis - Blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 100)
        glEnd()
        glLineWidth(1)

    def draw_cube(self, size: float):
        """Draw a simple cube"""
        glBegin(GL_QUADS)
        
        # Front face
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        
        # Back face
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, -size, -size)
        
        # Top face
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        
        # Bottom face
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(-size, -size, size)
        
        # Right face
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(size, size, size)
        glVertex3f(size, -size, size)
        
        # Left face
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, size, -size)
        
        glEnd()

    def draw_uav(self, uav: UAV):
        """Draw a UAV in 3D"""
        glPushMatrix()
        glTranslatef(uav.position[0], uav.position[1], uav.position[2])
        
        # UAV body color
        color = self.colors['uav_selected'] if uav.id == self.selected_uav else self.colors['uav']
        glColor4f(*color)
        
        # UAV body as a cube for simplicity
        self.draw_cube(8)
        
        # Height indicator line
        glColor4f(0.8, 0.8, 0.8, 0.5)
        glBegin(GL_LINES)
        glVertex3f(0, 0, -uav.position[2])  # Line to ground
        glVertex3f(0, 0, 0)
        glEnd()
        
        # Coverage sphere (wireframe)
        if self.show_coverage:
            coverage_radius = 100
            glColor4f(*self.colors['coverage'])
            
            # Draw wireframe sphere
            glBegin(GL_LINE_STRIP)
            for i in range(37):  # 36 segments + 1 to close
                angle = 2 * math.pi * i / 36
                x = coverage_radius * math.cos(angle)
                y = coverage_radius * math.sin(angle)
                glVertex3f(x, y, 0)
            glEnd()
            
            glBegin(GL_LINE_STRIP)
            for i in range(37):
                angle = 2 * math.pi * i / 36
                x = coverage_radius * math.cos(angle)
                z = coverage_radius * math.sin(angle)
                glVertex3f(x, 0, z)
            glEnd()
        
        glPopMatrix()

    def draw_ue(self, ue: UE):
        """Draw a user equipment"""
        glPushMatrix()
        glTranslatef(ue.position[0], ue.position[1], 5)  # Slightly above ground
        
        # Color based on slice type
        color_map = {
            "embb": self.colors['ue_embb'],
            "urllc": self.colors['ue_urllc'], 
            "mmtc": self.colors['ue_mmtc']
        }
        color = color_map.get(ue.slice_type, (1.0, 1.0, 1.0, 1.0))
        glColor4f(*color)
        
        # UE as small cube
        self.draw_cube(3)
        
        glPopMatrix()
        
        # Connection line to assigned UAV
        if self.show_connections and ue.assigned_uav is not None and ue.assigned_uav in self.env.uavs:
            uav = self.env.uavs[ue.assigned_uav]
            
            glColor4f(0.6, 0.6, 1.0, 0.6)
            glBegin(GL_LINES)
            glVertex3f(ue.position[0], ue.position[1], 5)
            glVertex3f(uav.position[0], uav.position[1], uav.position[2])
            glEnd()

    def draw_uav_paths(self):
        """Draw UAV movement paths in 3D"""
        if not self.show_paths:
            return
        
        for uav_id, path in self.uav_paths.items():
            if len(path) < 2:
                continue
            
            color = self.colors['path']
            glColor4f(*color)
            
            # Draw path as connected line segments
            glBegin(GL_LINE_STRIP)
            for pos in path:
                glVertex3f(pos[0], pos[1], pos[2])
            glEnd()

    def update_uav_paths(self):
        """Update UAV path tracking"""
        for uav_id, uav in self.env.uavs.items():
            current_pos = (uav.position[0], uav.position[1], uav.position[2])
            
            if uav_id not in self.uav_paths:
                self.uav_paths[uav_id] = []
            
            # Only add if position has changed significantly
            if (not self.uav_paths[uav_id] or 
                np.linalg.norm(np.array(current_pos) - np.array(self.uav_paths[uav_id][-1])) > 5.0):
                
                self.uav_paths[uav_id].append(current_pos)
                
                # Limit path length
                if len(self.uav_paths[uav_id]) > self.max_path_length:
                    self.uav_paths[uav_id].pop(0)

    def step_simulation(self):
        """Execute one simulation step"""
        if self.agent:
            actions = self.agent.select_actions(self.observations, explore=False)
            
            self.update_uav_paths()
            self.observations, reward, done, info = self.env.step(actions)
            self.update_uav_paths()
            
            print(f"Step completed - Reward: {reward:.3f}")
        else:
            # Random movement for testing
            import random
            for uav in self.env.uavs.values():
                uav.position[0] += random.uniform(-10, 10)
                uav.position[1] += random.uniform(-10, 10)
                uav.position[0] = max(50, min(self.env.service_area[0] - 50, uav.position[0]))
                uav.position[1] = max(50, min(self.env.service_area[1] - 50, uav.position[1]))
            self.update_uav_paths()

    def reset_simulation(self):
        """Reset the simulation"""
        self.observations = self.env.reset()
        for uav_id in self.uav_paths:
            self.uav_paths[uav_id] = []
        print("Simulation reset")

    def handle_mouse_motion(self, rel_pos):
        """Handle mouse movement for camera control"""
        if self.mouse_pressed:
            dx, dy = rel_pos
            self.camera_angle_h += dx * self.mouse_sensitivity
            self.camera_angle_v = max(-89, min(89, self.camera_angle_v + dy * self.mouse_sensitivity))

    def handle_mouse_wheel(self, y):
        """Handle mouse wheel for zoom"""
        self.camera_distance = max(100, min(2000, self.camera_distance - y * 50))

    def run(self):
        """Main visualization loop"""
        running = True
        print("Starting 3D visualization...")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.step_simulation()
                    elif event.key == pygame.K_r:
                        self.reset_simulation()
                    elif event.key == pygame.K_c:
                        self.show_coverage = not self.show_coverage
                        print(f"Coverage: {'ON' if self.show_coverage else 'OFF'}")
                    elif event.key == pygame.K_l:
                        self.show_connections = not self.show_connections
                        print(f"Connections: {'ON' if self.show_connections else 'OFF'}")
                    elif event.key == pygame.K_p:
                        self.show_paths = not self.show_paths
                        print(f"Paths: {'ON' if self.show_paths else 'OFF'}")
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                        print(f"Grid: {'ON' if self.show_grid else 'OFF'}")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.mouse_pressed = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_pressed = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_pressed:
                        self.handle_mouse_motion(event.rel)
                elif event.type == pygame.MOUSEWHEEL:
                    self.handle_mouse_wheel(event.y)
            
            # Clear screen and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Update camera
            self.update_camera()
            
            # Draw scene
            self.draw_grid()
            self.draw_uav_paths()
            
            # Draw entities
            for ue in self.env.ues.values():
                self.draw_ue(ue)
            
            for uav in self.env.uavs.values():
                self.draw_uav(uav)
            
            # Update animation
            self.animation_time += 0.016
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

def main():
    """Main function to run the 3D visualizer"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='UAV Network Slicing 3D Visualizer')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    
    try:
        # config files
        env_config_file_path = "./config/environment/default.yaml"
        model_file_path = "./saved_models/model4/"

        # Create environment
        env = NetworkSlicingEnv(config_path=env_config_file_path)
        print("Environment created successfully")

        # Load agent if checkpoint provided
        agent = None
        if args.checkpoint and os.path.exists(args.checkpoint):
            try:
                model_dir = args.checkpoint
                model_checkpoint = os.path.join(model_dir, "checkpoint_step_100000.pth")

                if os.path.exists(model_checkpoint):
                    env_config = Configuration("./config/environment/default.yaml")
                    num_agents = env_config.system.num_uavs
                    obs_dim = 98 - 31
                    action_dim = 31

                    agent = MADRLAgent(
                        num_agents=num_agents,
                        obs_dim=obs_dim,
                        action_dim=action_dim
                    )
                    agent.load_models(model_checkpoint)
                    print(f"Loaded model from {model_checkpoint}")
                else:
                    print(f"Checkpoint file not found: {model_checkpoint}")
            except Exception as e:
                print(f"Error loading agent: {e}")
                agent = None
        
        # Create and run 3D visualizer
        visualizer = Network3DVisualizer(env, agent, 1200, 800)
        print("3D Visualizer Controls:")
        print("- Mouse drag: Rotate camera")
        print("- Mouse wheel: Zoom in/out")
        print("- SPACE: Step simulation")
        print("- R: Reset simulation")
        print("- C: Toggle coverage areas")
        print("- L: Toggle connections")
        print("- P: Toggle UAV paths")
        print("- G: Toggle grid")
        print("- ESC: Exit")
        
        visualizer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()