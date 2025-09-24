# visualizer.py
import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from environment import NetworkSlicingEnv, UAV, UE, DemandArea
from agents import MADRLAgent
import colorsys
from utils import Configuration

class NetworkVisualizer:
    """Interactive pygame visualizer for UAV network slicing system with UAV path tracking"""

    def __init__(self, env: NetworkSlicingEnv, agent: Optional[MADRLAgent] = None,
                 window_width: int = 1200, window_height: int = 800):
        pygame.init()
        
        self.env = env
        self.agent = agent
        self.window_width = window_width
        self.window_height = window_height
        
        # Create main display
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("UAV Network Slicing Visualizer")
        
        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Colors
        self.colors = {
            'background': (20, 20, 30),
            'grid': (40, 40, 50),
            'text': (200, 200, 200),
            'uav': (255, 100, 100),
            'uav_selected': (255, 200, 100),
            'ue_embb': (100, 150, 255),      # Blue for eMBB
            'ue_urllc': (100, 255, 150),     # Green for URLLC
            'ue_mmtc': (255, 180, 100),      # Orange for mMTC
            'coverage': (100, 255, 100, 30),  # Semi-transparent green
            'connection': (150, 150, 255, 100),
            'da_boundary': (200, 200, 100, 50),
            'panel_bg': (30, 30, 40),
            'button': (60, 60, 80),
            'button_hover': (80, 80, 100),
            'button_active': (100, 100, 120),
            'good': (100, 255, 100),
            'warning': (255, 255, 100),
            'bad': (255, 100, 100)
        }
        
        # Layout
        self.main_view_width = window_width - 300  # Reserve 300px for side panel
        self.main_view_height = window_height - 100  # Reserve 100px for top panel
        self.side_panel_x = self.main_view_width
        self.top_panel_height = 80
        
        # Scaling factors for converting environment coordinates to screen
        self.scale_x = (self.main_view_width - 40) / env.service_area[0]
        self.scale_y = (self.main_view_height - 40) / env.service_area[1]
        self.offset_x = 20
        self.offset_y = self.top_panel_height + 20
        
        # View options
        self.show_coverage = True
        self.show_connections = True
        self.show_das = True
        self.show_interference = False
        self.selected_uav = None
        self.selected_slice = None  # None means show all
        
        # Animation
        self.animation_time = 0
        self.clock = pygame.time.Clock()
        
        # Performance metrics
        self.metrics_history = {
            'qos': [],
            'energy': [],
            'interference': []
        }
        self.max_history = 100

        # UAV path tracking
        self.uav_paths = {}  # Dictionary to store path for each UAV
        self.max_path_length = 100  # Maximum number of positions to store
        self.show_paths = True  # Toggle for path display
        
        # Initialize paths for each UAV
        for uav_id in self.env.uavs.keys():
            self.uav_paths[uav_id] = []
        
        # Buttons
        self.buttons = {
            'coverage': Button(self.side_panel_x + 10, 150, 120, 30, "Coverage", self.toggle_coverage),
            'connections': Button(self.side_panel_x + 140, 150, 120, 30, "Connections", self.toggle_connections),
            'das': Button(self.side_panel_x + 10, 190, 120, 30, "DAs", self.toggle_das),
            'interference': Button(self.side_panel_x + 140, 190, 120, 30, "Interference", self.toggle_interference),
            'step': Button(self.side_panel_x + 10, 650, 120, 40, "Step", self.step_simulation),
            'reset': Button(self.side_panel_x + 140, 650, 120, 40, "Reset", self.reset_simulation),
            'slice_all': Button(self.side_panel_x + 10, 250, 60, 25, "All", lambda: self.set_slice_filter(None)),
            'slice_embb': Button(self.side_panel_x + 80, 250, 60, 25, "eMBB", lambda: self.set_slice_filter("embb")),
            'slice_urllc': Button(self.side_panel_x + 150, 250, 60, 25, "URLLC", lambda: self.set_slice_filter("urllc")),
            'slice_mmtc': Button(self.side_panel_x + 220, 250, 60, 25, "mMTC", lambda: self.set_slice_filter("mmtc")),
            'paths': Button(self.side_panel_x + 10, 710, 120, 30, "UAV Paths", self.toggle_paths)
        }
        
        # Initialize
        self.reset_simulation()
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(x * self.scale_x + self.offset_x)
        screen_y = int(y * self.scale_y + self.offset_y)
        return screen_x, screen_y
    
    def draw_grid(self):
        """Draw background grid"""
        # Main view background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                        (0, self.top_panel_height, self.main_view_width, self.main_view_height))
        
        # Grid lines
        grid_spacing = 100  # meters
        for x in range(0, int(self.env.service_area[0]) + 1, grid_spacing):
            screen_x, _ = self.world_to_screen(x, 0)
            pygame.draw.line(self.screen, self.colors['grid'],
                           (screen_x, self.offset_y),
                           (screen_x, self.offset_y + self.main_view_height - 40), 1)
        
        for y in range(0, int(self.env.service_area[1]) + 1, grid_spacing):
            _, screen_y = self.world_to_screen(0, y)
            pygame.draw.line(self.screen, self.colors['grid'],
                           (self.offset_x, screen_y),
                           (self.offset_x + self.main_view_width - 40, screen_y), 1)
    
    def draw_uav(self, uav: UAV):
        """Draw a UAV with its coverage area"""
        x, y = self.world_to_screen(uav.position[0], uav.position[1])
        
        # Coverage area (if enabled)
        if self.show_coverage:
            # Coverage radius based on power and height
            coverage_radius = 100 + 20 * np.sqrt(uav.current_power) + 0.2 * uav.position[2]
            screen_radius = int(coverage_radius * self.scale_x)
            
            # Create semi-transparent surface for coverage
            coverage_surface = pygame.Surface((screen_radius * 2, screen_radius * 2), pygame.SRCALPHA)
            
            # Gradient effect for coverage
            for i in range(screen_radius, 0, -5):
                alpha = int(30 * (1 - i / screen_radius))
                color = (*self.colors['uav'][:3], alpha)
                pygame.draw.circle(coverage_surface, color, (screen_radius, screen_radius), i)
            
            self.screen.blit(coverage_surface, (x - screen_radius, y - screen_radius))
        
        # UAV triangle
        size = 15
        color = self.colors['uav_selected'] if uav.id == self.selected_uav else self.colors['uav']
        
        # Animate rotation
        angle = self.animation_time * 2 + uav.id * 120
        points = []
        for i in range(3):
            point_angle = angle + i * 120
            px = x + size * math.cos(math.radians(point_angle))
            py = y + size * math.sin(math.radians(point_angle))
            points.append((px, py))
        
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
        
        # UAV ID and info
        text = self.font_small.render(f"UAV {uav.id}", True, self.colors['text'])
        self.screen.blit(text, (x - 20, y + 20))
        
        # Height indicator
        height_text = self.font_small.render(f"{uav.position[2]:.0f}m", True, self.colors['text'])
        self.screen.blit(height_text, (x - 20, y + 35))
        
        # Battery indicator
        battery_percent = uav.current_battery / uav.battery_capacity
        battery_color = self.get_status_color(battery_percent)
        battery_width = 30
        battery_height = 5
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (x - battery_width // 2, y - 25, battery_width, battery_height), 1)
        pygame.draw.rect(self.screen, battery_color,
                        (x - battery_width // 2 + 1, y - 24, 
                         int((battery_width - 2) * battery_percent), battery_height - 2))
    
    def draw_ue(self, ue: UE):
        """Draw a user equipment"""
        # Filter by slice if needed
        if self.selected_slice is not None and ue.slice_type != self.selected_slice:
            return
        
        x, y = self.world_to_screen(ue.position[0], ue.position[1])
        
        # Get color based on slice type
        color_map = {
            "embb": self.colors['ue_embb'],
            "urllc": self.colors['ue_urllc'],
            "mmtc": self.colors['ue_mmtc']
        }
        color = color_map.get(ue.slice_type, self.colors['text'])
        
        # Draw UE as circle
        pygame.draw.circle(self.screen, color, (x, y), 4)
        
        # Draw connection to assigned UAV
        if self.show_connections and ue.assigned_uav is not None:
            uav = self.env.uavs[ue.assigned_uav]
            uav_x, uav_y = self.world_to_screen(uav.position[0], uav.position[1])
            
            # Calculate connection quality (based on SINR)
            rb = ue.assigned_rb[0] if ue.assigned_rb else None
            sinr = self.env._calculate_sinr(self.env.ues[ue.id], uav, rb)
            quality = min(1.0, max(0.0, sinr / 20))  # Normalize SINR to 0-1, clamp values
            
            # Connection line with quality-based color (RGB only, no alpha)
            connection_color = self.interpolate_color(
                (255, 100, 100), (100, 255, 100), quality
            )
            
            # Draw dashed line for connection (remove alpha channel)
            self.draw_dashed_line(self.screen, connection_color, 
                                (x, y), (uav_x, uav_y), 2, 5)
    
    def draw_demand_area(self, da: DemandArea):
        """Draw demand area boundaries"""
        if not self.show_das or da.center_position is None:
            return
        
        # Filter by slice if needed
        if self.selected_slice is not None and da.slice_type != self.selected_slice:
            return
        
        # Get all UE positions in this DA
        ue_positions = []
        for ue_id in da.user_ids:
            if ue_id in self.env.ues:
                ue = self.env.ues[ue_id]
                x, y = self.world_to_screen(ue.position[0], ue.position[1])
                ue_positions.append((x, y))
        
        if len(ue_positions) < 3:
            return
        
        # Draw convex hull around UEs
        # Simple approach: draw lines between outermost points
        if len(ue_positions) > 2:
            # Find convex hull (simplified)
            hull_points = self.simple_convex_hull(ue_positions)
            
            # Ensure we have at least 3 points for polygon drawing
            if len(hull_points) >= 3:
                # Draw semi-transparent polygon
                color_map = {
                    "embb": (100, 150, 255, 30),   # Blue for eMBB
                    "urllc": (100, 255, 150, 30),   # Green for URLLC  
                    "mmtc": (255, 180, 100, 30)    # Orange for mMTC
                }
                color = color_map.get(da.slice_type, (200, 200, 200, 30))
                
                # Create surface for transparent drawing
                surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                pygame.draw.polygon(surface, color, hull_points)
                pygame.draw.polygon(surface, (*color[:3], 100), hull_points, 2)
                self.screen.blit(surface, (0, 0))
            elif len(hull_points) == 2:
                # Draw line if only 2 points
                color_map = {
                    "embb": (100, 150, 255, 100),   # Blue for eMBB
                    "urllc": (100, 255, 150, 100),   # Green for URLLC  
                    "mmtc": (255, 180, 100, 100)    # Orange for mMTC
                }
                color = color_map.get(da.slice_type, (200, 200, 200, 100))
                pygame.draw.line(self.screen, color[:3], hull_points[0], hull_points[1], 3)
        
        # Draw DA info at center
        center_x, center_y = self.world_to_screen(da.center_position[0], da.center_position[1])
        slice_names = {"embb": "eMBB", "urllc": "URLLC", "mmtc": "mMTC"}
        text = self.font_small.render(f"DA{da.id} ({slice_names[da.slice_type]})", 
                                    True, self.colors['text'])
        self.screen.blit(text, (center_x - 30, center_y - 10))
    
    def draw_interference(self):
        """Draw interference patterns between UAVs"""
        if not self.show_interference:
            return
        
        # Create interference surface
        surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        
        for i, uav1 in self.env.uavs.items():
            for j, uav2 in self.env.uavs.items():
                if i >= j:
                    continue
                
                # Calculate interference level
                distance = np.linalg.norm(uav1.position - uav2.position)
                interference = (uav1.current_power * uav2.current_power) / (distance ** 2 + 1e-8)
                normalized_interference = min(1.0, interference / 0.001)  # Normalize
                
                # Draw interference line
                x1, y1 = self.world_to_screen(uav1.position[0], uav1.position[1])
                x2, y2 = self.world_to_screen(uav2.position[0], uav2.position[1])
                
                # Color based on interference level
                color = self.interpolate_color(
                    (100, 255, 100), (255, 100, 100), normalized_interference
                )
                alpha = int(100 * normalized_interference)
                
                pygame.draw.line(surface, (*color, alpha), (x1, y1), (x2, y2), 
                               int(1 + 3 * normalized_interference))
        
        self.screen.blit(surface, (0, 0))
    
    def draw_top_panel(self):
        """Draw top information panel"""
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'], 
                        (0, 0, self.window_width, self.top_panel_height))
        pygame.draw.line(self.screen, self.colors['grid'], 
                        (0, self.top_panel_height), 
                        (self.window_width, self.top_panel_height), 2)
        
        # Title
        title = self.font_large.render("UAV Network Slicing Visualizer", True, self.colors['text'])
        self.screen.blit(title, (10, 10))
        
        # Current metrics
        if hasattr(self, 'current_metrics'):
            metrics_text = (f"QoS: {self.current_metrics['qos']:.2%} | "
                          f"Energy: {(1-self.current_metrics['energy']):.2%} | "
                          f"Interference: {self.current_metrics['interference']:.3f}")
            metrics = self.font_medium.render(metrics_text, True, self.colors['text'])
            self.screen.blit(metrics, (10, 45))
        
        # Time
        time_text = self.font_medium.render(f"Time: {self.env.current_time:.1f}s", 
                                          True, self.colors['text'])
        self.screen.blit(time_text, (self.window_width - 150, 45))
    
    def draw_side_panel(self):
        """Draw side control panel"""
        # Background
        pygame.draw.rect(self.screen, self.colors['panel_bg'],
                        (self.side_panel_x, 0, 300, self.window_height))
        pygame.draw.line(self.screen, self.colors['grid'],
                        (self.side_panel_x, 0),
                        (self.side_panel_x, self.window_height), 2)
        
        # Panel title
        title = self.font_medium.render("Controls", True, self.colors['text'])
        self.screen.blit(title, (self.side_panel_x + 10, 10))
        
        # Statistics
        y_offset = 40
        stats_title = self.font_medium.render("Network Statistics", True, self.colors['text'])
        self.screen.blit(stats_title, (self.side_panel_x + 10, y_offset))
        
        y_offset += 30
        stats = [
            f"UAVs: {len(self.env.uavs)}",
            f"UEs: {len(self.env.ues)}  Active: {sum(1 for ue in self.env.ues.values() if ue.is_active)}  Inactive: {sum(1 for ue in self.env.ues.values() if not ue.is_active)}",
            f"Demand Areas: {len(self.env.demand_areas)}"
        ]
        
        for stat in stats:
            text = self.font_small.render(stat, True, self.colors['text'])
            self.screen.blit(text, (self.side_panel_x + 20, y_offset))
            y_offset += 20
        
        # View options
        y_offset = 130
        options_title = self.font_medium.render("View Options", True, self.colors['text'])
        self.screen.blit(options_title, (self.side_panel_x + 10, y_offset))
        
        # Slice filter
        y_offset = 230
        filter_title = self.font_small.render("Filter by Slice:", True, self.colors['text'])
        self.screen.blit(filter_title, (self.side_panel_x + 10, y_offset))
        
        # UAV selection info
        if self.selected_uav is not None and self.selected_uav in self.env.uavs:
            y_offset = 300
            uav = self.env.uavs[self.selected_uav]
            uav_title = self.font_medium.render(f"UAV {self.selected_uav} Info", 
                                              True, self.colors['text'])
            self.screen.blit(uav_title, (self.side_panel_x + 10, y_offset))
            
            y_offset += 30
            info = [
                f"Position: ({uav.position[0]:.0f}, {uav.position[1]:.0f}, {uav.position[2]:.0f})",
                f"Power: {uav.current_power:.1f}W / {uav.max_power}W",
                f"Battery: {uav.current_battery:.0f}J / {uav.battery_capacity}J",
                f"Bandwidth: {uav.max_bandwidth/1e6:.1f}MHz"
            ]
            
            for line in info:
                text = self.font_small.render(line, True, self.colors['text'])
                self.screen.blit(text, (self.side_panel_x + 20, y_offset))
                y_offset += 20
                    
            # Add path information
            if self.selected_uav in self.uav_paths:
                path_length = len(self.uav_paths[self.selected_uav])
                if path_length > 1:
                    # Calculate total distance traveled
                    total_distance = 0
                    path = self.uav_paths[self.selected_uav]
                    for i in range(1, len(path)):
                        dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
                        total_distance += dist
                    
                    path_info = [
                        f"Path points: {path_length}",
                        f"Distance traveled: {total_distance:.1f}m"
                    ]
                    
                    for line in path_info:
                        text = self.font_small.render(line, True, self.colors['text'])
                        self.screen.blit(text, (self.side_panel_x + 20, y_offset))
                        y_offset += 20
        
        # Performance graph
        self.draw_performance_graph()
        
        # Draw buttons
        for button in self.buttons.values():
            button.draw(self.screen, self.font_small)   
    
    def draw_performance_graph(self):
        """Draw mini performance graph"""
        graph_x = self.side_panel_x + 10
        graph_y = 450
        graph_width = 280
        graph_height = 150
        
        # Background
        pygame.draw.rect(self.screen, (40, 40, 50),
                        (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, self.colors['grid'],
                        (graph_x, graph_y, graph_width, graph_height), 1)
        
        # Title
        title = self.font_small.render("Performance History", True, self.colors['text'])
        self.screen.blit(title, (graph_x + 5, graph_y - 20))
        
        # Draw metrics
        if len(self.metrics_history['qos']) > 1:
            # QoS line (green)
            self.draw_metric_line(self.metrics_history['qos'], graph_x, graph_y, 
                                graph_width, graph_height, self.colors['good'])
            
            # Energy line (yellow)
            self.draw_metric_line(self.metrics_history['energy'], graph_x, graph_y,
                                graph_width, graph_height, self.colors['warning'])
            
            # Interference line (red)
            interference_normalized = [i * 10 for i in self.metrics_history['interference']]
            self.draw_metric_line(interference_normalized, graph_x, graph_y,
                                graph_width, graph_height, self.colors['bad'])
        
        # Legend
        legend_y = graph_y + graph_height + 5
        legends = [
            ("QoS", self.colors['good']),
            ("Energy", self.colors['warning']),
            ("Interf.", self.colors['bad'])
        ]
        
        legend_x = graph_x + 5
        for label, color in legends:
            pygame.draw.rect(self.screen, color, (legend_x, legend_y, 10, 10))
            text = self.font_small.render(label, True, self.colors['text'])
            self.screen.blit(text, (legend_x + 15, legend_y - 2))
            legend_x += 80
    
    def draw_metric_line(self, data: List[float], x: int, y: int, 
                        width: int, height: int, color: Tuple[int, int, int]):
        """Draw a single metric line on the graph"""
        if len(data) < 2:
            return
        
        points = []
        for i, value in enumerate(data[-50:]):  # Show last 50 points
            px = x + (i / len(data[-50:])) * width
            py = y + height - (value * height)
            points.append((px, py))
        
        pygame.draw.lines(self.screen, color, False, points, 2)
    
    def draw_dashed_line(self, surface, color, start_pos, end_pos, width=1, dash_length=5):
        """Draw a dashed line"""
        # Ensure color is RGB tuple (remove alpha if present)
        if len(color) > 3:
            color = color[:3]
        
        x1, y1 = start_pos
        x2, y2 = end_pos
        distance = math.hypot(x2 - x1, y2 - y1)
        
        if distance == 0:
            return
        
        dashes = int(distance // dash_length)
        if dashes == 0:
            dashes = 1
        
        for i in range(0, dashes, 2):
            start = i / dashes
            end = min((i + 1) / dashes, 1)
            
            start_x = x1 + (x2 - x1) * start
            start_y = y1 + (y2 - y1) * start
            end_x = x1 + (x2 - x1) * end
            end_y = y1 + (y2 - y1) * end
            
            pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), width)

    def draw_uav_paths(self):
        """Draw UAV movement paths"""
        if not self.show_paths:
            return
        
        for uav_id, path in self.uav_paths.items():

            if len(path) < 2:
                continue
            
            # Get UAV color (similar to UAV drawing)
            base_color = self.colors['uav'] if uav_id != self.selected_uav else self.colors['uav_selected']
            
            # Convert path to screen coordinates
            screen_path = []
            for pos in path:
                screen_x, screen_y = self.world_to_screen(pos[0], pos[1])
                screen_path.append((screen_x, screen_y))
            
            # Draw path with gradient effect (older positions are more transparent)
            for i in range(len(screen_path) - 1):
                # Calculate alpha based on position in path (newer = more opaque)
                alpha_factor = (i + 1) / len(screen_path)
                alpha = int(50 + 150 * alpha_factor)  # Range from 50 to 200
                
                # Create color with alpha
                path_color = (*base_color, alpha)
                
                # Draw line segment with varying thickness
                thickness = int(1 + 2 * alpha_factor)  # Thicker for newer segments
                
                # Draw on a separate surface for alpha blending
                path_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                pygame.draw.line(path_surface, path_color, screen_path[i], screen_path[i + 1], thickness)
                self.screen.blit(path_surface, (0, 0))
            
            # Draw path markers (small circles) at key positions
            marker_interval = max(1, len(screen_path) // 10)  # Show ~10 markers per path
            for i in range(0, len(screen_path), marker_interval):
                if i < len(screen_path):
                    alpha_factor = (i + 1) / len(screen_path)
                    marker_size = int(2 + 3 * alpha_factor)
                    marker_color = (*base_color, int(100 + 100 * alpha_factor))
                    
                    marker_surface = pygame.Surface((marker_size * 2, marker_size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(marker_surface, marker_color, (marker_size, marker_size), marker_size)
                    self.screen.blit(marker_surface, (screen_path[i][0] - marker_size, screen_path[i][1] - marker_size))
            
            # Draw direction arrows for recent movement
            if len(screen_path) >= 2:
                # Draw arrow at the end of path to show movement direction
                start_pos = screen_path[-2]
                end_pos = screen_path[-1]
                
                # Calculate arrow direction
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 5:  # Only draw arrow if movement is significant
                    # Normalize direction
                    dx /= length
                    dy /= length
                    
                    # Arrow head size
                    arrow_size = 8
                    
                    # Calculate arrow head points
                    arrow_tip = end_pos
                    left_point = (
                        arrow_tip[0] - arrow_size * dx + arrow_size * 0.5 * dy,
                        arrow_tip[1] - arrow_size * dy - arrow_size * 0.5 * dx
                    )
                    right_point = (
                        arrow_tip[0] - arrow_size * dx - arrow_size * 0.5 * dy,
                        arrow_tip[1] - arrow_size * dy + arrow_size * 0.5 * dx
                    )
                    
                    # Draw arrow head
                    arrow_surface = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
                    pygame.draw.polygon(arrow_surface, (*base_color, 200), 
                                    [arrow_tip, left_point, right_point])
                    self.screen.blit(arrow_surface, (0, 0))

    def simple_convex_hull(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Simple convex hull algorithm (gift wrapping)"""
        if len(points) < 3:
            return points
        
        # Remove duplicate points
        unique_points = list(set(points))
        if len(unique_points) < 3:
            return unique_points
        
        # Find leftmost point
        start = min(unique_points, key=lambda p: (p[0], p[1]))
        hull = [start]
        current = start
        
        max_iterations = len(unique_points) + 1  # Prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            next_point = None
            
            for point in unique_points:
                if point == current:
                    continue
                
                if next_point is None:
                    next_point = point
                    continue
                    
                # Cross product to determine if point is more counterclockwise
                cross = ((point[0] - current[0]) * (next_point[1] - current[1]) -
                        (point[1] - current[1]) * (next_point[0] - current[0]))
                
                if cross > 0:
                    next_point = point
                elif cross == 0:
                    # Points are collinear, choose the farther one
                    dist_current = ((point[0] - current[0])**2 + (point[1] - current[1])**2)
                    dist_next = ((next_point[0] - current[0])**2 + (next_point[1] - current[1])**2)
                    if dist_current > dist_next:
                        next_point = point
            
            if next_point is None or next_point == start:
                break
            
            current = next_point
            hull.append(current)
        
        # Ensure we have at least 3 unique points
        if len(hull) < 3 and len(unique_points) >= 3:
            return unique_points[:3]  # Return first 3 unique points as fallback
        
        return hull
    
    def concave_hull(self, points: List[Tuple[int, int]], alpha: float = 50.0) -> List[Tuple[int, int]]:
        """
        Create concave hull (alpha shape) that wraps tightly around points
        Args:
            points: List of (x, y) coordinates
            alpha: Controls concaveness - smaller values = more concave, larger = more convex
        """
        if len(points) < 3:
            return points
        
        # Remove duplicate points
        unique_points = list(set(points))
        if len(unique_points) < 3:
            return unique_points
        
        # For small point sets, use a simpler approach
        if len(unique_points) <= 6:
            return self.simple_boundary_trace(unique_points)
        
        # Alpha shape algorithm (simplified version)
        from scipy.spatial import Delaunay
        import numpy as np
        
        try:
            # Convert to numpy array for easier processing
            points_array = np.array(unique_points)
            
            # Create Delaunay triangulation
            tri = Delaunay(points_array)
            
            # Find boundary edges using alpha criterion
            boundary_edges = set()
            
            for simplex in tri.simplices:
                # Get triangle vertices
                a, b, c = points_array[simplex]
                
                # Calculate circumradius of triangle
                def circumradius(p1, p2, p3):
                    # Calculate circumradius using cross products
                    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if abs(d) < 1e-10:
                        return float('inf')
                    
                    ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + 
                        (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + 
                        (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
                    
                    uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + 
                        (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + 
                        (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
                    
                    return np.sqrt((ux - p1[0])**2 + (uy - p1[1])**2)
                
                radius = circumradius(a, b, c)
                
                # If circumradius is small enough, add edges to boundary
                if radius < alpha:
                    edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
                    for edge in edges:
                        # Check if edge is on boundary (appears in only one triangle)
                        edge_key = tuple(sorted(edge))
                        if self.is_boundary_edge(tri, edge_key):
                            boundary_edges.add(edge_key)
            
            # Trace boundary path
            if boundary_edges:
                boundary_path = self.trace_boundary_path(boundary_edges, points_array)
                return [(int(p[0]), int(p[1])) for p in boundary_path]
            else:
                # Fallback to convex hull if alpha shape fails
                return self.simple_convex_hull(unique_points)
                
        except ImportError:
            # Fallback if scipy not available
            return self.simple_boundary_trace(unique_points)
        except Exception:
            # Any other error, fallback to convex hull
            return self.simple_convex_hull(unique_points)

    def simple_boundary_trace(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Simple boundary tracing for small point sets using nearest neighbor approach
        """
        if len(points) < 3:
            return points
        
        # Start with leftmost point
        start = min(points, key=lambda p: (p[0], p[1]))
        boundary = [start]
        current = start
        visited = {start}
        
        max_iterations = len(points) * 2  # Prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations and len(visited) < len(points):
            iterations += 1
            
            # Find nearest unvisited point
            candidates = [p for p in points if p not in visited]
            if not candidates:
                break
            
            # Choose point that creates most "outward" turn
            best_point = None
            best_score = float('-inf')
            
            for candidate in candidates:
                # Calculate angle and distance
                if len(boundary) >= 2:
                    # Vector from previous point to current
                    prev = boundary[-2]
                    v1 = (current[0] - prev[0], current[1] - prev[1])
                    # Vector from current to candidate
                    v2 = (candidate[0] - current[0], candidate[1] - current[1])
                    
                    # Cross product for turn direction
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    
                    # Distance
                    dist = ((candidate[0] - current[0])**2 + (candidate[1] - current[1])**2)**0.5
                    
                    # Score favors outward turns and closer points
                    score = cross / (dist + 1)
                else:
                    # For first point, just use distance
                    dist = ((candidate[0] - current[0])**2 + (candidate[1] - current[1])**2)**0.5
                    score = -dist  # Prefer closer points
                
                if score > best_score:
                    best_score = score
                    best_point = candidate
            
            if best_point:
                boundary.append(best_point)
                visited.add(best_point)
                current = best_point
            else:
                break
        
        return boundary

    def is_boundary_edge(self, tri, edge_key):
        """Check if edge is on the boundary of triangulation"""
        count = 0
        for simplex in tri.simplices:
            edges_in_simplex = [
                tuple(sorted([simplex[0], simplex[1]])),
                tuple(sorted([simplex[1], simplex[2]])),
                tuple(sorted([simplex[2], simplex[0]]))
            ]
            if edge_key in edges_in_simplex:
                count += 1
        
        return count == 1  # Boundary edges appear in exactly one triangle

    def trace_boundary_path(self, boundary_edges, points_array):
        """Trace a continuous path along boundary edges"""
        if not boundary_edges:
            return []
        
        # Build adjacency list
        adj = {}
        for edge in boundary_edges:
            i, j = edge
            if i not in adj:
                adj[i] = []
            if j not in adj:
                adj[j] = []
            adj[i].append(j)
            adj[j].append(i)
        
        # Find starting point (any boundary point)
        start = next(iter(adj.keys()))
        
        # Trace path
        path = [start]
        current = start
        prev = None
        
        max_iterations = len(boundary_edges) + 1
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Find next unvisited neighbor
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
            
            next_point = neighbors[0]  # Simple choice - could be improved
            path.append(next_point)
            prev = current
            current = next_point
            
            # Stop if we've returned to start
            if current == start and len(path) > 2:
                break
        
        # Convert indices to coordinates
        return [points_array[i] for i in path]

    def interpolate_color(self, color1: Tuple[int, int, int], 
                         color2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
        """Interpolate between two colors"""
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))
    
    def get_status_color(self, value: float) -> Tuple[int, int, int]:
        """Get color based on status value (0-1)"""
        if value > 0.7:
            return self.colors['good']
        elif value > 0.3:
            return self.colors['warning']
        else:
            return self.colors['bad']

    def update_uav_paths(self):
        """Update UAV path tracking"""
        for uav_id, uav in self.env.uavs.items():
            # Add current position to path
            current_pos = (uav.position[0], uav.position[1])  # Store 2D position
            
            if uav_id not in self.uav_paths:
                self.uav_paths[uav_id] = []
            
            # Only add if position has changed significantly (avoid duplicate points)
            if (not self.uav_paths[uav_id] or 
                np.linalg.norm(np.array(current_pos) - np.array(self.uav_paths[uav_id][-1])) > 5.0):
                
                self.uav_paths[uav_id].append(current_pos)
                
                # Limit path length to prevent memory issues
                if len(self.uav_paths[uav_id]) > self.max_path_length:
                    self.uav_paths[uav_id].pop(0)

    def clear_uav_paths(self):
        """Clear all UAV paths"""
        for uav_id in self.uav_paths:
            self.uav_paths[uav_id] = []

    def toggle_coverage(self):
        """Toggle coverage area display"""
        self.show_coverage = not self.show_coverage
    
    def toggle_connections(self):
        """Toggle UE-UAV connections display"""
        self.show_connections = not self.show_connections
    
    def toggle_interference(self):
        """Toggle interference display"""
        self.show_interference = not self.show_interference

    def toggle_das(self):
        """Toggle demand area display"""
        self.show_das = not self.show_das
    
    def toggle_paths(self):
        """Toggle UAV path display"""
        self.show_paths = not self.show_paths
    
    def set_slice_filter(self, slice_type: Optional[str]):
        """Set slice type filter"""
        self.selected_slice = slice_type
    
    def step_simulation(self):
        """Execute one simulation step"""
        if self.agent:
            # Get actions from agent
            actions = self.agent.select_actions(self.observations, explore=False)
            # print(f"Actions: {actions}")
            
            # Update UAV paths after movement
            self.update_uav_paths()
            
            # Execute step
            self.observations, reward, done, info = self.env.step(actions)
            
            # Update UAV paths after movement
            self.update_uav_paths()
            
            # Update metrics
            self.current_metrics = {
                'qos': info['qos_satisfaction'],
                'energy': info['energy_efficiency'],
                'interference': info['sinr_level']
            }
            
            # Update history
            self.metrics_history['qos'].append(info['qos_satisfaction'])
            self.metrics_history['energy'].append(info['energy_efficiency'])
            self.metrics_history['interference'].append(info['sinr_level'])
            
            # Limit history size
            for key in self.metrics_history:
                if len(self.metrics_history[key]) > self.max_history:
                    self.metrics_history[key].pop(0)
            
            # if done:
            #     print("Episode finished!")
            #     self.reset_simulation()
        # else:
        #     self.env.reset()
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.observations = self.env.reset()

        # Clear UAV paths on reset
        self.clear_uav_paths()
        self.current_metrics = {
            'qos': 0.0,
            'energy': 0.0,
            'interference': 0.0
        }
        self.metrics_history = {
            'qos': [],
            'energy': [],
            'interference': []
        }
    
    def handle_click(self, pos):
        """Handle mouse click events"""
        x, y = pos
        
        # Check buttons
        for button in self.buttons.values():
            if button.is_clicked(pos):
                button.callback()
                return
        
        # Check if click is in main view
        if x < self.main_view_width and y > self.top_panel_height:
            # Convert to world coordinates
            world_x = (x - self.offset_x) / self.scale_x
            world_y = (y - self.offset_y) / self.scale_y
            
            # Find nearest UAV
            min_dist = float('inf')
            nearest_uav = None
            
            for uav_id, uav in self.env.uavs.items():
                dist = np.linalg.norm(np.array([world_x, world_y]) - uav.position[:2])
                if dist < min_dist and dist < 50:  # 50m selection radius
                    min_dist = dist
                    nearest_uav = uav_id
            
            self.selected_uav = nearest_uav
    
    def run(self):
        """Main visualization loop"""
        running = True
        
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
                        self.toggle_coverage()
                    elif event.key == pygame.K_l:
                        self.toggle_connections()
                    elif event.key == pygame.K_d:
                        self.toggle_das()
                    elif event.key == pygame.K_i:
                        self.toggle_interference()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw components
            self.draw_grid()

            # Draw UAV paths (behind everything else)
            self.draw_uav_paths()
            
            # Draw interference first (background layer)
            self.draw_interference()
            
            # Draw demand areas
            for da in self.env.demand_areas.values():
                self.draw_demand_area(da)
            
            # Draw UEs
            for ue in self.env.ues.values():
                self.draw_ue(ue)
            
            # Draw UAVs
            for uav in self.env.uavs.values():
                self.draw_uav(uav)
            
            # Draw UI panels
            self.draw_top_panel()
            self.draw_side_panel()
            
            # Update animation
            self.animation_time += 0.016  # ~60 FPS
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

class Button:
    """Simple button class for pygame"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.is_hovered = False
        self.is_pressed = False
    
    def is_clicked(self, pos: Tuple[int, int]) -> bool:
        """Check if button is clicked"""
        return self.rect.collidepoint(pos)
    
    def draw(self, screen, font):
        """Draw button"""
        # Determine color based on state
        if self.is_pressed:
            color = (100, 100, 120)
        elif self.is_hovered:
            color = (80, 80, 100)
        else:
            color = (60, 60, 80)
        
        # Draw button background
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (150, 150, 150), self.rect, 2)
        
        # Draw text
        text_surface = font.render(self.text, True, (200, 200, 200))
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
        # Update hover state
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())

def main():
    """Main function to run the visualizer"""
    import argparse
    import os
    import json
    
    parser = argparse.ArgumentParser(description='UAV Network Slicing Visualizer')
    parser.add_argument('--checkpoint', type=str, default="./saved_models/model14", help='Path to trained model checkpoint')
    
    args = parser.parse_args()
    # Create environment
    env = NetworkSlicingEnv(config_path="./config/environment/default.yaml")

    # Load agent if checkpoint provided
    agent = None
    if args.checkpoint:

        # Assume checkpoint is a folder, e.g., model5
        model_dir = args.checkpoint
        model_checkpoint = os.path.join(model_dir, "checkpoints/checkpoint_step_30000.pth")

        env_config = Configuration("./config/environment/default.yaml")
        # Extract agent config from env_info.json
        num_agents = env_config.system.num_uavs
        obs_dim = 67
        action_dim = 13

        agent = MADRLAgent(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim
        )
        agent.load_models(model_checkpoint)
        print(f"Loaded model from {model_checkpoint}")
    
    # Create and run visualizer
    visualizer = NetworkVisualizer(env, agent, 1200, 800)
    visualizer.run()

if __name__ == "__main__":
    main()

