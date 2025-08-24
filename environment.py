
import numpy as np
from scipy.stats import poisson
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import Configuration

@dataclass
class QoSProfile:
    """Quality of Service profile for network slices"""
    min_rate: float  # bps
    max_latency: float  # ms
    min_reliability: float  # percentage

@dataclass
class UAV:
    """UAV entity with position and resource constraints"""
    id: int
    position: np.ndarray  # [x, y, h]
    max_bandwidth: float  # MHz
    max_power: float  # Watts
    current_power: float
    battery_capacity: float  # Joules
    current_battery: float
    velocity_max: float  # m/s
    
    def update_position(self, delta_pos: np.ndarray):
        """Update UAV position with velocity constraints"""
        self.position += delta_pos

@dataclass
class UE:
    """User Equipment entity"""
    id: int
    position: np.ndarray  # [x, y, h]
    slice_type: str  # embb, urllc, mmtc
    assigned_uav: Optional[int] = None
    assigned_da: Optional[int] = None
    is_active: bool = True  # NEW: whether UE is currently active
    velocity: np.ndarray = None  # NEW: velocity vector for movement    
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(3)
    
@dataclass
class DemandArea:
    """Demand Area for clustering users"""
    id: int
    uav_id: int
    slice_type: str
    user_ids: List[int]
    allocated_bandwidth: float = 2.0
    center_position: Optional[np.ndarray] = None
    sinr_level: str = "Medium"  # New field: "Low", "Medium", "High"

class NetworkSlicingEnv:
    """
    UAV-based network slicing environment with CTDE support
    """
    def __init__(self, config_path):

        env_config = Configuration(config_path)

        # Load configuration parameters

        self.num_uavs = env_config.system.num_uavs
        self.num_ues = env_config.system.num_ues
        self.service_area = tuple(env_config.system.service_area)  # (width, height)
        self.height_range = tuple(env_config.system.height_range)  # (min_height, max_height)
        self.num_das_per_slice = env_config.system.num_das_per_slice

        # Time scale parameters
        self.T_L = env_config.time.T_L  # Long-term period (seconds)
        self.T_S = env_config.time.T_S  # Short-term period (seconds)
        self.current_time = 0.0
        
        # Network slice QoS profiles
        qos = env_config.qos_profiles
        self.qos_profiles = {
            "embb": QoSProfile(min_rate=qos.embb.min_rate, 
                          max_latency=qos.embb.max_latency, 
                          min_reliability=qos.embb.min_reliability),   # eMBB
            "urllc": QoSProfile(min_rate=qos.urllc.min_rate, 
                          max_latency=qos.urllc.max_latency, 
                          min_reliability=qos.urllc.min_reliability),   # URLLC
            "mmtc": QoSProfile(min_rate=qos.mmtc.min_rate, 
                          max_latency=qos.mmtc.max_latency, 
                          min_reliability=qos.mmtc.min_reliability)     # mMTC
        }
        
        # Slice-related parameters
        self.slice_weights = env_config.slicing.slice_weights  # Weights for each slice type
        self.slice_probs = env_config.slicing.slice_probabilities  # Probabilities for each slice type

        # Channel parameters
        self.noise_power = env_config.channel.noise_power  # Watts
        self.path_loss_exponent = env_config.channel.path_loss_exponent
        self.carrier_frequency = env_config.channel.carrier_frequency  # Hz

        # Resource block parameters
        self.rb_bandwidth = env_config.channel.rb_bandwidth  # Hz (LTE standard)
        self.total_bandwidth = env_config.channel.total_bandwidth  # Hz per UAV
        self.total_rbs = int(self.total_bandwidth / self.rb_bandwidth)
        print(f"Total Resource Blocks per UAV: {self.total_rbs}")

        # UAV parameters
        self.uav_params = env_config.uav_params  # UAV parameters like max power, battery, etc.
        
        # UE dynamics parameters
        self.ue_dynamics = env_config.ue_dynamics  # UE dynamics parameters
        self.ue_arrival_rate = self.ue_dynamics.arrival_rate  # New UEs per second
        self.ue_departure_rate = self.ue_dynamics.departure_rate  # Probability of UE leaving per minute
        self.ue_max_initial_velocity = self.ue_dynamics.max_initial_velocity  # m/s
        self.max_ues = int(self.num_ues * self.ue_dynamics.max_ues_multiplier)  # Maximum UEs allowed
        self.next_ue_id = self.num_ues  # For generating new UE IDs
        self.change_direction_prob = self.ue_dynamics.change_direction_prob  # Probability of changing direction


        # Distance thresholds for forming Demand Areas
        self.da_distance_thresholds = env_config.da_distance_thresholds

        # SINR thresholds for DA classification (in dB)
        self.sinr_thresholds = env_config.sinr_thresholds  

        # Reward weights
        self.reward_weights = env_config.reward_weights

        # Energy consumption parameters
        self.movement_energy_factor = env_config.energy_models.movement_energy_factor  # Joules per meter
        self.transmission_energy_factor = env_config.energy_models.transmission_energy_factor  # Joules per

        # Statistics tracking
        self.stats = {
            'arrivals': 0,
            'departures': 0,
            'handovers': 0
        }
        
        # Initialize entities
        self.reset()
        self._print_statistics()
        
    def reset(self) -> Dict[int, np.ndarray]:   
        """Reset environment to initial state"""
        # Initialize UAVs at random positions within service area
        self.uavs = {}
        for i in range(self.num_uavs):
            x = np.random.uniform(0, self.service_area[0])
            y = np.random.uniform(0, self.service_area[1])
            h = np.random.uniform(self.height_range[0], self.height_range[1])
            
            self.uavs[i] = UAV(
                id=i,
                position=np.array([x, y, h]),
                max_bandwidth=self.total_bandwidth,
                max_power=self.uav_params.max_power,  # Watts
                current_power=self.uav_params.initial_power,
                battery_capacity=self.uav_params.battery_capacity,  # Joules
                current_battery=self.uav_params.initial_battery,  # Start fully charged
                velocity_max=self.uav_params.velocity_max  # m/s
            )
        
        # Initialize UEs with random positions and slice types
        self.ues = {}
        self.next_ue_id = 0

        for i in range(self.num_ues):
            self._add_new_ue()
        
        # Initialize demand areas
        self.demand_areas = {}
        self.da_counter = 0
        
        # Perform initial UAV-UE association and DA formation
        self._associate_ues_to_uavs()
        self._form_demand_areas()
        
        # Get initial observations for each UAV
        observations = self._get_observations()
        
        self.current_time = 0.0
        return observations
    
    def _calculate_channel_gain(self, uav_pos: np.ndarray, ue_pos: np.ndarray) -> float:
        """Calculate channel gain between UAV and UE"""
        distance = np.linalg.norm(uav_pos - ue_pos)
        # Free space path loss model
        wavelength = 3e8 / self.carrier_frequency
        path_loss = (wavelength / (4 * np.pi * distance)) ** self.path_loss_exponent
        return path_loss
    
    def _calculate_sinr(self, uav_id: int, ue_id: int) -> float:
        """Calculate SINR for UE from serving UAV"""
        uav = self.uavs[uav_id]
        ue = self.ues[ue_id]
        
        # Signal power from serving UAV
        channel_gain = self._calculate_channel_gain(uav.position, ue.position)
        signal_power = uav.current_power * channel_gain
        
        # Interference from other UAVs
        interference = 0.0
        for other_uav_id, other_uav in self.uavs.items():
            if other_uav_id != uav_id:
                other_gain = self._calculate_channel_gain(other_uav.position, ue.position)
                interference += other_uav.current_power * other_gain
        
        # Calculate SINR
        sinr = signal_power / (interference + self.noise_power)
        sinr_db = 10 * np.log10(sinr)  # Convert to dB
        return sinr_db

    def _update_ue_dynamics(self):
        """Update UE positions, handle arrivals/departures"""
        # Reset statistics for this step
        self.stats['arrivals'] = 0
        self.stats['departures'] = 0
    
        # 1. Update existing UE positions
        for ue in list(self.ues.values()):
            if not ue.is_active:
                continue
            
            # Update position based on velocity
            new_position = ue.position + ue.velocity * self.T_L
            
            # Boundary handling - bounce off walls
            for dim in range(2):  # Only x and y
                if new_position[dim] < 0 or new_position[dim] > self.service_area[dim]:
                    # Bounce: reverse velocity in this dimension
                    ue.velocity[dim] = -ue.velocity[dim]
                    new_position[dim] = np.clip(new_position[dim], 0, self.service_area[dim])
            
            ue.position = new_position
            
            # Randomly change direction occasionally
            if np.random.random() < self.change_direction_prob:  # change direction (default 30% chance)
                # Keep same speed but change direction
                speed = np.linalg.norm(ue.velocity[:2])
                if speed > 0:
                    new_direction = np.random.uniform(0, 2 * np.pi)
                    ue.velocity[0] = speed * np.cos(new_direction)
                    ue.velocity[1] = speed * np.sin(new_direction)
            
            # Check if UE should leave (default 8% chance per minute)
            if np.random.random() < self.ue_departure_rate:
                ue.is_active = False
                self.stats['departures'] += 1
        
        # 2. Handle new UE arrivals
        active_ue_count = len([ue for ue in self.ues.values() if ue.is_active])
        
        # Poisson arrivals
        expected_arrivals = self.ue_arrival_rate * self.T_L
        num_arrivals = min(
            np.random.poisson(expected_arrivals),
            self.max_ues - active_ue_count
        )
        
        for _ in range(num_arrivals):
            self._add_new_ue()
            self.stats['arrivals'] += 1
        
        # 3. Clean up inactive UEs periodically
        if len(self.ues) > self.max_ues:
            inactive_ues = [ue_id for ue_id, ue in self.ues.items() if not ue.is_active]
            for ue_id in inactive_ues[:len(inactive_ues)//2]:  # Remove half of inactive UEs
                del self.ues[ue_id]
    
    def _add_new_ue(self):
        """Add a new UE to the network"""
        # Simple random position anywhere in service area
        x = np.random.uniform(0, self.service_area[0])
        y = np.random.uniform(0, self.service_area[1])
        
        # Simple random velocity (0-2 m/s in random direction)
        speed = np.random.uniform(0, self.ue_max_initial_velocity)
        direction = np.random.uniform(0, 2 * np.pi)
        velocity_x = speed * np.cos(direction)
        velocity_y = speed * np.sin(direction)
        
        # Equal probability for all slice types
        slice_type = np.random.choice(
            ["embb", "urllc", "mmtc"],
            p=[self.slice_probs["embb"], self.slice_probs["urllc"], self.slice_probs["mmtc"]]
        )


        # Create new UE
        new_ue = UE(
            id=self.next_ue_id,
            position=np.array([x, y, 0.0]),
            slice_type=slice_type,
            velocity=np.array([velocity_x, velocity_y, 0.0]),
            is_active=True
        )
        
        self.ues[self.next_ue_id] = new_ue
        self.next_ue_id += 1
        
    def _associate_ues_to_uavs(self):
        """Associate UEs to nearest UAV based on distance"""
        self.stats['handovers'] = 0
        
        for ue in self.ues.values():
            if not ue.is_active:
                continue
                
            min_distance = float('inf')
            best_uav = None
            previous_uav = ue.assigned_uav
            
            for uav in self.uavs.values():
                distance = np.linalg.norm(uav.position - ue.position)
                if distance < min_distance:
                    min_distance = distance
                    best_uav = uav.id
            
            ue.assigned_uav = best_uav
            
            # Track handovers
            if previous_uav is not None and previous_uav != best_uav:
                self.stats['handovers'] += 1
   
    def _form_demand_areas(self):
        """Form demand areas by clustering UEs into 3 static distance levels per slice type"""
        self.demand_areas.clear()
        self.da_counter = 0
        
        # Distance thresholds for DA classification (in meters)
        distance_thresholds = self.da_distance_thresholds
        
        for uav_id in range(self.num_uavs):
            # Get active UEs assigned to this UAV
            uav_ues = [ue for ue in self.ues.values() 
                    if ue.assigned_uav == uav_id and ue.is_active]
            
            # Group by slice type
            for slice_type in ["embb", "urllc", "mmtc"]:
                slice_ues = [ue for ue in uav_ues if ue.slice_type == slice_type]
                
                # Initialize distance groups (but call them SINR groups externally)
                distance_groups = {
                    'Low': [],      # Actually near UEs (distance <= 200m)
                    'Medium': [],   # Actually medium UEs (200m < distance <= 400m)
                    'High': []      # Actually far UEs (distance > 400m)
                }
                
                if len(slice_ues) > 0:
                    # Calculate DISTANCE for each UE and categorize 
                    for ue in slice_ues:
                        distance = np.linalg.norm(self.uavs[uav_id].position - ue.position)

                        # Categorize UE based on DISTANCE
                        if distance <= distance_thresholds['near_max']:
                            distance_groups['Low'].append(ue)      # Near = "Low" SINR level externally
                        elif distance <= distance_thresholds['medium_max']:
                            distance_groups['Medium'].append(ue)   # Medium = "Medium" SINR level externally
                        else:
                            distance_groups['High'].append(ue)     # Far = "High" SINR level externally
                

                # Create DAs for each distance level (actually distance level, even if empty)
                for level_name in ['Low', 'Medium', 'High']:
                    ue_group = distance_groups[level_name]
                    
                    da = DemandArea(
                        id=self.da_counter,
                        uav_id=uav_id,
                        slice_type=slice_type,
                        user_ids=[ue.id for ue in ue_group],
                        sinr_level=level_name  # Keep the same field name for external compatibility
                    )
                    
                    # Calculate center position (if UEs exist)
                    if ue_group:
                        positions = np.array([ue.position for ue in ue_group])
                        da.center_position = np.mean(positions, axis=0)
                        
                        # Update UE assignments
                        for ue in ue_group:
                            ue.assigned_da = da.id
                    else:
                        # Empty DA - set center to UAV position as placeholder
                        da.center_position = self.uavs[uav_id].position.copy()
                    
                    self.demand_areas[self.da_counter] = da
                    self.da_counter += 1   
        
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """Get observations for each UAV agent
        Logic: 
        Each UAV agent combines:
        - The UAV's state: 
            + position (x y h) 
            + power level (normalized)
            + battery level (normalized)
        - The UAV's demand area info:
            - Each DA's info: (num_slice * num_sinr_levels = 3 * 3 = 9)
                + number of users (normalized by dividing by total UEs)
                + slice type (one hot encoded)
                + allocated bandwidth (normalized by dividing by total bandwidth)
                + connection quality level (normalized)
        - The UAV's surrounding condition:
            - number of UEs to each direction (N, E, S, W)
            - interference from other UAVs to each direction (N, E, S, W)

        => Total dimension: 5 (UAV state) + 9 (DA info) * 5 + 8 (surrounding condition) = 4 + 45 + 8 = 58
        """
        observations = {}

        num_total_ues = max(1, len([ue for ue in self.ues.values() if ue.is_active]))
        slice_types = ["embb", "urllc", "mmtc"]
        sinr_levels = ["Low", "Medium", "High"]

        for uav_id, uav in self.uavs.items():
            obs = []

            # UAV state
            obs.extend(uav.position.tolist())  # x, y, h (3)
            obs.append(uav.current_power / uav.max_power)  # normalized power (1)
            obs.append(uav.current_battery / uav.battery_capacity)  # normalized battery (1)

            # Demand Area info (fixed order: embb-Low, embb-Medium, embb-High, urllc-Low, ...)
            for slice_type in slice_types: # (x3)
                for sinr_level in sinr_levels: # (x3)
                    da = next((da for da in self.demand_areas.values()
                               if da.uav_id == uav_id and da.slice_type == slice_type and da.sinr_level == sinr_level), None)
                    # Number of users (normalized) (1)
                    num_users = len(da.user_ids) / num_total_ues if da else 0.0 
                    obs.append(num_users)
                    # Slice type one-hot (3)
                    obs.extend([1.0 if slice_type == st else 0.0 for st in slice_types])
                    # Allocated bandwidth (normalized) (1)
                    allocated_bw = da.allocated_bandwidth / uav.max_bandwidth if da else 0.0
                    obs.append(allocated_bw)
                    # Connection quality (normalized SINR level: Low=0, Medium=0.5, High=1) (1)
                    sinr_norm = {"Low": 0.2, "Medium": 0.5, "High": 1.0}[sinr_level]
                    obs.append(sinr_norm)

            # Surrounding condition: number of UEs in N/E/S/W
            directions = {'N': 0, 'E': 0, 'S': 0, 'W': 0}
            for ue in self.ues.values():
                if not ue.is_active:
                    continue
                dx = ue.position[0] - uav.position[0]
                dy = ue.position[1] - uav.position[1]
                if abs(dx) > abs(dy):
                    if dx > 0:
                        directions['E'] += 1
                    else:
                        directions['W'] += 1
                else:
                    if dy > 0:
                        directions['N'] += 1
                    else:
                        directions['S'] += 1
            for dir in ['N', 'E', 'S', 'W']:
                obs.append(directions[dir] / num_total_ues)

            # Interference from other UAVs in N/E/S/W
            interference_dirs = {'N': 0.0, 'E': 0.0, 'S': 0.0, 'W': 0.0}
            for other_id, other_uav in self.uavs.items():
                if other_id == uav_id:
                    continue
                dx = other_uav.position[0] - uav.position[0]
                dy = other_uav.position[1] - uav.position[1]
                interference = other_uav.current_power / (np.linalg.norm(other_uav.position - uav.position) + 1e-6)
                if abs(dx) > abs(dy):
                    if dx > 0:
                        interference_dirs['E'] += interference
                    else:
                        interference_dirs['W'] += interference
                else:
                    if dy > 0:
                        interference_dirs['N'] += interference
                    else:
                        interference_dirs['S'] += interference
            # Normalize interference by max_power
            for dir in ['N', 'E', 'S', 'W']:
                obs.append(interference_dirs[dir] / uav.max_power)

            observations[uav_id] = np.array(obs, dtype=np.float32)

        return observations
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], float, bool, Dict]:
        """Execute actions and return next observations, reward, done, info"""
        # Parse and apply actions for each UAV
        for uav_id, action in actions.items():
            # print(f"UAV {uav_id} action: {action}")
            uav = self.uavs[uav_id]
            
            # Action format: [delta_x, delta_y, delta_h, delta_power, bandwidth_allocation_per_da...]
            # Position update (constrained by velocity)
            delta_pos = action[:3] * uav.velocity_max * self.T_L
            
            # delta_pos[0] *= self.service_area[0]
            # delta_pos[1] *= self.service_area[1]
            # delta_pos[2] *= (self.height_range[1] - self.height_range[0])

            # print(f"UAV {uav_id} delta_pos: {delta_pos}")

            new_pos = uav.position + delta_pos
            # Constrain to service area
            new_pos[0] = np.clip(new_pos[0], 0, self.service_area[0])
            new_pos[1] = np.clip(new_pos[1], 0, self.service_area[1])
            new_pos[2] = np.clip(new_pos[2], self.height_range[0], self.height_range[1])
            
            # Calculate movement energy cost
            movement_distance = np.linalg.norm(delta_pos)
            movement_energy = self.movement_energy_factor * movement_distance  # Simple linear model

            uav.position = new_pos
            
            # Power update
            uav.current_power = action[3] * uav.max_power  # Scale power to max_power
            
            # Energy consumption
            transmission_energy = uav.current_power * self.T_L
            uav.current_battery -= (transmission_energy + movement_energy)
            uav.current_battery = max(0, uav.current_battery)
            
            # Bandwidth allocation to DAs
            uav_das = [da for da in self.demand_areas.values() if da.uav_id == uav_id]
            if len(uav_das) > 0:
                # Normalize bandwidth allocation
                bandwidth_actions = action[4:4+len(uav_das)]
                bandwidth_fractions = np.abs(bandwidth_actions) / (np.sum(np.abs(bandwidth_actions)) + 1e-8)
                
                for i, da in enumerate(uav_das):
                    da.allocated_bandwidth = bandwidth_fractions[i] * uav.max_bandwidth
                    # print(f"UAV {uav_id} allocated {da.allocated_bandwidth:.2f} MHz to DA {da.id} (slice {da.slice_type}, SINR {da.sinr_level})")
        
        # Calculate rewards
        # print(self.demand_areas)
        reward = self._calculate_global_reward()

        # Update UE dynamics (NEW)
        self._update_ue_dynamics()
        
        # Update associations and DAs (every long-term period)
        self._associate_ues_to_uavs()
        self._form_demand_areas()
        
        # Update time
        self.current_time += self.T_L
        
        # Check termination conditions
        done = any(uav.current_battery <= 0 for uav in self.uavs.values())
        done = False
        
        # Get new observations
        observations = self._get_observations()
        
        info = {
            'qos_satisfaction': self._calculate_qos_satisfaction(),
            'energy_efficiency': self._calculate_energy_efficiency(),
            'interference_level': self._calculate_interference_level(),
            'active_ues': len([ue for ue in self.ues.values() if ue.is_active]),
            'ue_arrivals': self.stats['arrivals'],
            'ue_departures': self.stats['departures']
        }
        
        return observations, reward, done, info
    
    def _calculate_global_reward(self) -> float:
        """Calculate global reward based on QoS, energy, and interference"""
        # Weights for different objectives
        alpha = self.reward_weights.qos  # default = 0.5
        beta = self.reward_weights.energy  # default = 0.3
        gamma = self.reward_weights.interference  # default = 0.2

        qos_reward = self._calculate_qos_satisfaction()
        energy_penalty = self._calculate_energy_efficiency()
        interference_penalty = self._calculate_interference_level()

        # print(f"QoS Satisfaction: {qos_reward:.4f}, Energy Consumption Rate: {energy_penalty:.4f}, Interference Level: {interference_penalty:.4f}")
        
        reward = alpha * qos_reward - beta * energy_penalty - gamma * interference_penalty
        return reward
    
    def _calculate_qos_satisfaction(self) -> float:
        """Calculate QoS satisfaction across all DAs"""
        total_satisfaction = 0.0
        total_weight = 0.0
        
        for da in self.demand_areas.values():
            # Simple model: satisfaction based on allocated bandwidth vs required
            num_users = len(da.user_ids)
            if num_users > 0:
                qos_profile = self.qos_profiles[da.slice_type]
                required_bandwidth = num_users * qos_profile.min_rate
                
                satisfaction = min(1.0, da.allocated_bandwidth / (required_bandwidth + 1e-8))
                # print("da:", da.id, "type:", da.slice_type, "satisfaction:", satisfaction, "allocated_bandwidth:", da.allocated_bandwidth, "required_bandwidth:", required_bandwidth, "num_users:", num_users)
                weight = self.slice_weights[da.slice_type] * num_users
                
                total_satisfaction += satisfaction * weight
                total_weight += weight
    
        return total_satisfaction / (total_weight + 1e-8)
    
    def _calculate_energy_efficiency(self) -> float:
        """Calculate normalized energy consumption"""
        total_energy_ratio = 0.0
        
        for uav in self.uavs.values():
            energy_used = (uav.battery_capacity - uav.current_battery) / uav.battery_capacity
            total_energy_ratio += energy_used
        
        return total_energy_ratio / self.num_uavs
    
    def _calculate_interference_level(self) -> float:
        """Calculate interference level between UAVs"""
        total_interference = 0.0
        
        for i, uav_i in self.uavs.items():
            for j, uav_j in self.uavs.items():
                if i < j:  # Avoid double counting
                    distance = np.linalg.norm(uav_i.position - uav_j.position)
                    interference = (uav_i.current_power * uav_j.current_power) / (distance ** 2 + 1)
                    total_interference += interference
        
        # Normalize by maximum possible interference
        max_interference = (self.num_uavs * (self.num_uavs - 1) / 2) * (10.0 ** 2) / (1)
        return total_interference / max_interference

    def _print_statistics(self):
        """Print current statistics of the environment"""

        print("Environment Initialization Parameters:")
        print(f"  Number of UAVs: {self.num_uavs}")
        print(f"  Number of UEs: {self.num_ues}")
        print(f"  Service Area: {self.service_area} meters")
        print(f"  Height Range: {self.height_range} meters")
        print(f"  Number of DAs per UAV: {self.num_das_per_slice}")
        print(f"  Long-term period T_L: {self.T_L} seconds")
        print(f"  Short-term period T_S: {self.T_S} seconds")
        print(f"  QoS Profiles: {self.qos_profiles}")
        print(f"  Slice Weights: {self.slice_weights}")
        print(f"  Noise Power: {self.noise_power} Watts")
        print(f"  Path Loss Exponent: {self.path_loss_exponent}")
        print(f"  Carrier Frequency: {self.carrier_frequency} Hz")
        print(f"  RB Bandwidth: {self.rb_bandwidth} Hz")
        print(f"  Total Bandwidth per UAV: {self.total_bandwidth} Hz")
        print(f"  Total Resource Blocks: {self.total_rbs}")
        print(f"  UE Arrival Rate: {self.ue_arrival_rate} per second")
        print(f"  UE Departure Rate: {self.ue_departure_rate} per minute")
        print(f"  UE dynamics Params: {self.ue_dynamics}")
        print(f"  Max UEs Allowed: {self.max_ues}")

    def render(self, mode='2d'):
        """Visualize the current state of the environment"""
        fig = plt.figure(figsize=(12, 10))
        
        if mode == '2d':
            ax = fig.add_subplot(111)
            
            # Plot UAVs
            for uav in self.uavs.values():
                ax.scatter(uav.position[0], uav.position[1], 
                          s=200, marker='^', c='red', 
                          label=f'UAV {uav.id}' if uav.id == 0 else '')
                
                # Show coverage area (simplified as circle)
                coverage_radius = 200 * np.sqrt(uav.current_power / 10.0)
                circle = plt.Circle((uav.position[0], uav.position[1]), 
                                  coverage_radius, fill=False, 
                                  edgecolor='red', alpha=0.3)
                ax.add_patch(circle)
            
            # Plot UEs by slice type
            colors = {1: 'blue', 2: 'green', 3: 'orange'}
            labels = {1: 'eMBB', 2: 'URLLC', 3: 'mMTC'}
            
            for slice_type in [1, 2, 3]:
                slice_ues = [ue for ue in self.ues.values() if ue.slice_type == slice_type]
                if slice_ues:
                    positions = np.array([ue.position[:2] for ue in slice_ues])
                    ax.scatter(positions[:, 0], positions[:, 1], 
                             c=colors[slice_type], s=30, alpha=0.6,
                             label=labels[slice_type])
            
            # Plot DA centers
            for da in self.demand_areas.values():
                if da.center_position is not None:
                    ax.scatter(da.center_position[0], da.center_position[1],
                             marker='x', s=100, c='black', alpha=0.5)
            
            ax.set_xlim(0, self.service_area[0])
            ax.set_ylim(0, self.service_area[1])
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title('UAV Network Coverage')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        elif mode == '3d':
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot UAVs in 3D
            for uav in self.uavs.values():
                ax.scatter(uav.position[0], uav.position[1], uav.position[2],
                          s=200, marker='^', c='red')
            
            # Plot UEs
            for slice_type in [1, 2, 3]:
                slice_ues = [ue for ue in self.ues.values() if ue.slice_type == slice_type]
                if slice_ues:
                    positions = np.array([ue.position for ue in slice_ues])
                    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c=colors[slice_type], s=30, alpha=0.6)
            
            ax.set_xlim(0, self.service_area[0])
            ax.set_ylim(0, self.service_area[1])
            ax.set_zlim(0, self.height_range[1])
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Height (m)')
            ax.set_title('3D UAV Network Configuration')
        
        plt.tight_layout()
        plt.show()