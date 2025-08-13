import gym
import numpy as np
import traci
import os
from gym import spaces
from sumolib import checkBinary
from config import ENV_CONFIG
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any

class TrafficSignalPhase:
    """Traffic signal phase constants and utilities"""
    
    # Phase definitions
    PHASE_NS_GREEN = 0
    PHASE_NS_YELLOW = 1
    PHASE_NSL_GREEN = 2
    PHASE_NSL_YELLOW = 3
    PHASE_EW_GREEN = 4
    PHASE_EW_YELLOW = 5
    PHASE_EWL_GREEN = 6
    PHASE_EWL_YELLOW = 7
    PHASE_WL_GREEN = 8
    PHASE_WL_YELLOW = 9
    PHASE_EL_GREEN = 10
    PHASE_EL_YELLOW = 11
    PHASE_SL_GREEN = 12
    PHASE_SL_YELLOW = 13
    PHASE_NL_GREEN = 14
    PHASE_NL_YELLOW = 15
    
    @classmethod
    def get_green_phase(cls, action_number: int) -> int:
        """Map action number to green phase"""
        phase_mapping = {
            0: cls.PHASE_NS_GREEN,
            1: cls.PHASE_NSL_GREEN,
            2: cls.PHASE_EW_GREEN,
            3: cls.PHASE_EWL_GREEN,
            4: cls.PHASE_WL_GREEN,
            5: cls.PHASE_EL_GREEN,
            6: cls.PHASE_SL_GREEN,
            7: cls.PHASE_NL_GREEN
        }
        return phase_mapping.get(action_number, cls.PHASE_NS_GREEN)
    
    @classmethod
    def get_yellow_phase(cls, action_number: int) -> int:
        """Map action number to yellow phase"""
        return action_number * 2 + 1

class NetworkTopologyManager:
    """Manages traffic network topology and adjacency relationships"""
    
    def __init__(self):
        self.intersection_ids: List[str] = []
        self.num_nodes: int = 0
        self.node_index: Dict[str, int] = {}
        self.incoming_all_edges: Dict[int, List[str]] = {}
        self.adjacency_matrix: Optional[np.ndarray] = None
    
    def initialize_network_topology(self) -> np.ndarray:
        """Extract and initialize network topology from SUMO"""
        self.intersection_ids = traci.trafficlight.getIDList()
        self.num_nodes = len(self.intersection_ids)
        self.node_index = {node: idx for idx, node in enumerate(self.intersection_ids)}
        
        self._extract_incoming_edges()
        self.adjacency_matrix = self._build_adjacency_matrix()
        
        return self.adjacency_matrix
    
    def _extract_incoming_edges(self) -> None:
        """Extract incoming edges for each intersection"""
        node_edges = {}
        for intersection_id in self.intersection_ids:
            incoming_edges = traci.junction.getIncomingEdges(intersection_id)[-4:]
            node_edges[intersection_id] = incoming_edges
        
        for node_id, edges in node_edges.items():
            index = self.node_index[node_id]
            self.incoming_all_edges[index] = edges
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix for graph neural networks"""
        adj = np.zeros((self.num_nodes, self.num_nodes))
        edges_ids = traci.edge.getIDList()
        
        for edge_id in edges_ids:
            from_intersection = traci.edge.getFromJunction(edge_id)
            to_intersection = traci.edge.getToJunction(edge_id)
            
            if (from_intersection in self.node_index and 
                to_intersection in self.node_index):
                idx_i = self.node_index[from_intersection]
                idx_j = self.node_index[to_intersection]
                adj[idx_i, idx_j] = 1
                adj[idx_j, idx_i] = 1
        
        return adj

class StateObserver:
    """Observes and processes traffic state information"""
    
    def __init__(self, topology_manager: NetworkTopologyManager):
        self.topology_manager = topology_manager
    
    def get_state_and_waiting_times(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract current traffic state and waiting times"""
        state = np.zeros((self.topology_manager.num_nodes, ENV_CONFIG['state_dim']))
        intersection_wait_times = np.zeros(self.topology_manager.num_nodes)

        for intersection_idx, incoming_roads in self.topology_manager.incoming_all_edges.items():
            for road_idx, road_id in enumerate(incoming_roads):
                straight_right_group = road_idx * 2  
                left_group = road_idx * 2 + 1  
                
                # Aggregate halting vehicles for straight/right lanes
                state[intersection_idx, straight_right_group] += (
                    traci.lane.getLastStepHaltingNumber(f"{road_id}_0") + 
                    traci.lane.getLastStepHaltingNumber(f"{road_id}_1")
                )
                # Aggregate halting vehicles for left turn lanes
                state[intersection_idx, left_group] += (
                    traci.lane.getLastStepHaltingNumber(f"{road_id}_2")
                )
                
                intersection_wait_times[intersection_idx] += traci.edge.getWaitingTime(road_id)

        return state[np.newaxis], intersection_wait_times
    
    def get_queue_lengths(self) -> np.ndarray:
        """Get queue lengths for all intersections"""
        intersection_queue_lengths = np.zeros(self.topology_manager.num_nodes)
        
        for intersection_id, incoming_edges_list in self.topology_manager.incoming_all_edges.items():
            queue_length = sum(
                traci.edge.getLastStepHaltingNumber(road_id) 
                for road_id in incoming_edges_list
            )
            intersection_queue_lengths[intersection_id] = queue_length
            
        return intersection_queue_lengths

class TrafficSignalController:
    """Controls traffic signal phases and timing"""
    
    def __init__(self, topology_manager: NetworkTopologyManager):
        self.topology_manager = topology_manager
        self.phase_manager = TrafficSignalPhase()
    
    def set_traffic_signals(self, old_action: np.ndarray, new_action: np.ndarray, 
                          current_step: int, simulation_times: int) -> None:
        """Set traffic signals based on actions"""
        action_changed = self._check_action_changes(old_action, new_action)
        is_not_first_step = (current_step != 0)
        
        for i in range(self.topology_manager.num_nodes):
            if is_not_first_step and action_changed[i]:
                self._set_yellow_phase(i, old_action.squeeze()[i])
            else:
                self._set_green_phase(i, new_action.squeeze()[i])
        
        self._simulate_traffic_flow(simulation_times)
    
    def _check_action_changes(self, old_action: np.ndarray, new_action: np.ndarray) -> List[bool]:
        """Check which actions have changed"""
        return [
            old_action.squeeze()[i] != new_action.squeeze()[i] 
            for i in range(self.topology_manager.num_nodes)
        ]
    
    def _set_yellow_phase(self, node_id: int, old_action: int) -> None:
        """Set traffic light to yellow phase"""
        signal_id = self._get_signal_id(node_id)
        yellow_phase_code = self.phase_manager.get_yellow_phase(old_action)
        traci.trafficlight.setPhase(signal_id, yellow_phase_code)
    
    def _set_green_phase(self, node_id: int, action_number: int) -> None:
        """Set traffic light to green phase"""
        signal_id = self._get_signal_id(node_id)
        green_phase_code = self.phase_manager.get_green_phase(action_number)
        traci.trafficlight.setPhase(signal_id, green_phase_code)
    
    def _get_signal_id(self, node_id: int) -> str:
        """Get signal ID from node ID"""
        node_trafficlight = {
            value: key for key, value in self.topology_manager.node_index.items()
        }
        return node_trafficlight.get(node_id)
    
    def _simulate_traffic_flow(self, simulation_times: int) -> None:
        """Simulate traffic flow for specified time steps"""
        for _ in range(simulation_times):
            traci.simulationStep()

class SUMOSimulationManager:
    """Manages SUMO simulation lifecycle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sumo_running = False
    
    def start_simulation(self) -> None:
        """Start SUMO simulation"""
        if self.config['gui']:
            sumo_binary = checkBinary("sumo-gui")
        else:
            sumo_binary = checkBinary("sumo")
            
        sumo_cmd = [
            sumo_binary, "-c", self.config['sumocfg_file'], 
            "--no-warnings", "--no-step-log"
        ]
        traci.start(sumo_cmd)
        self.sumo_running = True
    
    def close_simulation(self) -> None:
        """Close SUMO simulation"""
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

class TrafficEnv(gym.Env):
    """Traffic light control environment using reinforcement learning"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TrafficEnv, self).__init__()
        
        # Initialize configuration
        self.config = ENV_CONFIG
        
        # Initialize managers
        self.simulation_manager = SUMOSimulationManager(self.config)
        self.topology_manager = NetworkTopologyManager()
        self.state_observer = StateObserver(self.topology_manager)
        self.signal_controller = TrafficSignalController(self.topology_manager)
        
        # Initialize environment state
        self.current_step = 0
        self.old_action = None
        self.old_total_wait = None
        self.queue_length_history = []
        
        # Initialize spaces
        self._initialize_spaces()
        
        # Statistics storage
        self.reward_store = []
        self.queue_length_store = []
    
    def _initialize_spaces(self) -> None:
        """Initialize action and observation spaces"""
        # Temporary simulation to get network info
        self.simulation_manager.start_simulation()
        self.topology_manager.initialize_network_topology()
        self.simulation_manager.close_simulation()
        
        self.action_space = spaces.MultiDiscrete(
            [self.config['action_dim']] * self.topology_manager.num_nodes
        )
        
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.topology_manager.num_nodes, self.config['state_dim']),
            dtype=np.float32
        )
    
    @property
    def num_nodes(self) -> int:
        """Get number of traffic light nodes"""
        return self.topology_manager.num_nodes
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        if self.simulation_manager.sumo_running:
            self.simulation_manager.close_simulation()
     
        self.simulation_manager.start_simulation()
        
        state, waiting_times = self.state_observer.get_state_and_waiting_times()
        self.old_action = np.zeros((1, self.topology_manager.num_nodes, 1))
        self.old_total_wait = waiting_times
        self.current_step = 0
        self.queue_length_history = []
        
        return state.squeeze(0)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        """Execute action and return next state, reward, and info"""
        # Record queue length
        current_queue_lengths = self.state_observer.get_queue_lengths()
        self.queue_length_history.append(sum(current_queue_lengths))
        
        # Reshape action and execute
        action_reshaped = action.reshape(1, self.topology_manager.num_nodes, 1)
        self.signal_controller.set_traffic_signals(
            self.old_action, action_reshaped, self.current_step, self.config['times']
        )
        self.current_step += self.config['times']
        
        # Get new state and calculate reward
        new_state, new_waiting_times = self.state_observer.get_state_and_waiting_times()
        reward = [
            (self.old_total_wait[i] - new_waiting_times[i]) 
            for i in range(self.topology_manager.num_nodes)
        ]
        
        # Update state
        self.old_action = action_reshaped
        self.old_total_wait = new_waiting_times
        
        info = {
            'queue_length': sum(self.state_observer.get_queue_lengths()),
            'waiting_times': new_waiting_times,
            'step': self.current_step
        }
        
        return new_state.squeeze(0), reward, info
    
    def get_adj_matrix(self) -> np.ndarray:
        """Get adjacency matrix for graph neural networks"""
        return self.topology_manager.adjacency_matrix
    
    def save_episode_stats(self, reward: float, queue_length: float) -> None:
        """Save episode statistics"""
        self.reward_store.append(reward)
        self.queue_length_store.append(queue_length)
        
    def get_stats(self) -> Tuple[List[float], List[float]]:
        """Get accumulated statistics"""
        return self.reward_store, self.queue_length_store
    
    def render(self, mode: str = 'human') -> None:
        """Render environment (not implemented)"""
        pass
    
    def close(self) -> None:
        """Close environment and cleanup resources"""
        self.simulation_manager.close_simulation()