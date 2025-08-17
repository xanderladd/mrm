"""Multi-Region Gated Neural ODE"""
import torch
import torch.nn as nn
import numpy as np


class MRgnODE_DynamicComm(nn.Module):
    def __init__(self, N=4, input_dim=2, tau_regions=0.01, hidden_dim=64, tau_comm=0.05, 
                 num_regions=2, comm_dim=16, output_dim=2, comm_penalty=0.001):
        super().__init__()
        self.num_regions = num_regions
        self.region_dim = N // num_regions
        self.comm_dim = comm_dim
        self.tau_regions = tau_regions
        self.tau_comm = tau_comm
        self.output_dim = output_dim
        self.comm_penalty = comm_penalty
        # Total state: regions + communication channels
        self.num_comm_channels = num_regions * (num_regions - 1)  # Bidirectional
        self.N_regions = N
        self.N_total = N + self.num_comm_channels * comm_dim
        
        region_input_dim = self.region_dim + input_dim
        
        # Region dynamics networks
        self.region_flow_net = nn.Sequential(
            nn.Linear(region_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.region_dim)
        )
        
        self.region_gate_net = nn.Sequential(
            nn.Linear(region_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.region_dim),
            nn.Sigmoid()
        )
        
        # Communication dynamics networks
        comm_input_dim = self.region_dim + comm_dim  # Source region + current comm state
        
        self.comm_flow_net = nn.Sequential(
            nn.Linear(comm_input_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, comm_dim)
        )
        
        self.comm_gate_net = nn.Sequential(
            nn.Linear(comm_input_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, comm_dim),
            nn.Sigmoid()
        )
        
        # Communication-to-region coupling
        self.comm_to_region = nn.Sequential(
            nn.Linear(comm_dim, self.region_dim),
            nn.Tanh(),
            nn.Linear(self.region_dim, self.region_dim)
        )
        
        # Separate readout layers for each region
        self.region_readouts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.region_dim, hidden_dim//2),
                nn.Tanh(),
                nn.Linear(hidden_dim//2, output_dim)
            ) for _ in range(num_regions)
        ])
        
        # Learnable asymmetric weights for each communication direction
        self.comm_signs = nn.Parameter(torch.ones(self.num_comm_channels))  # Can learn positive/negative
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def partition_state(self, h):
        """Partition hidden state into regions and communication channels"""
        batch_size = h.shape[0]
        h_regions = h[:, :self.N_regions]  # [batch, N_regions]
        h_comm = h[:, self.N_regions:]     # [batch, num_comm_channels * comm_dim]
        
        # Reshape regions: [batch, num_regions, region_dim]
        h_regions = h_regions.view(batch_size, self.num_regions, self.region_dim)
        
        # Reshape communication: [batch, num_comm_channels, comm_dim]
        h_comm = h_comm.view(batch_size, self.num_comm_channels, self.comm_dim)
        
        return h_regions, h_comm
    
    def get_comm_channel_idx(self, source_region, target_region):
        """Get communication channel index for source->target"""
        # Map (source, target) pairs to channel indices
        # For 2 regions: (0,1)->0, (1,0)->1
        if source_region == target_region:
            return None
        
        # Channel indexing: each source region has (num_regions-1) outgoing channels
        channel_idx = source_region * (self.num_regions - 1)
        target_offset = target_region if target_region < source_region else target_region - 1
        return channel_idx + target_offset
    
    def compute_region_dynamics(self, h_regions, x, h_comm):
        """Compute region dynamics with communication input"""
        batch_size = h_regions.shape[0]
        flow_regions = []
        gate_regions = []
        
        for region_idx in range(self.num_regions):
            # Get region state and input
            h_region = h_regions[:, region_idx, :]  # [batch, region_dim]
            x_region = x[:, region_idx, :]  # [batch, input_dim]
            
            # Compute communication input to this region
            comm_input = torch.zeros_like(h_region)
            for source_region in range(self.num_regions):
                if source_region != region_idx:
                    channel_idx = self.get_comm_channel_idx(source_region, region_idx)
                    if channel_idx is not None:
                        comm_state = h_comm[:, channel_idx, :]  # [batch, comm_dim]
                        comm_effect = self.comm_to_region(comm_state)
                        # Apply learnable sign for asymmetric effects
                        comm_input += self.comm_signs[channel_idx] * comm_effect
            
            # Region dynamics
            hx_region = torch.cat([h_region + comm_input, x_region], dim=-1)
            flow_region = self.region_flow_net(hx_region)
            gate_region = self.region_gate_net(hx_region)
            
            flow_regions.append(flow_region)
            gate_regions.append(gate_region)
        
        flow = torch.stack(flow_regions, dim=1).view(batch_size, -1)
        gate = torch.stack(gate_regions, dim=1).view(batch_size, -1)
        h_regions_flat = h_regions.view(batch_size, -1)
        
        h_regions_dot = gate * (-h_regions_flat + flow) / self.tau_regions
        return h_regions_dot
    
    def compute_comm_dynamics(self, h_regions, h_comm):
        """Compute communication channel dynamics"""
        batch_size = h_regions.shape[0]
        comm_flows = []
        comm_gates = []
        
        for channel_idx in range(self.num_comm_channels):
            # Determine source region for this channel
            source_region = channel_idx // (self.num_regions - 1)
            
            # Get source region state and current communication state
            h_source = h_regions[:, source_region, :]  # [batch, region_dim]
            h_comm_current = h_comm[:, channel_idx, :]  # [batch, comm_dim]
            
            # Communication dynamics
            comm_input = torch.cat([h_source, h_comm_current], dim=-1)
            flow_comm = self.comm_flow_net(comm_input)
            gate_comm = self.comm_gate_net(comm_input)
            
            comm_flows.append(flow_comm)
            comm_gates.append(gate_comm)
        
        flow = torch.stack(comm_flows, dim=1)  # [batch, num_comm_channels, comm_dim]
        gate = torch.stack(comm_gates, dim=1)  # [batch, num_comm_channels, comm_dim]
        
        h_comm_dot = gate * (-h_comm + flow) / self.tau_comm
        return h_comm_dot.view(batch_size, -1)  # Flatten for concatenation
    
    def get_outputs(self, h):
        """Return region-specific readouts as [batch, regions, output_dim]"""
        h_regions, _ = self.partition_state(h)
        batch_size = h_regions.shape[0]
        
        # Apply separate readout for each region
        region_outputs = []
        for region_idx in range(self.num_regions):
            h_region = h_regions[:, region_idx, :]  # [batch, region_dim]
            region_output = self.region_readouts[region_idx](h_region)  # [batch, output_dim]
            region_outputs.append(region_output)
        
        # Stack to get [batch, num_regions, output_dim]
        outputs = torch.stack(region_outputs, dim=1)
        return outputs
        
    def forward(self, h, x, timestep=0):
        """
        h: [batch, N_total] - regions + communication channels
        x: [batch, num_regions, input_dim] - region-specific inputs  
        """
        # Partition state
        h_regions, h_comm = self.partition_state(h)
        
        # Compute dynamics for each partition
        h_regions_dot = self.compute_region_dynamics(h_regions, x, h_comm)
        h_comm_dot = self.compute_comm_dynamics(h_regions, h_comm)
        
        # Concatenate dynamics
        h_dot = torch.cat([h_regions_dot, h_comm_dot], dim=-1)
        
        # Return communication norms for tracking
        comm_norm = torch.norm(h_comm)
        
        return h_dot, comm_norm
        
def generate_mr_flip_flop_data(n_trials, task_type='square', combine_method='sum',
                               trial_length=100, n_regions=2, n_channels=2):
    """Generate multi-region flip-flop task data"""
    data = []
    
    for _ in range(n_trials):
        inputs = np.zeros((trial_length, n_regions, n_channels))
        targets = np.zeros((trial_length, n_regions, n_channels))
        
        if combine_method == 'sum':
            # Integrated sum task
            integrated = np.zeros(n_channels)
            
            for t in range(trial_length):
                # Random pulses
                if np.random.rand() < 0.1:  # 10% chance of pulse
                    for region in range(n_regions):
                        channel = np.random.randint(n_channels)
                        pulse = np.random.uniform(-1, 1)
                        inputs[t, region, channel] = pulse
                        integrated[channel] += pulse
                
                # All regions output the integrated sum
                for region in range(n_regions):
                    targets[t, region] = integrated.copy()
            
            # Normalize
            max_abs = np.max(np.abs(targets))
            if max_abs > 0:
                targets = targets / max_abs
        
        else:
            # Independent flip-flop per region
            for region in range(n_regions):
                current_output = np.zeros(n_channels)
                n_pulses = np.random.poisson(6)
                pulse_times = np.random.choice(trial_length, min(n_pulses, trial_length), replace=False)
                
                for t in range(trial_length):
                    if t in pulse_times:
                        if task_type == 'disk':
                            # Disk task - both channels get values
                            radius = np.random.uniform(1, 2)
                            angle = np.random.uniform(0, 2*np.pi)
                            pulse_vals = [radius * np.cos(angle), radius * np.sin(angle)]
                            inputs[t, region] = pulse_vals
                            current_output = pulse_vals
                        else:
                            # Square/rectangle task - one channel at a time
                            channel = np.random.randint(n_channels)
                            
                            if task_type == 'square':
                                pulse_val = np.random.uniform(-1, 1)
                            elif task_type == 'rectangle':
                                pulse_val = np.random.uniform(-2, 2) if channel == 0 else np.random.uniform(-1, 1)
                            else:  # fixed
                                pulse_val = np.random.choice([-1, 1])
                            
                            inputs[t, region, channel] = pulse_val
                            current_output[channel] = pulse_val
                    
                    targets[t, region] = current_output.copy()
        
        data.append((inputs, targets))
    
    return data