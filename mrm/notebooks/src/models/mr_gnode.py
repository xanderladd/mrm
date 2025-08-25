"""MR-gNODE with Dynamic Communication - Minimal changes for variable dimensions"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MRgnODE_DynamicComm(nn.Module):
    def __init__(self, N=4, input_dim=2, tau_regions=0.01, hidden_dim=64, tau_comm=0.05, 
                 num_regions=2, comm_dim=16, output_dim=2, comm_penalty=0.001,
                 region_dims=None, input_dims=None, output_dims=None, hidden_state_init='xavier'):
        super().__init__()
        
        # Handle variable dimensions
        if region_dims is None:
            self.region_dims = [N // num_regions] * num_regions
            self.region_dim = N // num_regions  # For backward compat
        else:
            self.region_dims = region_dims
            self.region_dim = region_dims[0] if len(set(region_dims)) == 1 else None
        
        if input_dims is None:
            self.input_dims = [input_dim] * num_regions
        else:
            self.input_dims = input_dims
            
        if output_dims is None:
            self.output_dims = [output_dim] * num_regions
        else:
            self.output_dims = output_dims
        
        self.num_regions = num_regions
        self.comm_dim = comm_dim
        self.tau_regions = tau_regions
        self.tau_comm = tau_comm
        self.output_dim = output_dim  # Keep for backward compat
        self.comm_penalty = comm_penalty
        self.hidden_state_init = hidden_state_init
        self.dt  = 0
        
        # Total state: regions + communication channels
        self.num_comm_channels = num_regions * (num_regions - 1)  # Bidirectional
        self.N_regions = sum(self.region_dims)
        self.N_total = self.N_regions + self.num_comm_channels * comm_dim
        
        # Region dynamics networks - one per region now for variable dims
        self.region_flow_nets = nn.ModuleList()
        self.region_gate_nets = nn.ModuleList()
        
        for i in range(num_regions):
            region_input_dim = self.region_dims[i] + self.input_dims[i]
            
            self.region_flow_nets.append(nn.Sequential(
                nn.Linear(region_input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.region_dims[i])
            ))
            
            self.region_gate_nets.append(nn.Sequential(
                nn.Linear(region_input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.region_dims[i]),
                nn.Sigmoid()
            ))
        
        # Communication dynamics networks - need to handle variable source dims
        self.comm_flow_nets = nn.ModuleList()
        self.comm_gate_nets = nn.ModuleList()
        
        for source_idx in range(num_regions):
            for target_idx in range(num_regions):
                if source_idx != target_idx:
                    comm_input_dim = self.region_dims[source_idx] + comm_dim
                    
                    self.comm_flow_nets.append(nn.Sequential(
                        nn.Linear(comm_input_dim, hidden_dim//2),
                        nn.Tanh(),
                        nn.Linear(hidden_dim//2, hidden_dim//2),
                        nn.Tanh(),
                        nn.Linear(hidden_dim//2, comm_dim)
                    ))
                    
                    self.comm_gate_nets.append(nn.Sequential(
                        nn.Linear(comm_input_dim, hidden_dim//2),
                        nn.Tanh(),
                        nn.Linear(hidden_dim//2, comm_dim),
                        nn.Sigmoid()
                    ))
        
        # Communication-to-region coupling - one per region for variable dims
        self.comm_to_region = nn.ModuleList([
            nn.Sequential(
                nn.Linear(comm_dim, self.region_dims[i]),
                nn.Tanh(),
                nn.Linear(self.region_dims[i], self.region_dims[i])
            ) for i in range(num_regions)
        ])
        
        # Separate readout layers for each region with variable dims
        self.region_readouts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.region_dims[i], hidden_dim//2),
                nn.Tanh(),
                nn.Linear(hidden_dim//2, self.output_dims[i])
            ) for i in range(num_regions)
        ])
        
        # Learnable asymmetric weights for each communication direction
        self.comm_signs = nn.Parameter(torch.ones(self.num_comm_channels))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def partition_state(self, h):
        """Partition hidden state into regions and communication channels"""
        batch_size = h.shape[0]
        h_regions_flat = h[:, :self.N_regions]  # [batch, N_regions]
        h_comm = h[:, self.N_regions:]          # [batch, num_comm_channels * comm_dim]
        
        # Split regions according to variable dimensions
        h_regions = []
        start_idx = 0
        for dim in self.region_dims:
            end_idx = start_idx + dim
            h_regions.append(h_regions_flat[:, start_idx:end_idx])
            start_idx = end_idx
        
        # Stack regions: [batch, num_regions, region_dim_i]
        # Note: can't stack if dims are different, keep as list
        
        # Reshape communication: [batch, num_comm_channels, comm_dim]
        h_comm = h_comm.view(batch_size, self.num_comm_channels, self.comm_dim)
        
        return h_regions, h_comm
    
    def get_comm_channel_idx(self, source_region, target_region):
        """Get communication channel index for source->target"""
        if source_region == target_region:
            return None
        
        # Channel indexing: each source region has (num_regions-1) outgoing channels
        channel_idx = source_region * (self.num_regions - 1)
        target_offset = target_region if target_region < source_region else target_region - 1
        return channel_idx + target_offset
    
    def compute_region_dynamics(self, h_regions, x, h_comm):
        """Compute region dynamics with communication input"""
        batch_size = h_regions[0].shape[0]
        flow_regions = []
        gate_regions = []
        
        for region_idx in range(self.num_regions):
            # Get region state and input
            h_region = h_regions[region_idx]  # [batch, region_dim_i]
            x_region = x[:, region_idx, :self.input_dims[region_idx]]  # [batch, input_dim_i]
            
            # Compute communication input to this region
            comm_input = torch.zeros_like(h_region)
            for source_region in range(self.num_regions):
                if source_region != region_idx:
                    channel_idx = self.get_comm_channel_idx(source_region, region_idx)
                    if channel_idx is not None:
                        comm_state = h_comm[:, channel_idx, :]  # [batch, comm_dim]
                        comm_effect = self.comm_to_region[region_idx](comm_state)
                        # Apply learnable sign for asymmetric effects
                        comm_input += self.comm_signs[channel_idx] * comm_effect
            
            # Region dynamics using region-specific networks
            hx_region = torch.cat([h_region + comm_input, x_region], dim=-1)
            flow_region = self.region_flow_nets[region_idx](hx_region)
            gate_region = self.region_gate_nets[region_idx](hx_region)
            
            flow_regions.append(flow_region)
            gate_regions.append(gate_region)
        
        # Concatenate all regions
        flow = torch.cat(flow_regions, dim=-1)  # [batch, sum(region_dims)]
        gate = torch.cat(gate_regions, dim=-1)
        h_regions_flat = torch.cat(h_regions, dim=-1)
        
        h_regions_dot = gate * (-h_regions_flat + flow) / self.tau_regions
        return h_regions_dot
    
    def compute_comm_dynamics(self, h_regions, h_comm):
        """Compute communication channel dynamics"""
        batch_size = h_regions[0].shape[0]
        comm_flows = []
        comm_gates = []
        
        net_idx = 0
        for channel_idx in range(self.num_comm_channels):
            # Determine source region for this channel
            source_region = channel_idx // (self.num_regions - 1)
            
            # Get source region state and current communication state
            h_source = h_regions[source_region]  # [batch, region_dim_i]
            h_comm_current = h_comm[:, channel_idx, :]  # [batch, comm_dim]
            
            # Communication dynamics using appropriate network
            comm_input = torch.cat([h_source, h_comm_current], dim=-1)
            flow_comm = self.comm_flow_nets[net_idx](comm_input)
            gate_comm = self.comm_gate_nets[net_idx](comm_input)
            net_idx += 1
            
            comm_flows.append(flow_comm)
            comm_gates.append(gate_comm)
        
        flow = torch.stack(comm_flows, dim=1)  # [batch, num_comm_channels, comm_dim]
        gate = torch.stack(comm_gates, dim=1)
        
        h_comm_dot = gate * (-h_comm + flow) / self.tau_comm
        return h_comm_dot.view(batch_size, -1)  # Flatten for concatenation
    
    def get_outputs(self, h):
        """Return region-specific readouts as [batch, regions, output_dim_i]"""
        h_regions, _ = self.partition_state(h)
        batch_size = h_regions[0].shape[0]
        
        # Apply separate readout for each region
        region_outputs = []
        for region_idx in range(self.num_regions):
            h_region = h_regions[region_idx]  # [batch, region_dim_i]
            region_output = self.region_readouts[region_idx](h_region)  # [batch, output_dim_i]
            region_outputs.append(region_output)
        
        # Stack if all outputs have same dim, otherwise keep as list
        if len(set(self.output_dims)) == 1:
            outputs = torch.stack(region_outputs, dim=1)  # [batch, num_regions, output_dim]
        else:
            outputs = region_outputs  # List of [batch, output_dim_i]
        
        return outputs
        
    def forward(self, h, x, timestep=0):
        """
        h: [batch, N_total] - regions + communication channels
        x: [batch, num_regions, max(input_dims)] - region-specific inputs  
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
        
    def predict(self, data, region_ids=None):
        """Predict using MR-GNODE reconstruction
        
        Args:
            data: Either single tensor [trials, time, features] or list of tensors
            region_ids: List specifying region ordering if data is a list
            
        Returns:
            Predictions in same format as input
        """
        import torch
        
        self.eval()
        device = next(self.parameters()).device
        if not self.dt:
            self.dt = 0.01
        if isinstance(data, dict):
            # Convert dict to list maintaining order
            region_keys = sorted([k for k in data.keys() if k.startswith('region_')])
            data_list = [data[k] for k in region_keys]
            predictions = self.predict(data_list, region_ids=region_keys)
            # Convert back to dict format
            return {region_keys[i]: pred for i, pred in enumerate(predictions)}
            
        # Handle different input formats
        if isinstance(data, list):
            if region_ids is None:
                raise ValueError("region_ids must be provided when data is a list")
            
            # List of variable-dim tensors
            all_data = []
            for i, d in enumerate(data):
                if not isinstance(d, torch.Tensor):
                    d = torch.tensor(d, dtype=torch.float32)
                all_data.append(d.to(device))
            
            n_trials = all_data[0].shape[0]
            seq_len = all_data[0].shape[1]
            
        else:
            # Single tensor input
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            combined_data = data.to(device)
            n_trials, seq_len, _ = combined_data.shape
            
            # Split into regions if concatenated
            all_data = []
            start_idx = 0
            for dim in self.region_dims:
                end_idx = start_idx + dim
                all_data.append(combined_data[..., start_idx:end_idx])
                start_idx = end_idx
        
        all_preds = []
        
        with torch.no_grad():
            for trial_idx in range(n_trials):
                # Get trial data for each region
                trial_data = [d[trial_idx:trial_idx+1] for d in all_data]  # List of [1, time, dim_i]
                
                # Pad to max input dim for forward pass
                max_input_dim = max(self.input_dims)
                padded_inputs = []
                for i, td in enumerate(trial_data):
                    if td.shape[-1] < max_input_dim:
                        padding = torch.zeros(1, seq_len, max_input_dim - td.shape[-1], device=device)
                        padded = torch.cat([td, padding], dim=-1)
                    else:
                        padded = td[..., :max_input_dim]
                    padded_inputs.append(padded)
                
                batch_input_reshaped = torch.stack(padded_inputs, dim=2).squeeze(0)  # [time, regions, max_input_dim]
                
                # Initialize state
                h = torch.zeros(size=(1, self.N_total), device=device)
                if self.hidden_state_init:
                    torch.nn.init.xavier_uniform_(h)
                outputs = []
                
                for t in range(seq_len):
                    h_dot, _ = self(h, batch_input_reshaped[t:t+1], timestep=t)
                    h = h + self.dt * h_dot
                    step_output = self.get_outputs(h)
                    outputs.append(step_output)
                
                # Stack outputs
                if isinstance(outputs[0], list):
                    # Variable output dims
                    trial_outputs = []
                    for r in range(self.num_regions):
                        region_outputs = torch.stack([out[r] for out in outputs], dim=1)
                        trial_outputs.append(region_outputs.squeeze(0).cpu().numpy())
                    all_preds.append(trial_outputs)
                else:
                    outputs = torch.stack(outputs, dim=1)  # [1, time, regions, output_dim]
                    
                    if self.region_dim is not None:
                        # Equal dims - reshape to concatenated format
                        outputs = outputs.view(1, seq_len, -1)
                    
                    all_preds.append(outputs.squeeze(0).cpu().numpy())
        
        # Format output
        if isinstance(all_preds[0], list):
            # Variable dims - reorganize by region
            predictions = []
            for r in range(self.num_regions):
                region_preds = np.array([pred[r] for pred in all_preds])
                predictions.append(region_preds)
            return predictions
        else:
            predictions = np.array(all_preds)
            
            # Return in same format as input
            if isinstance(data, list):
                # Split back into regions
                result = []
                start_idx = 0
                for dim in self.output_dims:
                    end_idx = start_idx + dim
                    result.append(predictions[..., start_idx:end_idx])
                    start_idx = end_idx
                return result
            else:
                return predictions

    def fit(self, train_data, targets={}, region_info=None, **training_params):
        """Fit MR-GNODE model using provided trajectory data"""
        import torch.optim as optim
        import torch.nn as nn
        from tqdm import tqdm
        
        optimizer = optim.AdamW(self.parameters(), lr=training_params['lr'], weight_decay=1e-2)
        
        n_epochs = training_params['epochs']
        batch_size = training_params['batch_size']
        dt = training_params.get('dt', 0.01)
        self.dt= dt
        comm_penalty = training_params.get('comm_penalty', 0.01)
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Prepare data from train_data
        region_inputs = []
        region_targets = []
        
        for i in range(self.num_regions):
            key = f'region_{i+1}'
            inp = np.array(train_data[key])
            region_inputs.append(torch.tensor(inp, dtype=torch.float32).to(device))
            
            if len(targets.keys()) and key in targets:
                tgt = np.array(targets[key])
                region_targets.append(torch.tensor(tgt, dtype=torch.float32).to(device))
            else:
                region_targets.append(region_inputs[-1])
        
        print(f"  Training data shapes: {[r.shape for r in region_inputs]}")
        
        # Setup scheduler
        n_trials = region_inputs[0].shape[0]
        steps_per_epoch = n_trials // batch_size + 1
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=training_params['lr']*5, 
                                                epochs=n_epochs, steps_per_epoch=steps_per_epoch)
        
        losses, train_mse_history, train_r2_history = [], [], []
        
        def communication_loss(model, outputs, targets, comm_norm, comm_penalty=0.01):
            """Simplified loss function with communication penalty"""
            mse_loss = nn.MSELoss()(outputs, targets)
            comm_cost = comm_penalty * comm_norm
            total_loss = mse_loss + comm_cost
            return total_loss, {'mse': mse_loss.item(), 'comm_cost': comm_cost.item()}
        
        for epoch in tqdm(range(n_epochs), desc="Training MR-GNODE"):
            epoch_loss = 0
            all_preds, all_targets = [], []
            
            # Shuffle indices
            indices = torch.randperm(n_trials)
            
            for i in range(0, n_trials, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_size_actual = len(batch_indices)
                
                # Get batch for each region
                batch_regions = [r[batch_indices] for r in region_inputs]
                batch_targets_list = [t[batch_indices] for t in region_targets]
                
                # Concatenate for loss computation
                batch_input = torch.cat(batch_regions, dim=-1)
                batch_target = torch.cat(batch_targets_list, dim=-1)
                
                optimizer.zero_grad()
                
                seq_len = batch_input.shape[1]
                
                # Format input for MR-GNODE: [batch, time, regions, max_input_dim]
                # Pad to max input dim
                max_input_dim = max(self.input_dims)
                batch_input_padded = []
                for r_idx, batch_r in enumerate(batch_regions):
                    if self.input_dims[r_idx] < max_input_dim:
                        padding = torch.zeros(batch_size_actual, seq_len, 
                                             max_input_dim - self.input_dims[r_idx], device=device)
                        padded = torch.cat([batch_r, padding], dim=-1)
                    else:
                        padded = batch_r[..., :max_input_dim]
                    batch_input_padded.append(padded)
                
                batch_input_reshaped = torch.stack(batch_input_padded, dim=2)  # [batch, time, regions, max_input_dim]
                
                # Initialize hidden state
                h = torch.zeros(size=(batch_size_actual, self.N_total), device=device)
                if self.hidden_state_init:
                    torch.nn.init.xavier_uniform_(h)
                outputs, total_comm = [], 0
                
                # Forward pass through time
                for t in range(seq_len):
                    h_dot, comm_norm = self(h, batch_input_reshaped[:, t], timestep=t)
                    h = h + self.dt * h_dot
                    total_comm += comm_norm
                    step_output = self.get_outputs(h)
                    outputs.append(step_output)
                
                # Process outputs based on type
                if isinstance(outputs[0], list):
                    # Variable output dims
                    outputs_concat = []
                    for t_idx in range(len(outputs)):
                        t_outputs = torch.cat(outputs[t_idx], dim=-1)
                        outputs_concat.append(t_outputs)
                    outputs = torch.stack(outputs_concat, dim=1)
                else:
                    outputs = torch.stack(outputs, dim=1)  # [batch, time, regions, output_dim]
                    
                    # Reshape outputs to match target format
                    if outputs.shape[2] > 1:
                        outputs = outputs.view(batch_size_actual, seq_len, -1)
                    else:
                        outputs = outputs.squeeze(2)
                
                # Compute loss
                loss, loss_components = communication_loss(self, outputs, batch_target, total_comm, comm_penalty)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                all_preds.append(outputs.detach().cpu().numpy())
                all_targets.append(batch_target.detach().cpu().numpy())
            
            losses.append(epoch_loss / (n_trials // batch_size))
            
            # Compute metrics
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            train_mse = float(np.mean((all_preds - all_targets) ** 2))
            
            from utils import compute_r2
            train_r2 = float(compute_r2(all_preds.flatten(), all_targets.flatten()))
            train_mse_history.append(train_mse)
            train_r2_history.append(train_r2)
            
            if epoch % (n_epochs // 10) == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}, MSE: {train_mse:.4f}, R²: {train_r2:.3f}")
        
        self.eval()
        
        return {
            'losses': losses,
            'train_mse': train_mse_history,
            'train_r2': train_r2_history,
            'final_mse': train_mse_history[-1],
            'final_r2': train_r2_history[-1],
            'model_type': 'mr_gnode'
        }        

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