"""Multi-Region RNN Model - Exact implementation from original script"""
import torch
import torch.nn as nn

"""Multi-Region RNN Model - Simple & Concise"""
import torch
import torch.nn as nn

class MultiRegionRNN(nn.Module):
    def __init__(self, evidence_hidden=64, motor_hidden=64, **kwargs):
        super().__init__()
        
        # Region 1: Evidence Integrator
        self.evidence_rnn = nn.LSTM(2, evidence_hidden, batch_first=True)
        self.evidence_output = nn.Linear(evidence_hidden, 1)
        
        # Region 2: Motor Controller  
        self.motor_rnn = nn.LSTM(3, motor_hidden, batch_first=True)
        self.motor_output = nn.Linear(motor_hidden, 2)
        
        # Communication layer
        self.communication = nn.Linear(evidence_hidden, 1)
        
        self.evidence_hidden_size = evidence_hidden
        self.motor_hidden_size = motor_hidden
        
    def forward(self, evidence_input):
        batch_size, seq_len, _ = evidence_input.shape
        device = evidence_input.device
        
        # Initialize states
        motor_position = torch.zeros(batch_size, seq_len, 2, device=device)
        evidence_hidden = (torch.zeros(1, batch_size, self.evidence_hidden_size, device=device),
                          torch.zeros(1, batch_size, self.evidence_hidden_size, device=device))
        motor_hidden = (torch.rand(1, batch_size, self.motor_hidden_size, device=device),
                       torch.rand(1, batch_size, self.motor_hidden_size, device=device))
        
        # Storage
        evidence_outputs, motor_outputs, communications = [], [], []
        evidence_states, motor_states = [], []
        
        for t in range(seq_len):
            # Evidence region
            evidence_out, evidence_hidden = self.evidence_rnn(evidence_input[:, t:t+1, :], evidence_hidden)
            evidence_outputs.append(self.evidence_output(evidence_out.squeeze(1)))
            evidence_states.append(evidence_hidden[0].squeeze(0))
            
            # Communication
            comm_signal = self.communication(evidence_out.squeeze(1))
            communications.append(comm_signal)
            
            # Motor region
            motor_input = torch.cat([comm_signal, motor_position[:, t, :]], dim=1).unsqueeze(1)
            motor_out, motor_hidden = self.motor_rnn(motor_input, motor_hidden)
            velocity = torch.tanh(self.motor_output(motor_out.squeeze(1)))
            motor_outputs.append(velocity)
            motor_states.append(motor_hidden[0].squeeze(0))
            
            # Update position
            if t < seq_len - 1:
                motor_position[:, t+1, :] = motor_position[:, t, :] + velocity * 0.1
        
        return {
            'evidence_integration': torch.stack(evidence_outputs, dim=1),
            'motor_velocity': torch.stack(motor_outputs, dim=1),
            'communication': torch.stack(communications, dim=1),
            'motor_position': motor_position,
            'evidence_states': torch.stack(evidence_states, dim=1),  # Enables trajectory analysis
            'motor_states': torch.stack(motor_states, dim=1)         # Enables trajectory analysis
        }