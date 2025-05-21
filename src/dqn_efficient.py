import torch.nn as nn
from gymnasium.spaces import Discrete, Dict
import torch
import torch.nn.functional as F

class LCALayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        lambda_=0.1,
        tau=10,
    ):
        """
        Locally Competitive Algorithm layer implementation

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (dictionary elements)
            kernel_size: Size of convolutional kernel
            stride: Stride of convolution
            padding: Padding of convolution
            lambda_: Sparsity penalty coefficient
            tau: Number of iterations for LCA dynamics
        """
        super().__init__()

        # Dictionary elements (filters)
        self.dictionary = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding
        self.lambda_ = lambda_
        self.tau = tau
        self.out_channels = out_channels

        # Normalize dictionary elements to have unit norm
        with torch.no_grad():
            norm = torch.norm(self.dictionary.view(out_channels, -1), dim=1)
            self.dictionary.div_(norm.view(out_channels, 1, 1, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        # Get input dimensions
        _, _, h, w = x.shape

        # Calculate output dimensions
        h_out = (h + 2 * self.padding - self.dictionary.shape[2]) // self.stride + 1
        w_out = (w + 2 * self.padding - self.dictionary.shape[3]) // self.stride + 1

        # Initialize membrane potentials u
        u = torch.zeros(batch_size, self.out_channels, h_out, w_out, device=x.device)

        # Initialize sparse activations a
        a = torch.zeros_like(u)

        # Compute initial driving input b (correlation of input with dictionary)
        b = F.conv2d(x, self.dictionary, stride=self.stride, padding=self.padding)

        # LCA dynamic iterations
        for _ in range(self.tau):
            # Compute inhibition term (competition between neurons)
            # We'll use a simplified version here
            inhibition = (
                F.conv2d(
                    F.conv_transpose2d(
                        a, self.dictionary, stride=self.stride, padding=self.padding
                    ),
                    self.dictionary,
                    stride=self.stride,
                    padding=self.padding,
                )
                - a
            )  # Subtract identity mapping

            # Update membrane potentials
            u = u + 0.1 * (b - u - inhibition)

            # Apply soft thresholding to get sparse activations
            a = F.relu(u - self.lambda_) - F.relu(-u - self.lambda_)

        return a

class QNetwork(nn.Module):
    def __init__(self, env, sensory_action_set=None):
        super().__init__()
        if sensory_action_set == None:
            if isinstance(env.single_action_space, Discrete):
                action_space_size = env.single_action_space.n
            else:
                action_space_size = env.single_action_space[
                    "motor_action"
                ].n
        else:
            action_space_size = len(sensory_action_set)

        self.network = nn.Sequential(
            LCALayer(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4,
                lambda_=0.1,
                tau=10,
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_size),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)