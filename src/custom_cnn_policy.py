import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DQNCnn(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        in_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512), 
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
            )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

AVAILABLE_EXTRACTORS = {
    "DQNCnn": DQNCnn,
    # Add other custom extractors here if needed:
    # "AnotherExtractor": AnotherExtractorClass,
}


def create_custom_policy(config):
    # --- Build policy_kwargs dynamically from config ---
    policy_kwargs = None  # Default: use SB3's internal feature extractor
    if "policy_kwargs" in config.get("algorithm", {}):
        print("Custom policy_kwargs found in config.")
        custom_kwargs_config = config["algorithm"]["policy_kwargs"]
        policy_kwargs = {}  # Initialize the dictionary

        # Feature Extractor Class
        if "features_extractor_class_name" in custom_kwargs_config:
            extractor_name = custom_kwargs_config["features_extractor_class_name"]
            if extractor_name in AVAILABLE_EXTRACTORS:
                policy_kwargs["features_extractor_class"] = AVAILABLE_EXTRACTORS[
                    extractor_name
                ]
                print(f"  Using custom features_extractor_class: {extractor_name}")
            else:
                print(
                    f"  Error: Unknown features_extractor_class_name '{extractor_name}'. Available: {list(AVAILABLE_EXTRACTORS.keys())}"
                )
                print("  Falling back to default feature extractor.")
                policy_kwargs = None  # Reset if class is invalid
        else:
            print(
                "  Warning: 'policy_kwargs' section found but 'features_extractor_class_name' is missing."
            )

        # Feature Extractor Kwargs (only if class was set successfully)
        if (
            policy_kwargs is not None
            and "features_extractor_kwargs" in custom_kwargs_config
        ):
            policy_kwargs["features_extractor_kwargs"] = custom_kwargs_config[
                "features_extractor_kwargs"
            ]
            print(
                f"  Using custom features_extractor_kwargs: {policy_kwargs['features_extractor_kwargs']}"
            )

    else:
        print("No custom policy_kwargs in config. Using default feature extractor.")
    
    return policy_kwargs