import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SeparatedVisionNetwork(nn.Module):
    """
    Réseau neuronal qui traite séparément les entrées fovéales, périphériques et les coordonnées.
    """
    def __init__(self, observation_space, motor_action_space, sensory_action_set):
        super().__init__()
        
        # Récupérer les dimensions des différentes entrées
        self.fovea_shape = observation_space.spaces["fovea"].shape
        self.peripheral_shape = observation_space.spaces["peripheral"].shape
        self.position_dim = observation_space.spaces["position"].shape[0]  # Généralement 2 (x, y)
        
        # Nombre d'actions possibles
        self.motor_action_dim = motor_action_space.n
        self.sensory_action_dim = len(sensory_action_set)

        # Réseau pour la vision fovéale (haute résolution)
        self.fovea_network = nn.Sequential(
            nn.Conv2d(self.fovea_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Réseau pour la vision périphérique (basse résolution)
        self.peripheral_network = nn.Sequential(
            nn.Conv2d(self.peripheral_shape[0], 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Réseau pour les coordonnées de position
        self.position_network = nn.Sequential(
            nn.Linear(self.position_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        """
        # Réseau pour la vision fovéale (haute résolution)
        self.fovea_network = nn.Sequential(
            nn.Conv2d(self.fovea_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Réseau pour la vision périphérique (basse résolution)
        self.peripheral_network = nn.Sequential(
            nn.Conv2d(self.peripheral_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Réseau pour les coordonnées de position
        self.position_network = nn.Sequential(
            nn.Linear(self.position_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        """
        
        # Calculer les tailles de sortie des réseaux
        fovea_out_size = self._get_conv_output_size(self.fovea_network, self.fovea_shape)
        peripheral_out_size = self._get_conv_output_size(self.peripheral_network, self.peripheral_shape)
        
        # Couche de fusion
        fusion_input_size = fovea_out_size + peripheral_out_size + 32  # 32 de position_network
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU()
        )
        
        # Têtes de décision pour les actions motrices et sensorielles
        self.motor_head = nn.Linear(256, self.motor_action_dim)
        self.sensory_head = nn.Linear(256, self.sensory_action_dim)
    
    def _get_conv_output_size(self, model, input_shape):
        """Calcule la taille de sortie d'un modèle convolutif"""
        with torch.no_grad():
            # Créer un tenseur d'entrée de la bonne taille
            x = torch.zeros(1, *input_shape)
            # Passer à travers le modèle
            x = model(x)
            # Retourner la taille aplatie
            return x.numel()
    
    def forward(self, observations):
        """
        Traitement séparé des différentes entrées puis fusion
        
        Args:
            observations: dictionnaire contenant 'fovea', 'peripheral' et 'position'
        """
        # Extraire les différentes composantes
        fovea = observations["fovea"]
        peripheral = observations["peripheral"]
        position = observations["position"]
        
        # Traiter chaque composante séparément
        fovea_features = self.fovea_network(fovea)
        peripheral_features = self.peripheral_network(peripheral)
        
        # S'assurer que position a la bonne forme [batch_size, position_dim]
        # Si position est [position_dim], ajouter une dimension de batch
        if position.dim() == 1:
            position = position.unsqueeze(0)  # Ajouter une dimension de batch
        
        position_features = self.position_network(position)
        
        # Vérifier que tous les tenseurs ont le même nombre de dimensions
        if fovea_features.dim() != position_features.dim():
            if fovea_features.dim() > position_features.dim():
                # Adapter position_features à la dimension de fovea_features
                position_features = position_features.unsqueeze(0)  # Ajouter une dimension de batch
            else:
                # Cas rare, mais par précaution
                position_features = position_features.squeeze(0)  # Enlever la dimension de batch
        
        # Concaténer les caractéristiques
        combined = torch.cat([fovea_features, peripheral_features, position_features], dim=1)
        
        # Fusionner les caractéristiques
        features = self.fusion_layer(combined)
        
        # Calculer les valeurs Q pour les actions motrices et sensorielles
        motor_q_values = self.motor_head(features)
        sensory_q_values = self.sensory_head(features)
        
        return motor_q_values, sensory_q_values
    

class SeparatedVisionFoveaNetwork(nn.Module):
    """
    Réseau neuronal qui traite séparément les entrées fovéales, périphériques et les coordonnées.
    """

    def __init__(self, observation_space, motor_action_space, sensory_action_set):
        super().__init__()

        # Récupérer les dimensions des différentes entrées
        self.fovea_shape = observation_space.spaces["fovea"].shape
        self.position_dim = observation_space.spaces["position"].shape[
            0
        ]  # Généralement 2 (x, y)

        # Nombre d'actions possibles
        self.motor_action_dim = motor_action_space.n
        self.sensory_action_dim = len(sensory_action_set)

        # Réseau pour la vision fovéale (haute résolution)
        self.fovea_network = nn.Sequential(
            nn.Conv2d(self.fovea_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Réseau pour la vision périphérique (basse résolution)

        # Réseau pour les coordonnées de position
        self.position_network = nn.Sequential(
            nn.Linear(self.position_dim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU()
        )

        """
        # Réseau pour la vision fovéale (haute résolution)
        self.fovea_network = nn.Sequential(
            nn.Conv2d(self.fovea_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Réseau pour la vision périphérique (basse résolution)
        self.peripheral_network = nn.Sequential(
            nn.Conv2d(self.peripheral_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Réseau pour les coordonnées de position
        self.position_network = nn.Sequential(
            nn.Linear(self.position_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        """

        # Calculer les tailles de sortie des réseaux
        fovea_out_size = self._get_conv_output_size(
            self.fovea_network, self.fovea_shape
        )

        # Couche de fusion
        fusion_input_size = (
            fovea_out_size + 32
        )  # 32 de position_network
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_input_size, 256), nn.ReLU())

        # Têtes de décision pour les actions motrices et sensorielles
        self.motor_head = nn.Linear(256, self.motor_action_dim)
        self.sensory_head = nn.Linear(256, self.sensory_action_dim)

    def _get_conv_output_size(self, model, input_shape):
        """Calcule la taille de sortie d'un modèle convolutif"""
        with torch.no_grad():
            # Créer un tenseur d'entrée de la bonne taille
            x = torch.zeros(1, *input_shape)
            # Passer à travers le modèle
            x = model(x)
            # Retourner la taille aplatie
            return x.numel()

    def forward(self, observations):
        """
        Traitement séparé des différentes entrées puis fusion

        Args:
            observations: dictionnaire contenant 'fovea', 'peripheral' et 'position'
        """
        # Extraire les différentes composantes
        fovea = observations["fovea"]
        position = observations["position"]

        # Traiter chaque composante séparément
        fovea_features = self.fovea_network(fovea)


        # S'assurer que position a la bonne forme [batch_size, position_dim]
        # Si position est [position_dim], ajouter une dimension de batch
        if position.dim() == 1:
            position = position.unsqueeze(0)  # Ajouter une dimension de batch

        position_features = self.position_network(position)

        # Vérifier que tous les tenseurs ont le même nombre de dimensions
        if fovea_features.dim() != position_features.dim():
            if fovea_features.dim() > position_features.dim():
                # Adapter position_features à la dimension de fovea_features
                position_features = position_features.unsqueeze(
                    0
                )  # Ajouter une dimension de batch
            else:
                # Cas rare, mais par précaution
                position_features = position_features.squeeze(
                    0
                )  # Enlever la dimension de batch

        # Concaténer les caractéristiques
        combined = torch.cat(
            [fovea_features, position_features], dim=1
        )

        # Fusionner les caractéristiques
        features = self.fusion_layer(combined)

        # Calculer les valeurs Q pour les actions motrices et sensorielles
        motor_q_values = self.motor_head(features)
        sensory_q_values = self.sensory_head(features)

        return motor_q_values, sensory_q_values
    

class SeparatedFoveaSFN(nn.Module):
    """
    State Forecasting Network adapté pour traiter des observations structurées.
    Ce réseau prédit l'action motrice à partir de l'observation actuelle et future.
    """

    def __init__(self, observation_space, motor_action_space):
        super().__init__()

        # Récupérer les dimensions des différentes entrées
        self.fovea_shape = observation_space.spaces["fovea"].shape
        self.position_dim = observation_space.spaces["position"].shape[0]

        # Nombre d'actions motrices possibles
        self.motor_action_dim = motor_action_space.n

        # Réseau pour la vision fovéale (observation actuelle et future)
        self.fovea_network = nn.Sequential(
            nn.Conv2d(self.fovea_shape[0] * 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Réseau pour les coordonnées de position (actuelle et future)
        self.position_network = nn.Sequential(
            nn.Linear(self.position_dim * 2, 32), nn.ReLU()
        )

        # Calculer les tailles de sortie des réseaux
        fovea_out_size = self._get_conv_output_size(
            self.fovea_network,
            (self.fovea_shape[0] * 2, self.fovea_shape[1], self.fovea_shape[2]),
        )

        # Couche de fusion
        fusion_input_size = (
            fovea_out_size + 32
        )  # 32 de position_network
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_input_size, 256), nn.ReLU())

        # Tête de prédiction pour l'action motrice
        self.motor_head = nn.Linear(256, self.motor_action_dim)

    def _get_conv_output_size(self, model, input_shape):
        """Calcule la taille de sortie d'un modèle convolutif"""
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = model(x)
            return x.numel()

    def forward(self, observations_pair):
        """
        Traitement des paires d'observations (actuelle et future)

        Args:
            observations_pair: dictionnaire contenant 'fovea', 'peripheral' et 'position'
                               pour les observations actuelles et futures
        """
        # Extraire et concaténer les observations actuelles et futures
        current_fovea = observations_pair["current"]["fovea"]
        next_fovea = observations_pair["next"]["fovea"]
        fovea_concat = torch.cat([current_fovea, next_fovea], dim=1)

        current_position = observations_pair["current"]["position"]
        next_position = observations_pair["next"]["position"]
        position_concat = torch.cat([current_position, next_position], dim=1)

        # Traiter chaque flux
        fovea_features = self.fovea_network(fovea_concat)
        position_features = self.position_network(position_concat)

        # Fusionner les caractéristiques
        combined = torch.cat(
            [fovea_features, position_features], dim=1
        )
        features = self.fusion_layer(combined)

        # Prédire l'action motrice
        motor_prediction = self.motor_head(features)

        return motor_prediction

    def get_loss(self, predicted_actions, true_actions):
        """Calcule la perte pour l'entraînement du SFN"""
        return F.cross_entropy(predicted_actions, true_actions)