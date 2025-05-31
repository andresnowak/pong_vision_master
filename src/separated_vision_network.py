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
    def __init__(self, observation_space, motor_action_space, sensory_action_set):
        super().__init__()

        # Taille originale de la fovéa (C, H, W)
        self.fovea_shape = observation_space.spaces["fovea"].shape

        # Après padding, H+1, W+1
        in_channels = self.fovea_shape[0]

        # Réseau convolutif unique pour l'image fovéale étendue
        self.fovea_network = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcul de la taille de sortie de la couche conv
        padded_shape = (in_channels, self.fovea_shape[1] + 1, self.fovea_shape[2] + 1)
        fovea_out_size = self._get_conv_output_size(self.fovea_network, padded_shape)

        # Réseau de fusion directe de la carte de caractéristiques
        self.fusion_layer = nn.Sequential(nn.Linear(fovea_out_size, 256), nn.ReLU())

        # Têtes pour actions motrices et sensorielles
        self.motor_head = nn.Linear(256, motor_action_space.n)
        self.sensory_head = nn.Linear(256, len(sensory_action_set))

    def _get_conv_output_size(self, model, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            return model(x).numel()

    def forward(self, observations):
        fovea = observations["fovea"]  # (B, C, H, W)
        position = observations[
            "position"
        ]  # (B, 2) avec (x, y) entiers [0..W-1], [0..H-1]

        B, C, H, W = fovea.shape
        # Créer la fovéa étendue de taille (H+1, W+1)
        padded = fovea.new_zeros(B, C, H + 1, W + 1)
        padded[:, :, :H, :W] = fovea

        # Normalisation des coordonnées entre -1 et 1
        pos_x = (position[:, 0].float() / (W - 1)) * 2 - 1  # (B,)
        pos_y = (position[:, 1].float() / (H - 1)) * 2 - 1  # (B,)

        # Remplir la dernière ligne (h == H) avec pos_x
        padded[:, :, H, :W] = pos_x.view(B, 1, 1).expand(B, C, W)
        # Remplir la dernière colonne (w == W) avec pos_y
        padded[:, :, :H, W] = pos_y.view(B, 1, 1).expand(B, C, H)

        # Passer la fovéa étendue dans le réseau conv
        fovea_features = self.fovea_network(padded)

        # Fusion et têtes d'action
        features = self.fusion_layer(fovea_features)
        motor_q = self.motor_head(features)
        sensory_q = self.sensory_head(features)

        return motor_q, sensory_q
    

class SeparatedFoveaSFN(nn.Module):
    """
    State Forecasting Network adapté pour traiter paires de fovéa + coordonnées spatiales.
    Prédit l'action motrice à partir des fovéas actuelle et future,
    en incorporant la position directement dans l'espace image.
    """

    def __init__(self, observation_space, motor_action_space):
        super().__init__()
        # Dimensions Fovea (C, H, W)
        C, H, W = observation_space.spaces["fovea"].shape
        # On double les canaux pour current + next
        in_channels = C * 2
        # Convolution unique après padding spatial
        self.fovea_network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calcul taille de sortie conv pour entrée (in_channels, H+1, W+1)
        conv_output_size = self._get_conv_output_size(
            self.fovea_network, (in_channels, H + 1, W + 1)
        )
        # Fusion et tête motrice
        self.fusion = nn.Sequential(nn.Linear(conv_output_size, 256), nn.ReLU())
        self.motor_head = nn.Linear(256, motor_action_space.n)

    def _get_conv_output_size(self, model, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            return model(x).numel()

    def forward(self, observations_pair):
        # Récupérer fovéas et positions
        curr_f = observations_pair["current"]["fovea"]  # (B, C, H, W)
        next_f = observations_pair["next"]["fovea"]  # (B, C, H, W)
        curr_p = observations_pair["current"]["position"]  # (B, 2)
        next_p = observations_pair["next"]["position"]  # (B, 2)

        B, C, H, W = curr_f.shape
        # Concatener le canal fovéa
        fovea_cat = torch.cat([curr_f, next_f], dim=1)  # (B, 2C, H, W)
        # Créer tensor padding (H+1, W+1)
        padded = fovea_cat.new_zeros(B, 2 * C, H + 1, W + 1)
        padded[:, :, :H, :W] = fovea_cat

        # Normaliser coords en [-1,1]
        cx = (curr_p[:, 0].float() / (W - 1)) * 2 - 1
        cy = (curr_p[:, 1].float() / (H - 1)) * 2 - 1
        nx = (next_p[:, 0].float() / (W - 1)) * 2 - 1
        ny = (next_p[:, 1].float() / (H - 1)) * 2 - 1

        # Remplir dernière ligne et colonne pour chaque moitié de canaux
        # canaux [0:C) = current, [C:2C) = next
        # ligne H
        padded[:, :C, H, :W] = cx.view(B, 1, 1).expand(B, C, W)
        padded[:, C:, H, :W] = nx.view(B, 1, 1).expand(B, C, W)
        # colonne W
        padded[:, :C, :H, W] = cy.view(B, 1, 1).expand(B, C, H)
        padded[:, C:, :H, W] = ny.view(B, 1, 1).expand(B, C, H)

        # Passer au conv
        features = self.fovea_network(padded)
        features = self.fusion(features)
        return self.motor_head(features)

    def get_loss(self, predicted, target):
        return F.cross_entropy(predicted, target)