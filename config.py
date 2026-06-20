"""Configuration centrale du projet de modélisation thermique de la maison.

Toutes les constantes partagées (chemins, fréquences d'échantillonnage,
fenêtres temporelles, hyperparamètres des modèles) vivent ici pour éviter
les valeurs en dur dispersées dans le code.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"

INDOOR_DIR = DATA_RAW_DIR / "indoor"
OUTDOOR_DIR = DATA_RAW_DIR / "outdoor"
WEATHER_DIR = DATA_RAW_DIR / "weather"
HOUSE_STATE_DIR = DATA_RAW_DIR / "house_state"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
STRATEGY_OUTPUT_DIR = PROJECT_ROOT / "strategy" / "output"

# ---------------------------------------------------------------------------
# Localisation (pour le calcul de position solaire — à adapter)
# ---------------------------------------------------------------------------

LATITUDE = 48.8566   # TODO: remplacer par la latitude réelle de la maison
LONGITUDE = 2.3522   # TODO: remplacer par la longitude réelle de la maison
ELEVATION = 35.0     # mètres au-dessus du niveau de la mer
TIMEZONE = "Europe/Paris"

# Orientation des façades de la maison, en degrés (0 = Nord, 90 = Est, ...)
# Sert à convertir l'azimut solaire en "exposition" par façade.
HOUSE_FACES = {
    "N": 0.0,
    "E": 90.0,
    "S": 180.0,
    "W": 270.0,
}

# ---------------------------------------------------------------------------
# Échantillonnage temporel
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_MINUTES = 2

# ---------------------------------------------------------------------------
# Fenêtre d'historique multi-résolution
#
# Chaque segment couvre une portion du passé avec une résolution donnée.
# Les segments sont ordonnés du plus récent au plus ancien et s'enchaînent :
# le premier segment couvre [0, duration), le suivant
# [duration, duration + duration_2), etc.
#
# Avec les valeurs par défaut : 18h d'historique en 108 pas au lieu de 540.
# ---------------------------------------------------------------------------

RESOLUTION_SEGMENTS = [
    {"duration_minutes": 120, "resolution_minutes": 2},   # 0-2h, résolution native
    {"duration_minutes": 240, "resolution_minutes": 10},  # 2-6h
    {"duration_minutes": 720, "resolution_minutes": 30},  # 6-18h
]

HISTORY_HOURS = sum(s["duration_minutes"] for s in RESOLUTION_SEGMENTS) / 60.0


def resolution_segments_for(history_hours: float) -> list[dict]:
    """Tronque RESOLUTION_SEGMENTS pour ne couvrir qu'une fenêtre plus courte.

    Permet de tester facilement des historiques de 6h/12h/18h en conservant
    la même structure multi-résolution.
    """
    remaining = history_hours * 60.0
    segments: list[dict] = []
    for seg in RESOLUTION_SEGMENTS:
        if remaining <= 0:
            break
        duration = min(seg["duration_minutes"], remaining)
        segments.append({"duration_minutes": duration, "resolution_minutes": seg["resolution_minutes"]})
        remaining -= duration
    return segments


# Horizon de prédiction (en pas de temps de SAMPLE_INTERVAL_MINUTES)
PREDICTION_HORIZON_STEPS = 1  # t+2min par défaut

# ---------------------------------------------------------------------------
# Convention de nommage des colonnes après pivot/fusion
#
# <source>__<clef>__<mesure>   ex: indoor__salon__temperature
# <source>__<mesure>           ex: weather__outdoor_temperature
# ---------------------------------------------------------------------------

COLUMN_SEP = "__"

# Préfixes/mesures attendus par source (utilisés par les loaders et la
# génération de données simulées).
INDOOR_MEASURES = ["temperature", "humidity"]
OUTDOOR_MEASURES = ["temperature", "humidity", "luminosity"]
HOUSE_STATE_TYPES = ["shutter", "window"]

# Mesures météo attendues (observées ou prévisions, voir colonne `kind`)
WEATHER_MEASURES = ["outdoor_temperature", "solar_irradiance", "cloud_cover"]

# Capteurs intérieurs simulés par défaut (4 pièces -> 8 features à prédire)
# Aligné sur config.ROOMS du dashboard HomeOS (salon, chambre parentale,
# chambre enfant, bureau).
DEFAULT_INDOOR_ROOMS = ["salon", "chambre1", "chambre2", "bureau"]

# Pièces possédant un volet et/ou une fenêtre pilotable (simulation)
DEFAULT_HOUSE_STATE_ROOMS = ["salon", "chambre1", "chambre2", "bureau"]

# Façade exposée par pièce : uniquement pour la VÉRITÉ TERRAIN du générateur
# de données simulées (chaque pièce simulée doit bien réagir à un côté de la
# maison). Ce n'est PAS une feature donnée au modèle : l'orientation de
# chaque pièce doit rester une caractéristique latente, apprise par le modèle
# à partir des corrélations entre les features solaires globales
# (solar__face_exposure__N/E/S/W) et la réponse de chaque capteur intérieur.
DEFAULT_ROOM_FACES = {"salon": "S", "chambre1": "E", "chambre2": "W", "bureau": "N"}

# ---------------------------------------------------------------------------
# Hyperparamètres des modèles
# ---------------------------------------------------------------------------

GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.1

# Taille de la couche de correction du modèle "full"
FULL_CORRECTION_HIDDEN_SIZE = 32

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 50
RANDOM_SEED = 42

# Arrêt anticipé : nombre d'epochs sans amélioration du val_loss avant
# d'arrêter l'entraînement. Le meilleur état (sur val_loss) est restauré
# avant de retourner le modèle.
EARLY_STOPPING_PATIENCE = 8

# ---------------------------------------------------------------------------
# Module strategy/ : planification volets/fenêtres
# ---------------------------------------------------------------------------

# Plage de température cible (confort), appliquée à toutes les pièces.
# Le coût de planification pénalise uniquement les dépassements de cette
# plage (pas d'objectif de valeur précise à l'intérieur).
COMFORT_TEMP_MIN = 19.0  # °C
COMFORT_TEMP_MAX = 26.0  # °C

# Horizon de planification et granularité des créneaux volets/fenêtres
# candidats (un même état sur tout le créneau).
PLANNING_HORIZON_HOURS = 24.0
PLANNING_BLOCK_HOURS = 2.0

# Résolution à laquelle la température prédite est évaluée pour le coût de
# confort (pas besoin du pas natif de 2 min pour juger du confort).
PLANNING_EVAL_STEP_MINUTES = 30.0

# Nombre de plannings candidats testés par recherche aléatoire.
PLANNING_N_CANDIDATES = 500
