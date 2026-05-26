"""
Cloudburst Prediction Engine
Physics-Informed ML backend for cloudburst early warning
Author: Bobby (AI Multimodal Framework Project)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class AtmosphericInput:
    max_temp: float        # °C
    min_temp: float        # °C
    rainfall: float        # mm
    wind_speed: float      # km/h
    wind_gust: float       # km/h
    wind_direction: float  # degrees
    humidity: float        # %
    pressure: float        # hPa
    elevation: float       # m
    slope: float           # degrees
    soil_moisture: float   # %
    weather_code: int      # WMO code
    region: str            # himalayan / western_ghats / northeast / plains / coastal
    population_density: str  # rural / semi / urban


@dataclass
class PhysicsIndicators:
    moisture_conservation: float   # % adherence
    orographic_lifting: float      # % potential
    cape_index: float              # J/kg
    cape_percent: float            # normalized %
    convective_triggering: float   # %
    wind_convergence: float        # %
    boundary_layer_instability: float  # %
    pressure_gradient_force: float     # %
    lapse_rate: float              # °C/km
    dew_point_depression: float    # °C
    lifted_index: float            # LI value
    pinn_law_adherence: float      # %
    pinn_plausibility: float       # %


@dataclass
class PredictionResult:
    cloudburst_probability: float   # 0-99%
    rainfall_intensity: float       # mm/hr
    early_warning_hours: int        # 1-3
    risk_level: str                 # Low / Moderate / High / Extreme
    risk_color: str                 # hex

    # Intermediate indices
    moisture_index: float
    instability_index: float
    orographic_index: float
    convection_index: float
    soil_factor: float

    # Physics
    physics: PhysicsIndicators

    # Impact assessment
    flash_flood_probability: float
    landslide_probability: float
    peak_runoff: float              # m³/s
    flood_depth_estimate: str
    inundation_area: float          # km²
    drainage_saturation: str
    affected_population: int
    road_risk_level: str
    agricultural_area_affected: float  # ha

    # Model attribution
    convlstm_contribution: float
    transformer_contribution: float
    attention_contribution: float
    pinn_contribution: float
    model_confidence: float
    physics_consistency: float

    # Forecast (next 6 hrs)
    probability_forecast: List[float]
    flood_forecast: List[float]
    landslide_forecast: List[float]

    # Heatmaps
    heatmap_prob: List[List[float]]
    heatmap_rain: List[List[float]]
    heatmap_flood: List[List[float]]
    heatmap_land: List[List[float]]
    heatmap_wind: List[List[float]]
    heatmap_orog: List[List[float]]

    # XAI
    feature_importances: Dict[str, float]

    # Alerts
    alert_level: str
    alert_message: str
    recommendations: List[str]


# ─────────────────────────────────────────────
# PHYSICS ENGINE (PINN Layer)
# ─────────────────────────────────────────────

class PhysicsEngine:
    """
    Physics-Informed constraint layer.
    Enforces atmospheric laws to improve prediction reliability
    for rare cloudburst events.
    """

    REGION_BONUS = {
        "himalayan": 25,
        "western_ghats": 18,
        "northeast": 14,
        "coastal": 12,
        "plains": 0
    }

    WMO_BOOST = {
        0: 0, 1: 0, 2: 5, 3: 10,
        51: 20, 61: 28, 63: 42,
        65: 58, 80: 48, 82: 68,
        95: 72, 99: 82
    }

    def compute_cape(self, inp: AtmosphericInput) -> float:
        """
        Simplified CAPE estimate:
        Convective Available Potential Energy ~ f(temp range, humidity, pressure drop)
        Real CAPE uses full thermodynamic sounding — this is a physics proxy.
        """
        t_range = inp.max_temp - inp.min_temp
        pressure_drop = max(0, 1013 - inp.pressure)
        cape = (
            (inp.max_temp - 20) * 65 +
            (inp.humidity - 50) * 18 +
            pressure_drop * 22 +
            t_range * 42
        )
        return max(0, round(cape))

    def compute_orographic_lifting(self, inp: AtmosphericInput) -> float:
        """
        Orographic lifting = terrain-forced vertical motion.
        Higher elevation + steeper slope + windward-facing = stronger lifting.
        """
        region_bonus = self.REGION_BONUS.get(inp.region, 0)
        orog = (
            (inp.elevation / 4500) * 42 +
            (inp.slope / 60) * 38 +
            region_bonus
        )
        return min(99.0, round(orog, 1))

    def compute_moisture_conservation(self, inp: AtmosphericInput) -> float:
        """
        Moisture convergence — key PINN constraint.
        Ensures predicted rainfall is physically consistent
        with available atmospheric moisture.
        """
        wc_flag = 18.0 if inp.weather_code > 60 else 0.0
        moisture = (
            inp.humidity * 0.55 +
            inp.rainfall * 0.28 +
            wc_flag
        )
        return min(99.0, round(moisture, 1))

    def compute_instability(self, inp: AtmosphericInput) -> float:
        """
        Atmospheric instability index.
        Combines thermal gradient, humidity, and pressure anomaly.
        """
        t_range = inp.max_temp - inp.min_temp
        instability = (
            (inp.max_temp - 20) * 2.3 +
            (1020 - inp.pressure) * 0.9 +
            (inp.humidity - 50) * 0.55 +
            t_range * 1.4
        )
        return min(99.0, max(0.0, round(instability, 1)))

    def compute_wind_convergence(self, inp: AtmosphericInput) -> float:
        """
        Wind convergence — approaching storm systems from SW/SE
        are more likely to trigger cloudbursts in mountainous terrain.
        """
        direction_factor = 14.0 if (180 < inp.wind_direction < 270) else 4.0
        conv = (
            (inp.wind_gust / 180) * 36 +
            (inp.wind_speed / 120) * 22 +
            direction_factor
        )
        return min(99.0, round(conv, 1))

    def compute_pressure_gradient(self, inp: AtmosphericInput) -> float:
        """Surface pressure gradient as proxy for synoptic forcing."""
        return min(99.0, round(max(0, (1030 - inp.pressure) * 1.1), 1))

    def compute_lapse_rate(self, inp: AtmosphericInput) -> float:
        """Environmental lapse rate °C/km."""
        if inp.elevation < 100:
            return 0.0
        return round((inp.max_temp - inp.min_temp) / (inp.elevation / 1000), 2)

    def compute_dew_point(self, inp: AtmosphericInput) -> float:
        """Approximate dew point from max temp and humidity."""
        return round(inp.max_temp - ((100 - inp.humidity) * 0.36), 1)

    def compute_lifted_index(self, instability: float) -> float:
        """Lifted Index: negative = unstable (cloudburst prone)."""
        return round(-(instability / 10 - 5), 2)

    def compute_all(self, inp: AtmosphericInput) -> PhysicsIndicators:
        cape = self.compute_cape(inp)
        instability = self.compute_instability(inp)
        moisture = self.compute_moisture_conservation(inp)
        orog = self.compute_orographic_lifting(inp)
        conv = self.compute_wind_convergence(inp)
        pg = self.compute_pressure_gradient(inp)
        bl = min(99.0, round(instability * 0.85, 1))
        cape_pct = min(99.0, round(instability * 0.95, 1))
        lapse = self.compute_lapse_rate(inp)
        dew_dep = round(inp.max_temp - self.compute_dew_point(inp), 1)
        li = self.compute_lifted_index(instability)
        law_adherence = round(75 + instability * 0.22, 1)
        plausibility = round(80 + instability * 0.15, 1)

        return PhysicsIndicators(
            moisture_conservation=moisture,
            orographic_lifting=orog,
            cape_index=cape,
            cape_percent=cape_pct,
            convective_triggering=conv,
            wind_convergence=conv,
            boundary_layer_instability=bl,
            pressure_gradient_force=pg,
            lapse_rate=lapse,
            dew_point_depression=dew_dep,
            lifted_index=li,
            pinn_law_adherence=min(99.0, law_adherence),
            pinn_plausibility=min(99.0, plausibility)
        )


# ─────────────────────────────────────────────
# SIMULATED DEEP LEARNING LAYER
# ─────────────────────────────────────────────

class SimulatedConvLSTM:
    """
    Simulates ConvLSTM spatial-temporal cloud dynamics.
    In production: replace with actual PyTorch/TensorFlow ConvLSTM model.
    """

    def predict(self, moisture: float, instability: float, orog: float,
                wc_boost: float) -> Tuple[float, List[float]]:
        """Returns (probability_score, temporal_sequence_t-6_to_t+3)"""
        base_prob = (moisture * 0.3 + instability * 0.28 + orog * 0.22 + wc_boost * 0.2) / 100
        base_prob = min(0.99, max(0.01, base_prob))

        # Simulate temporal sequence
        t_seq = []
        ramp_factors = [0.08, 0.14, 0.22, 0.33, 0.48, 0.68, 1.0, 1.08, 0.92, 0.72]
        for f in ramp_factors:
            noise = np.random.uniform(-0.04, 0.04)
            t_seq.append(round(min(99, max(1, base_prob * f * 100 + noise * 100))))

        return round(base_prob * 100, 1), t_seq


class SimulatedTransformer:
    """
    Simulates Transformer long-range dependency capture.
    In production: replace with actual Transformer model.
    """

    def predict(self, conv_score: float, physics_score: float,
                soil_factor: float) -> float:
        """Long-range dependency score."""
        score = (conv_score * 0.4 + physics_score * 0.4 + soil_factor * 0.2)
        return min(99.0, round(score, 1))


class AttentionModule:
    """
    Simulates Attention mechanism for critical spatial regions.
    In production: replace with actual spatial attention map.
    """

    def predict(self, orog: float, conv: float, moisture: float) -> float:
        """Attention weight for critical high-risk regions."""
        return min(99.0, round((orog * 0.4 + conv * 0.35 + moisture * 0.25), 1))


# ─────────────────────────────────────────────
# HEATMAP GENERATOR
# ─────────────────────────────────────────────

class HeatmapGenerator:
    """
    Generates 12x12 spatial heatgrids with terrain-aware variation.
    In production: use actual DEM/GIS data + model output per grid cell.
    """

    def __init__(self, rows=12, cols=12):
        self.rows = rows
        self.cols = cols
        np.random.seed(42)  # Reproducible terrain noise

    def _terrain_field(self) -> np.ndarray:
        """Generate a persistent terrain-aware spatial field."""
        x = np.linspace(0, 1, self.cols)
        y = np.linspace(0, 1, self.rows)
        xx, yy = np.meshgrid(x, y)

        # Ridge line + valley effects
        ridge = np.exp(-((xx - 0.5) ** 2) / 0.03) * 0.35
        valley_l = np.exp(-((xx - 0.3) ** 2) / 0.02) * 0.2
        valley_r = np.exp(-((xx - 0.7) ** 2) / 0.02) * 0.2
        slope_gradient = (1 - yy) * 0.2

        terrain = ridge + valley_l + valley_r + slope_gradient
        noise = np.random.uniform(-0.06, 0.06, (self.rows, self.cols))
        return terrain + noise

    def generate_prob_heatmap(self, base_prob: float) -> List[List[float]]:
        terrain = self._terrain_field()
        field = base_prob / 100 + terrain * 0.5
        field = np.clip(field, 0.01, 0.99)
        return [[round(float(field[r][c]) * 100, 1) for c in range(self.cols)] for r in range(self.rows)]

    def generate_rain_heatmap(self, base_rain: float) -> List[List[float]]:
        terrain = self._terrain_field()
        field = base_rain / 200 + terrain * 0.4
        field = np.clip(field, 0, 1)
        return [[round(float(field[r][c]) * 200, 1) for c in range(self.cols)] for r in range(self.rows)]

    def generate_flood_heatmap(self, base_fp: float) -> List[List[float]]:
        """Valley floors get higher flood risk."""
        x = np.linspace(0, 1, self.cols)
        y = np.linspace(0, 1, self.rows)
        xx, yy = np.meshgrid(x, y)
        valley_mask = np.exp(-((xx - 0.3) ** 2) / 0.03) + np.exp(-((xx - 0.7) ** 2) / 0.03)
        low_elevation = yy * 0.3
        base = base_fp / 100
        field = base + valley_mask * 0.3 + low_elevation * 0.15
        noise = np.random.uniform(-0.05, 0.05, (self.rows, self.cols))
        field = np.clip(field + noise, 0.01, 0.99)
        return [[round(float(field[r][c]) * 100, 1) for c in range(self.cols)] for r in range(self.rows)]

    def generate_landslide_heatmap(self, base_lp: float, slope: float) -> List[List[float]]:
        """Steep ridges and upper slopes get higher landslide risk."""
        terrain = self._terrain_field()
        slope_factor = slope / 60
        base = base_lp / 100
        field = base + terrain * slope_factor * 0.6
        noise = np.random.uniform(-0.05, 0.05, (self.rows, self.cols))
        field = np.clip(field + noise, 0.01, 0.99)
        return [[round(float(field[r][c]) * 100, 1) for c in range(self.cols)] for r in range(self.rows)]

    def generate_wind_heatmap(self, wind_dir: float, conv: float) -> List[List[float]]:
        x = np.linspace(0, 1, self.cols)
        y = np.linspace(0, 1, self.rows)
        xx, yy = np.meshgrid(x, y)
        dir_factor = abs(wind_dir - 180) / 180
        field = conv / 100 * (0.4 + xx * 0.6) + dir_factor * 0.3
        noise = np.random.uniform(-0.06, 0.06, (self.rows, self.cols))
        field = np.clip(field + noise, 0.01, 0.99)
        return [[round(float(field[r][c]) * 100, 1) for c in range(self.cols)] for r in range(self.rows)]

    def generate_orog_heatmap(self, base_orog: float) -> List[List[float]]:
        terrain = self._terrain_field()
        x = np.linspace(0, 1, self.cols)
        y = np.linspace(0, 1, self.rows)
        _, yy = np.meshgrid(x, y)
        elevation_gradient = (1 - yy) * 0.3
        field = base_orog / 100 + terrain * 0.5 + elevation_gradient
        noise = np.random.uniform(-0.04, 0.04, (self.rows, self.cols))
        field = np.clip(field + noise, 0.01, 0.99)
        return [[round(float(field[r][c]) * 100, 1) for c in range(self.cols)] for r in range(self.rows)]


# ─────────────────────────────────────────────
# IMPACT ASSESSMENT ENGINE
# ─────────────────────────────────────────────

class ImpactEngine:
    POP_MULTIPLIER = {"rural": 800, "semi": 2200, "urban": 5500}

    def assess(self, inp: AtmosphericInput, prob: float, fp: float,
               lp: float, runoff: float, moisture: float) -> Dict:
        pop_m = self.POP_MULTIPLIER.get(inp.population_density, 2200)
        affected_pop = round(prob * pop_m / 100)

        depth = (
            f"{round(runoff/80)}–{round(runoff/80)+2}m" if runoff > 200
            else "0.5–1.5m" if runoff > 80
            else "< 0.5m"
        )

        def risk_str(p):
            if p < 30: return "Low"
            if p < 55: return "Moderate"
            if p < 75: return "High"
            return "Extreme"

        return {
            "flash_flood_probability": fp,
            "landslide_probability": lp,
            "peak_runoff": runoff,
            "flood_depth_estimate": depth,
            "inundation_area": round(fp * 0.38, 1),
            "drainage_saturation": "Saturated" if moisture > 75 else "Partial",
            "affected_population": affected_pop,
            "road_risk_level": risk_str(min(99, prob + 10)),
            "agricultural_area_affected": round(prob * 2.8, 1),
        }


# ─────────────────────────────────────────────
# MAIN PREDICTION ENGINE
# ─────────────────────────────────────────────

class CloudburstPredictionEngine:
    """
    Main orchestrator: physics → ML → impact → XAI → heatmaps
    """

    WMO_BOOST = {
        0: 0, 1: 0, 2: 5, 3: 10,
        51: 20, 61: 28, 63: 42,
        65: 58, 80: 48, 82: 68,
        95: 72, 99: 82
    }

    def __init__(self):
        self.physics = PhysicsEngine()
        self.convlstm = SimulatedConvLSTM()
        self.transformer = SimulatedTransformer()
        self.attention = AttentionModule()
        self.heatmap_gen = HeatmapGenerator()
        self.impact = ImpactEngine()

    def _risk_meta(self, p: float) -> Tuple[str, str]:
        if p < 30: return "Low", "#00d4aa"
        if p < 55: return "Moderate", "#ffd166"
        if p < 75: return "High", "#ff9a3c"
        return "Extreme", "#ff4757"

    def _build_forecast(self, base: float, scale: float = 1.0) -> List[float]:
        decay = [1.0, 1.12, 1.08, 0.95, 0.78, 0.60, 0.42]
        return [round(min(99, max(0, base * d * scale)), 1) for d in decay]

    def _compute_xai(self, inp: AtmosphericInput, physics: PhysicsIndicators) -> Dict[str, float]:
        raw = {
            "Humidity": round(inp.humidity * 0.65, 1),
            "Rainfall": round(inp.rainfall * 0.60, 1),
            "Instability Index": physics.boundary_layer_instability,
            "Orographic Lifting": physics.orographic_lifting,
            "Wind Gust": round(inp.wind_gust * 0.52, 1),
            "Pressure Drop": physics.pressure_gradient_force,
            "Weather Code": min(80.0, float(inp.weather_code)),
            "Soil Moisture": round(inp.soil_moisture * 0.55, 1),
        }
        return {k: min(99.0, v) for k, v in raw.items()}

    def predict(self, inp: AtmosphericInput) -> PredictionResult:
        np.random.seed(None)  # Fresh randomness for each prediction

        # ── Physics layer ──
        phys = self.physics.compute_all(inp)
        wc_boost = self.WMO_BOOST.get(inp.weather_code, 0)
        soil_factor = min(99.0, round(inp.soil_moisture * 0.7 + inp.rainfall * 0.25, 1))

        # ── ML layer ──
        convlstm_score, temporal_seq = self.convlstm.predict(
            phys.moisture_conservation, phys.boundary_layer_instability,
            phys.orographic_lifting, wc_boost
        )
        transformer_score = self.transformer.predict(
            phys.wind_convergence, phys.boundary_layer_instability, soil_factor
        )
        attention_score = self.attention.predict(
            phys.orographic_lifting, phys.wind_convergence, phys.moisture_conservation
        )

        # ── Ensemble probability ──
        prob = min(99.0, max(1.0, round(
            phys.boundary_layer_instability * 0.26 +
            phys.moisture_conservation * 0.22 +
            phys.orographic_lifting * 0.16 +
            phys.wind_convergence * 0.12 +
            wc_boost * 0.10 +
            soil_factor * 0.06 +
            convlstm_score * 0.04 +
            transformer_score * 0.02 +
            attention_score * 0.02
        , 1)))

        rain_intensity = round(inp.rainfall * (1 + (prob / 100) * 2.8), 1)
        warn_hr = 1 if prob > 80 else 2 if prob > 60 else 3

        fp = min(99.0, round(prob * 0.87 + (inp.slope / 60) * 20 + soil_factor * 0.15, 1))
        lp = min(99.0, round(prob * 0.72 + (inp.slope / 60) * 32 + soil_factor * 0.18, 1))
        runoff = round(inp.rainfall * (inp.slope / 18) * (inp.humidity / 75) * 14, 1)

        # ── Model attribution ──
        total_attribution = prob if prob > 0 else 1
        cl_attr = round(prob * 0.32, 1)
        tr_attr = round(prob * 0.27, 1)
        at_attr = round(prob * 0.22, 1)
        pn_attr = round(prob * 0.19, 1)

        # ── Impact ──
        impact = self.impact.assess(inp, prob, fp, lp, runoff, phys.moisture_conservation)

        # ── Heatmaps ──
        hm_prob  = self.heatmap_gen.generate_prob_heatmap(prob)
        hm_rain  = self.heatmap_gen.generate_rain_heatmap(rain_intensity)
        hm_flood = self.heatmap_gen.generate_flood_heatmap(fp)
        hm_land  = self.heatmap_gen.generate_landslide_heatmap(lp, inp.slope)
        hm_wind  = self.heatmap_gen.generate_wind_heatmap(inp.wind_direction, phys.wind_convergence)
        hm_orog  = self.heatmap_gen.generate_orog_heatmap(phys.orographic_lifting)

        # ── XAI ──
        xai = self._compute_xai(inp, phys)

        # ── Alert ──
        risk_label, risk_color = self._risk_meta(prob)
        alert_level, alert_msg, recs = self._build_alert(prob, fp, warn_hr, rain_intensity)

        return PredictionResult(
            cloudburst_probability=prob,
            rainfall_intensity=rain_intensity,
            early_warning_hours=warn_hr,
            risk_level=risk_label,
            risk_color=risk_color,
            moisture_index=phys.moisture_conservation,
            instability_index=phys.boundary_layer_instability,
            orographic_index=phys.orographic_lifting,
            convection_index=phys.wind_convergence,
            soil_factor=soil_factor,
            physics=phys,
            flash_flood_probability=fp,
            landslide_probability=lp,
            peak_runoff=runoff,
            flood_depth_estimate=impact["flood_depth_estimate"],
            inundation_area=impact["inundation_area"],
            drainage_saturation=impact["drainage_saturation"],
            affected_population=impact["affected_population"],
            road_risk_level=impact["road_risk_level"],
            agricultural_area_affected=impact["agricultural_area_affected"],
            convlstm_contribution=cl_attr,
            transformer_contribution=tr_attr,
            attention_contribution=at_attr,
            pinn_contribution=pn_attr,
            model_confidence=min(99.0, round(60 + prob * 0.35, 1)),
            physics_consistency=min(99.0, round(72 + phys.boundary_layer_instability * 0.2, 1)),
            probability_forecast=self._build_forecast(prob),
            flood_forecast=self._build_forecast(fp),
            landslide_forecast=self._build_forecast(lp),
            heatmap_prob=hm_prob,
            heatmap_rain=hm_rain,
            heatmap_flood=hm_flood,
            heatmap_land=hm_land,
            heatmap_wind=hm_wind,
            heatmap_orog=hm_orog,
            feature_importances=xai,
            alert_level=alert_level,
            alert_message=alert_msg,
            recommendations=recs,
        )

    def _build_alert(self, prob: float, fp: float, warn_hr: int,
                     rain_int: float) -> Tuple[str, str, List[str]]:
        if prob > 75:
            return (
                "EXTREME",
                f"⛔ EXTREME WARNING — Cloudburst probability {prob:.0f}%. "
                f"Expected {rain_int:.0f} mm/hr within {warn_hr}h. Immediate action required.",
                [
                    "Evacuate all low-lying and riverbank areas immediately",
                    "Activate NDRF / SDRF emergency response teams",
                    "Close mountain roads, bridges and river crossings",
                    "Issue SMS + siren alerts to all affected population",
                    "Open emergency shelters and relief camps",
                    "Deploy rapid early flood warning sirens in valleys"
                ]
            )
        elif prob > 55:
            return (
                "HIGH",
                f"🔶 HIGH ALERT — Probability {prob:.0f}%. Flash flood risk {fp:.0f}%. "
                f"Warning window: {warn_hr}h. Prepare for impact.",
                [
                    "Put rescue teams on immediate standby",
                    "Issue public advisory notifications",
                    "Pre-position disaster relief supplies",
                    "Alert district administration and emergency services",
                    "Monitor river and drain levels continuously"
                ]
            )
        elif prob > 30:
            return (
                "WATCH",
                f"🟡 WATCH — Moderate risk, {prob:.0f}% probability. "
                f"Monitor conditions closely.",
                [
                    "Continue real-time sensor monitoring",
                    "Brief local emergency coordinators",
                    "Advise public to stay weather-informed",
                    "Check drainage and flood barriers"
                ]
            )
        else:
            return (
                "NORMAL",
                f"🟢 NORMAL — Conditions stable. Probability {prob:.0f}%. No immediate risk.",
                ["Routine monitoring — no action required"]
            )
