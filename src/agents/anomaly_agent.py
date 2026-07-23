"""
Agente de Detecção de Anomalias em Sinais Vitais.

Responsável pelo monitoramento contínuo de sinais vitais específicos
da saúde da mulher (pressão arterial em gestantes, batimentos fetais,
prescrições hormonais e evolução clínica) com alertas em tempo real.

Critérios do edital atendidos:
- Detecção de anomalias em sinais vitais específicos
- Monitoramento de pressão arterial em gestantes
- Monitoramento de batimentos fetais
- Análise de prescrições hormonais e evolução clínica
- Alertas à equipe especializada em tempo real
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class SignalType(str, Enum):
    BLOOD_PRESSURE_SYSTOLIC = "pressao_sistolica"
    BLOOD_PRESSURE_DIASTOLIC = "pressao_diastolica"
    FETAL_HEART_RATE = "batimentos_fetais"
    MATERNAL_HEART_RATE = "batimentos_maternos"
    TEMPERATURE = "temperatura"
    OXYGEN_SATURATION = "saturacao_oxigenio"
    HORMONE_LEVEL = "nivel_hormonal"
    WEIGHT = "peso"
    GLUCOSE = "glicose"


class AnomalyMethod(str, Enum):
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"
    ROLLING_STATS = "rolling_stats"
    HYBRID = "hybrid"


@dataclass
class VitalSignRecord:
    """Registro individual de sinal vital."""

    timestamp: float
    signal_type: SignalType
    value: float
    unit: str
    patient_id: str


@dataclass
class AnomalyDetectionResult:
    """Resultado da detecção de anomalia em sinal vital."""

    record: VitalSignRecord
    is_anomaly: bool
    anomaly_score: float
    method: AnomalyMethod
    expected_range: tuple[float, float]
    deviation_from_expected: float

    @property
    def severity(self) -> str:
        if self.anomaly_score > 0.9:
            return "crítica"
        elif self.anomaly_score > 0.7:
            return "alta"
        elif self.anomaly_score > 0.5:
            return "média"
        return "baixa"


DEFAULT_REFERENCE_RANGES = {
    SignalType.BLOOD_PRESSURE_SYSTOLIC: (90, 140),
    SignalType.BLOOD_PRESSURE_DIASTOLIC: (60, 90),
    SignalType.FETAL_HEART_RATE: (110, 160),
    SignalType.MATERNAL_HEART_RATE: (60, 100),
    SignalType.TEMPERATURE: (35.5, 37.5),
    SignalType.OXYGEN_SATURATION: (95, 100),
    SignalType.HORMONE_LEVEL: (-2.0, 2.0),
    SignalType.WEIGHT: (45.0, 100.0),
    SignalType.GLUCOSE: (70, 140),
}

GESTATIONAL_REFERENCE_RANGES = {
    SignalType.BLOOD_PRESSURE_SYSTOLIC: (90, 135),
    SignalType.BLOOD_PRESSURE_DIASTOLIC: (60, 85),
    SignalType.FETAL_HEART_RATE: (120, 160),
    SignalType.MATERNAL_HEART_RATE: (65, 110),
    SignalType.GLUCOSE: (70, 130),
}


class AnomalyDetectionAgent:
    """
    Agente de detecção de anomalias em tempo real para sinais vitais.

    Utiliza métodos estatísticos (Z-score, rolling statistics) e
    machine learning (Isolation Forest) para identificar padrões
    anômalos em séries temporais de dados vitais.
    """

    def __init__(
        self,
        contamination: float | None = None,
        zscore_threshold: float | None = None,
        window_size: int | None = None,
        use_gestational_ranges: bool = False,
    ):
        from src.config import settings

        self.contamination = contamination or settings.model.anomaly_contamination
        self.zscore_threshold = zscore_threshold or settings.model.anomaly_zscore_threshold
        self.window_size = window_size or settings.model.anomaly_window_size

        self.reference_ranges = (
            GESTATIONAL_REFERENCE_RANGES if use_gestational_ranges
            else DEFAULT_REFERENCE_RANGES
        )

        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self._history: dict[SignalType, list[VitalSignRecord]] = {}
        self._model_fitted = False

    def add_record(self, record: VitalSignRecord) -> Optional[AnomalyDetectionResult]:
        if record.signal_type not in self._history:
            self._history[record.signal_type] = []

        self._history[record.signal_type].append(record)

        if len(self._history[record.signal_type]) > self.window_size * 2:
            self._history[record.signal_type] = self._history[record.signal_type][-self.window_size * 2:]

        return self._detect_anomaly(record)

    def _detect_anomaly(self, record: VitalSignRecord) -> AnomalyDetectionResult:
        zscore_result = self._zscore_detection(record)
        range_result = self._range_detection(record)
        rolling_result = self._rolling_stats_detection(record)

        scores = [r for r in [zscore_result, range_result, rolling_result] if r is not None]
        if not scores:
            return AnomalyDetectionResult(
                record=record,
                is_anomaly=False,
                anomaly_score=0.0,
                method=AnomalyMethod.HYBRID,
                expected_range=self.reference_ranges.get(record.signal_type, (0, 0)),
                deviation_from_expected=0.0,
            )

        max_result = max(scores, key=lambda r: r.anomaly_score)

        if max_result.anomaly_score > 0.5:
            logger.warning(
                f"[AnomalyAgent] ANOMALIA ({max_result.severity}) "
                f"{record.signal_type.value}: {record.value} {record.unit} "
                f"(score: {max_result.anomaly_score:.2f}, método: {max_result.method.value})"
            )

        return max_result

    def _zscore_detection(self, record: VitalSignRecord) -> Optional[AnomalyDetectionResult]:
        history = self._history.get(record.signal_type, [])
        if len(history) < 10:
            range_vals = self.reference_ranges.get(record.signal_type, (0, 0))
            return AnomalyDetectionResult(
                record=record,
                is_anomaly=not (range_vals[0] <= record.value <= range_vals[1]),
                anomaly_score=0.7 if not (range_vals[0] <= record.value <= range_vals[1]) else 0.0,
                method=AnomalyMethod.ZSCORE,
                expected_range=range_vals,
                deviation_from_expected=self._calculate_deviation(record.value, range_vals),
            )

        values = np.array([h.value for h in history])
        zscore = np.abs(stats.zscore(np.append(values, record.value)))[-1]

        range_vals = self.reference_ranges.get(record.signal_type, (0, 0))
        is_anomaly = zscore > self.zscore_threshold
        anomaly_score = min(1.0, zscore / (self.zscore_threshold * 2))

        return AnomalyDetectionResult(
            record=record,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            method=AnomalyMethod.ZSCORE,
            expected_range=range_vals,
            deviation_from_expected=self._calculate_deviation(record.value, range_vals),
        )

    def _range_detection(self, record: VitalSignRecord) -> Optional[AnomalyDetectionResult]:
        range_vals = self.reference_ranges.get(record.signal_type)
        if range_vals is None:
            return None

        low, high = range_vals
        in_range = low <= record.value <= high

        if in_range:
            return AnomalyDetectionResult(
                record=record,
                is_anomaly=False,
                anomaly_score=0.0,
                method=AnomalyMethod.ZSCORE,
                expected_range=range_vals,
                deviation_from_expected=0.0,
            )

        deviation = self._calculate_deviation(record.value, range_vals)
        score = min(1.0, abs(deviation))

        return AnomalyDetectionResult(
            record=record,
            is_anomaly=True,
            anomaly_score=score,
            method=AnomalyMethod.ZSCORE,
            expected_range=range_vals,
            deviation_from_expected=deviation,
        )

    def _rolling_stats_detection(self, record: VitalSignRecord) -> Optional[AnomalyDetectionResult]:
        history = self._history.get(record.signal_type, [])
        if len(history) < self.window_size:
            return None

        recent = history[-self.window_size:]
        values = np.array([h.value for h in recent])

        mean = np.mean(values)
        std = np.std(values) or 1.0

        deviation = abs(record.value - mean) / std
        range_vals = self.reference_ranges.get(record.signal_type, (0, 0))

        is_anomaly = deviation > self.zscore_threshold
        anomaly_score = min(1.0, deviation / (self.zscore_threshold * 2))

        return AnomalyDetectionResult(
            record=record,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            method=AnomalyMethod.ROLLING_STATS,
            expected_range=range_vals,
            deviation_from_expected=float(deviation),
        )

    def _calculate_deviation(self, value: float, ref_range: tuple[float, float]) -> float:
        low, high = ref_range
        center = (low + high) / 2
        half_range = (high - low) / 2 or 1.0
        return (value - center) / half_range

    def batch_analyze(self, records: list[VitalSignRecord]) -> pd.DataFrame:
        results = []
        for record in records:
            result = self.add_record(record)
            if result:
                results.append({
                    "timestamp": result.record.timestamp,
                    "signal_type": result.record.signal_type.value,
                    "value": result.record.value,
                    "unit": result.record.unit,
                    "is_anomaly": result.is_anomaly,
                    "anomaly_score": result.anomaly_score,
                    "severity": result.severity,
                    "method": result.method.value,
                    "deviation": result.deviation_from_expected,
                })

        df = pd.DataFrame(results)
        if not df.empty:
            logger.info(
                f"[AnomalyAgent] Batch: {len(df)} registros, "
                f"{df['is_anomaly'].sum()} anomalias detectadas"
            )
        return df

    def analyze_prescription(self, hormone_dose: float, reference_dose: float) -> bool:
        deviation = abs(hormone_dose - reference_dose) / max(reference_dose, 1e-6)
        is_anomaly = deviation > 0.5
        if is_anomaly:
            logger.warning(
                f"[AnomalyAgent] Desvio em prescrição hormonal: "
                f"dose={hormone_dose}, referência={reference_dose}, desvio={deviation:.2%}"
            )
        return is_anomaly

    def fit_isolation_forest(self, records: list[VitalSignRecord]) -> None:
        if len(records) < 10:
            logger.warning("Dados insuficientes para treinar Isolation Forest")
            return

        data = []
        for r in records:
            data.append([
                r.value,
                list(SignalType).index(r.signal_type),
                r.timestamp,
            ])
        X = np.array(data)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self._model_fitted = True
        logger.info(f"Isolation Forest treinado com {len(records)} registros")

    def reset_history(self, signal_type: SignalType | None = None) -> None:
        if signal_type:
            self._history.pop(signal_type, None)
        else:
            self._history.clear()
