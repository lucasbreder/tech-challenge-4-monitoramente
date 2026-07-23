"""
Interface Streamlit para demonstração do sistema de monitoramento multimodal.

Permite upload e análise interativa de vídeos, áudios e sinais vitais,
com exibição de resultados e alertas em tempo real.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Saúde da Mulher - Monitoramento Multimodal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Sistema de Monitoramento Multimodal para Saúde da Mulher")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Pipeline Multimodal", "Análise de Vídeo", "Análise de Áudio", "Sinais Vitais"]
)

with st.sidebar:
    st.header("Configuração")
    st.info(
        "Este sistema utiliza IA para detectar riscos em saúde da mulher "
        "através de análise multimodal de vídeo, áudio e sinais vitais."
    )

    st.subheader("Modalidades suportadas")
    st.markdown("- **Vídeo:** Cirurgias, consultas, fisioterapia")
    st.markdown("- **Áudio:** Consultas ginecológicas, pré-natal, pós-parto")
    st.markdown("- **Sinais Vitais:** Pressão, batimentos fetais, hormonais")

    st.subheader("Modelos")
    st.markdown("- YOLOv8 (detecção visual)")
    st.markdown("- Whisper (transcrição)")
    st.markdown("- Wav2Vec2 (emoção vocal)")
    st.markdown("- Isolation Forest (anomalias)")

    st.subheader("Serviços Cloud")
    st.markdown("- Azure Cognitive Services")
    st.markdown("- Azure Blob Storage")

with tab1:
    st.header("Pipeline Multimodal Completo")
    st.markdown("Análise integrada de vídeo, áudio e sinais vitais com alertas unificados.")

    col1, col2 = st.columns(2)

    with col1:
        patient_id = st.text_input("ID da Paciente", value="PAC-001")
        video_file = st.file_uploader(
            "Upload de Vídeo (opcional)",
            type=["mp4", "avi", "mov", "mkv"],
            key="multimodal_video",
        )
        video_type = st.selectbox(
            "Tipo de Vídeo",
            ["consulta", "cirurgia", "fisioterapia", "triagem_violencia"],
        )

    with col2:
        audio_file = st.file_uploader(
            "Upload de Áudio (opcional)",
            type=["wav", "mp3", "m4a", "ogg"],
            key="multimodal_audio",
        )
        consultation_type = st.selectbox(
            "Tipo de Consulta",
            ["ginecologica", "pre_natal", "pos_parto", "vitima_violencia"],
        )

    if st.button("Executar Pipeline Multimodal", type="primary", use_container_width=True):
        if not video_file and not audio_file:
            st.warning("Faça upload de pelo menos um arquivo (vídeo ou áudio).")
        else:
            with st.spinner("Processando pipeline multimodal..."):
                video_path = None
                audio_path = None

                if video_file:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(video_file.name).suffix
                    ) as tmp:
                        tmp.write(video_file.read())
                        video_path = tmp.name

                if audio_file:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(audio_file.name).suffix
                    ) as tmp:
                        tmp.write(audio_file.read())
                        audio_path = tmp.name

                try:
                    from src.pipelines.multimodal_pipeline import MultimodalPipeline

                    pipeline = MultimodalPipeline()
                    assessment = pipeline.run(
                        patient_id=patient_id,
                        video_path=video_path,
                        audio_path=audio_path,
                        video_type=video_type,
                        consultation_type=consultation_type,
                        export_report=True,
                    )

                    st.success("Pipeline concluído com sucesso!")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Score de Risco", f"{assessment.overall_risk_score:.2f}")
                    col_b.metric("Nível", assessment.overall_risk_level.upper())
                    col_c.metric("Alertas", len(assessment.video_alerts) + len(assessment.audio_alerts))

                    if assessment.correlated_risks:
                        st.subheader("Riscos Correlacionados")
                        for risk in assessment.correlated_risks:
                            st.warning(risk)

                    st.subheader("Recomendações")
                    for rec in assessment.recommendations:
                        st.info(rec)

                    with st.expander("Detalhes completos (JSON)"):
                        st.json(assessment.to_dict())

                except Exception as e:
                    st.error(f"Erro no pipeline: {e}")

with tab2:
    st.header("Análise de Vídeo Clínico")
    st.markdown("Detecção de sangramento anômalo, áreas críticas e sinais de desconforto.")

    vid_file = st.file_uploader(
        "Upload de Vídeo",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_only",
    )
    vid_type = st.selectbox(
        "Tipo de Vídeo",
        ["consulta", "cirurgia", "fisioterapia", "triagem_violencia"],
        key="vid_type",
    )

    if vid_file and st.button("Analisar Vídeo", type="primary"):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(vid_file.name).suffix
        ) as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        with st.spinner("Analisando vídeo com YOLOv8..."):
            from src.pipelines.video_pipeline import VideoPipeline
            from src.agents.video_agent import VideoType

            pipeline = VideoPipeline()
            report, alerts = pipeline.run(
                video_path=tmp_path,
                video_type=VideoType(vid_type),
                export_report=False,
            )

            st.success(f"Análise concluída! {report.frames_analyzed} frames analisados.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Detecções", len(report.detections))
            col2.metric("Anomalias", report.anomaly_count)
            col3.metric("Duração", f"{report.video_duration_seconds:.1f}s")

            if report.detections:
                df = pd.DataFrame([
                    {
                        "Timestamp (s)": f"{d.timestamp_seconds:.1f}",
                        "Classe": d.class_name,
                        "Confiança": f"{d.confidence:.2f}",
                        "Anomalia": "⚠" if d.is_anomaly else "✓",
                    }
                    for d in report.detections
                ])
                st.dataframe(df, use_container_width=True)

            if alerts:
                st.subheader("Alertas Gerados")
                for a in alerts:
                    st.warning(f"[{a.severity.upper()}] {a.description}")

with tab3:
    st.header("Análise de Áudio de Consulta")
    st.markdown("Transcrição, análise emocional e detecção de indicadores de risco na voz.")

    aud_file = st.file_uploader(
        "Upload de Áudio",
        type=["wav", "mp3", "m4a", "ogg"],
        key="audio_only",
    )
    aud_type = st.selectbox(
        "Tipo de Consulta",
        ["ginecologica", "pre_natal", "pos_parto", "vitima_violencia"],
        key="aud_type",
    )

    if aud_file and st.button("Analisar Áudio", type="primary"):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(aud_file.name).suffix
        ) as tmp:
            tmp.write(aud_file.read())
            tmp_path = tmp.name

        with st.spinner("Transcrevendo e analisando áudio..."):
            from src.pipelines.audio_pipeline import AudioPipeline
            from src.agents.audio_agent import AudioConsultationType

            pipeline = AudioPipeline()
            report, alerts = pipeline.run(
                audio_path=tmp_path,
                consultation_type=AudioConsultationType(aud_type),
                export_report=False,
            )

            st.success("Análise concluída!")

            col1, col2, col3 = st.columns(3)
            risk_color = "inverse" if report.risk_level in ("alto", "crítico") else "normal"
            col1.metric("Score de Risco", f"{report.risk_score:.2f}")
            col2.metric("Nível", report.risk_level.upper(), delta_color=risk_color)
            col3.metric("Duração", f"{report.duration_seconds:.1f}s")

            st.subheader("Transcrição")
            st.text_area("Texto transcrito", report.transcription, height=150)

            if report.emotions:
                st.subheader("Emoções Detectadas")
                emotion_df = pd.DataFrame([
                    {"Início (s)": f"{e.start_seconds:.1f}", "Emoção": e.emotion, "Confiança": f"{e.confidence:.2f}"}
                    for e in report.emotions
                ])
                st.dataframe(emotion_df, use_container_width=True)

            if report.risk_factors:
                st.subheader("Fatores de Risco Textuais")
                for f in report.risk_factors:
                    st.warning(f"Indicador: {f}")

            if alerts:
                st.subheader("Alertas Gerados")
                for a in alerts:
                    st.warning(f"[{a.risk_level.upper()}] {a.evidence}")

with tab4:
    st.header("Monitoramento de Sinais Vitais")
    st.markdown("Detecção de anomalias em sinais vitais com Isolation Forest e Z-Score.")

    col1, col2 = st.columns(2)

    with col1:
        signal_type = st.selectbox(
            "Tipo de Sinal",
            [
                "pressao_sistolica", "pressao_diastolica",
                "batimentos_fetais", "batimentos_maternos",
                "temperatura", "saturacao_oxigenio",
                "nivel_hormonal", "glicose",
            ],
        )
        value = st.number_input("Valor", value=120.0, step=0.1)

    with col2:
        unit = st.text_input("Unidade", value="mmHg")
        is_gestational = st.checkbox("Faixas gestacionais", value=False)

    if st.button("Verificar Sinal Vital", type="primary"):
        from src.agents.anomaly_agent import (
            AnomalyDetectionAgent,
            SignalType,
            VitalSignRecord,
        )

        agent = AnomalyDetectionAgent(use_gestational_ranges=is_gestational)
        record = VitalSignRecord(
            timestamp=datetime.now().timestamp(),
            signal_type=SignalType(signal_type),
            value=value,
            unit=unit,
            patient_id="PAC-DEMO",
        )

        result = agent.add_record(record)

        if result:
            if result.is_anomaly:
                st.error(
                    f"ANOMALIA DETECTADA! Score: {result.anomaly_score:.2f} "
                    f"(Severidade: {result.severity.upper()})"
                )
            else:
                st.success(f"Valor dentro do esperado. Score: {result.anomaly_score:.2f}")

            st.metric(
                "Desvio do esperado",
                f"{result.deviation_from_expected:.2f}",
                delta=f"Range: {result.expected_range}",
            )

    st.markdown("---")
    st.subheader("Simular Série Temporal")

    n_points = st.slider("Número de pontos", 10, 200, 50)
    noise_level = st.slider("Nível de ruído", 0.01, 0.5, 0.1)

    if st.button("Gerar e Analisar Série Temporal"):
        timestamps = np.linspace(0, 100, n_points)
        base_value = 120.0
        noise = np.random.normal(0, noise_level * 20, n_points)

        values = base_value + noise
        anomaly_idx = np.random.choice(n_points, size=max(1, int(n_points * 0.05)), replace=False)
        values[anomaly_idx] += np.random.choice([-1, 1], len(anomaly_idx)) * np.random.uniform(30, 60, len(anomaly_idx))

        agent = AnomalyDetectionAgent(use_gestational_ranges=False)
        results = []

        for t, v in zip(timestamps, values):
            record = VitalSignRecord(
                timestamp=float(t),
                signal_type=SignalType.BLOOD_PRESSURE_SYSTOLIC,
                value=float(v),
                unit="mmHg",
                patient_id="PAC-DEMO",
            )
            result = agent.add_record(record)
            if result:
                results.append(result)

        df = pd.DataFrame([
            {
                "Timestamp": r.record.timestamp,
                "Valor": r.record.value,
                "Score": r.anomaly_score,
                "Anomalia": "⚠" if r.is_anomaly else "✓",
                "Severidade": r.severity,
            }
            for r in results
        ])

        st.subheader("Resultados da Série Temporal")
        st.dataframe(df.tail(20), use_container_width=True)

        anomalies_detected = sum(1 for r in results if r.is_anomaly)
        st.metric("Anomalias detectadas", f"{anomalies_detected}/{len(results)}")

        st.subheader("Gráfico")
        chart_df = pd.DataFrame({
            "timestamp": [r.record.timestamp for r in results],
            "valor": [r.record.value for r in results],
            "anomalia": [r.anomaly_score > 0.5 for r in results],
        })
        st.line_chart(chart_df.set_index("timestamp")["valor"])

if __name__ == "__main__":
    st.sidebar.info("Sistema de Monitoramento Multimodal - Fase 4 Tech Challenge")
