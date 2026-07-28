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
    azure_status = []
    try:
        from src.config import settings as app_settings
        if app_settings.azure.speech_key:
            azure_status.append("Azure Speech ✅")
        else:
            azure_status.append("Azure Speech ❌")
        if app_settings.azure.storage_connection_string:
            azure_status.append("Azure Storage ✅")
        else:
            azure_status.append("Azure Storage ❌")
        if app_settings.azure.vision_key and app_settings.azure.vision_endpoint:
            azure_status.append("Azure Vision ✅")
        else:
            azure_status.append("Azure Vision ❌")
    except Exception:
        azure_status = ["Azure: verifique .env"]

    for s in azure_status:
        st.markdown(f"- {s}")

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

    with st.expander("🫀 Sinais Vitais (opcional)", expanded=False):
        vitals_data = []
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)

        with vcol1:
            vs_pressao_sis = st.number_input("Pressão Sistólica", value=0.0, step=0.1, help="mmHg — deixe 0 para ignorar")
            vs_pressao_dia = st.number_input("Pressão Diastólica", value=0.0, step=0.1, help="mmHg")
            vs_temp = st.number_input("Temperatura", value=0.0, step=0.1, help="°C")

        with vcol2:
            vs_fetal_hr = st.number_input("Bat. Fetais", value=0.0, step=1.0, help="bpm")
            vs_maternal_hr = st.number_input("Bat. Maternos", value=0.0, step=1.0, help="bpm")
            vs_o2 = st.number_input("Saturação O₂", value=0.0, step=0.1, help="%")

        with vcol3:
            vs_glucose = st.number_input("Glicose", value=0.0, step=1.0, help="mg/dL")
            vs_hormone = st.number_input("Nível Hormonal", value=0.0, step=0.1, help="Z-score")
            vs_weight = st.number_input("Peso", value=0.0, step=0.1, help="kg")

        with vcol4:
            st.caption("Valores > 0 serão analisados")
            is_gestational = st.checkbox("Faixas gestacionais", value=False, key="gest_multimodal")

        signals = [
            ("pressao_sistolica", vs_pressao_sis, "mmHg"),
            ("pressao_diastolica", vs_pressao_dia, "mmHg"),
            ("batimentos_fetais", vs_fetal_hr, "bpm"),
            ("batimentos_maternos", vs_maternal_hr, "bpm"),
            ("temperatura", vs_temp, "°C"),
            ("saturacao_oxigenio", vs_o2, "%"),
            ("glicose", vs_glucose, "mg/dL"),
            ("nivel_hormonal", vs_hormone, "Z-score"),
            ("peso", vs_weight, "kg"),
        ]
        for sig_type, val, unit in signals:
            if val > 0:
                vitals_data.append({
                    "timestamp": 0.0,
                    "signal_type": sig_type,
                    "value": float(val),
                    "unit": unit,
                })

    if st.button("Executar Pipeline Multimodal", type="primary", use_container_width=True):
        if not video_file and not audio_file and not vitals_data:
            st.warning("Faça upload de pelo menos um arquivo (vídeo ou áudio) ou preencha sinais vitais.")
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

                    video_progress = None
                    video_text = None
                    audio_progress = None
                    audio_text = None

                    if video_file:
                        video_progress = st.progress(0, "Processando vídeo...")
                        video_text = st.empty()

                        def update_video_progress(current, total):
                            pct = min(1.0, current / max(total, 1))
                            video_progress.progress(pct, f"Processando vídeo... {current}/{total} frames")
                            video_text.text(f"Vídeo: {pct:.0%} concluído")
                    else:
                        update_video_progress = None

                    if audio_file:
                        audio_progress = st.progress(0, "Processando áudio...")
                        audio_text = st.empty()

                        def update_audio_progress(current, total):
                            pct = min(1.0, current / max(total, 1))
                            audio_progress.progress(pct, f"Analisando emoções... {current}/{total} segmentos")
                            audio_text.text(f"Áudio: {pct:.0%} concluído")
                    else:
                        update_audio_progress = None

                    pipeline = MultimodalPipeline()
                    assessment = pipeline.run(
                        patient_id=patient_id,
                        video_path=video_path,
                        audio_path=audio_path,
                        vital_signs=vitals_data if vitals_data else None,
                        video_type=video_type,
                        consultation_type=consultation_type,
                        export_report=True,
                        progress_callback=update_video_progress,
                        audio_progress_callback=update_audio_progress,
                        is_gestational=is_gestational,
                    )

                    if video_progress:
                        video_progress.empty()
                    if video_text:
                        video_text.empty()
                    if audio_progress:
                        audio_progress.empty()
                    if audio_text:
                        audio_text.empty()

                    st.success("Pipeline concluído com sucesso!")

                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Score de Risco", f"{assessment.overall_risk_score:.2f}")
                    col_b.metric("Nível", assessment.overall_risk_level.upper())
                    total_alerts = len(assessment.video_alerts) + len(assessment.audio_alerts) + len(assessment.vitals_alerts)
                    col_c.metric("Alertas", total_alerts)

                    if assessment.video_alerts:
                        with st.expander(f"🎬 Alertas de Vídeo ({len(assessment.video_alerts)})", expanded=False):
                            for a in assessment.video_alerts:
                                sev = a.get("severidade", "baixa")
                                icon = {"crítica": "🔴", "alta": "🟠", "média": "🟡", "baixa": "🟢"}.get(sev, "⚪")
                                ts = a.get("timestamp_segundos", 0)
                                st.warning(
                                    f"{icon} **[{sev.upper()}]** {a.get('tipo_anomalia', '-')} "
                                    f"({ts:.1f}s, conf: {a.get('confianca', 0):.2f})\n\n"
                                    f"{a.get('descricao', '')}"
                                )

                    if assessment.audio_alerts:
                        with st.expander(f"🎙️ Alertas de Áudio ({len(assessment.audio_alerts)})", expanded=False):
                            for a in assessment.audio_alerts:
                                sev = a.get("nivel_risco", "baixo")
                                icon = {"crítico": "🔴", "alto": "🟠", "médio": "🟡", "baixo": "🟢"}.get(sev, "⚪")
                                st.warning(
                                    f"{icon} **[{sev.upper()}]** {a.get('tipo_alerta', '-')} "
                                    f"(score: {a.get('score_risco', 0):.2f})\n\n"
                                    f"{a.get('evidencia', '')}"
                                )

                    if assessment.vitals_alerts:
                        with st.expander(f"🫀 Alertas de Sinais Vitais ({len(assessment.vitals_alerts)})", expanded=False):
                            for a in assessment.vitals_alerts:
                                sev = a.get("severidade", "baixa")
                                icon = {"crítica": "🔴", "alta": "🟠", "média": "🟡", "baixa": "🟢"}.get(sev, "⚪")
                            st.warning(
                                f"{icon} **[{sev.upper()}]** {a.get('tipo', '-')} "
                                f"= {a.get('valor', '-')} {a.get('unidade', '')} "
                                f"(score: {a.get('score', 0):.2f})"
                            )

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

            progress_bar = st.progress(0, "Analisando frames...")
            progress_text = st.empty()

            def update_progress(current, total):
                pct = min(1.0, current / max(total, 1))
                progress_bar.progress(pct, f"Analisando frames... {current}/{total}")
                progress_text.text(f"{pct:.0%} concluído")

            pipeline = VideoPipeline()
            report, alerts = pipeline.run(
                video_path=tmp_path,
                video_type=VideoType(vid_type),
                export_report=False,
                progress_callback=update_progress,
            )

            progress_bar.empty()
            progress_text.empty()

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

        from src.pipelines.audio_pipeline import AudioPipeline
        from src.agents.audio_agent import AudioConsultationType

        audio_progress = st.progress(0, "Processando áudio...")
        audio_text = st.empty()

        def update_audio_progress(current, total):
            pct = min(1.0, current / max(total, 1))
            audio_progress.progress(pct, f"Analisando emoções... {current}/{total} segmentos")
            audio_text.text(f"{pct:.0%} concluído")

        pipeline = AudioPipeline()
        report, alerts = pipeline.run(
            audio_path=tmp_path,
            consultation_type=AudioConsultationType(aud_type),
            export_report=False,
            progress_callback=update_audio_progress,
        )

        audio_progress.empty()
        audio_text.empty()

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

    is_gestational_v = st.checkbox("Faixas gestacionais", value=False, key="gest_vitals")

    col1, col2, col3 = st.columns(3)

    with col1:
        vs_pressao_sis = st.number_input("Pressão Sistólica", value=0.0, step=0.1, help="mmHg", key="vs_ps")
        vs_pressao_dia = st.number_input("Pressão Diastólica", value=0.0, step=0.1, help="mmHg", key="vs_pd")
        vs_temp = st.number_input("Temperatura", value=0.0, step=0.1, help="°C", key="vs_tp")

    with col2:
        vs_fetal_hr = st.number_input("Bat. Fetais", value=0.0, step=1.0, help="bpm", key="vs_fh")
        vs_maternal_hr = st.number_input("Bat. Maternos", value=0.0, step=1.0, help="bpm", key="vs_mh")
        vs_o2 = st.number_input("Saturação O₂", value=0.0, step=0.1, help="%", key="vs_o2")

    with col3:
        vs_glucose = st.number_input("Glicose", value=0.0, step=1.0, help="mg/dL", key="vs_gl")
        vs_hormone = st.number_input("Nível Hormonal", value=0.0, step=0.1, help="Z-score", key="vs_nh")
        vs_weight = st.number_input("Peso", value=0.0, step=0.1, help="kg", key="vs_wt")

    if st.button("Verificar Sinais Vitais", type="primary"):
        from src.agents.anomaly_agent import (
            AnomalyDetectionAgent,
            SignalType,
            VitalSignRecord,
        )

        signals = [
            ("pressao_sistolica", vs_pressao_sis, "mmHg"),
            ("pressao_diastolica", vs_pressao_dia, "mmHg"),
            ("batimentos_fetais", vs_fetal_hr, "bpm"),
            ("batimentos_maternos", vs_maternal_hr, "bpm"),
            ("temperatura", vs_temp, "°C"),
            ("saturacao_oxigenio", vs_o2, "%"),
            ("glicose", vs_glucose, "mg/dL"),
            ("nivel_hormonal", vs_hormone, "Z-score"),
            ("peso", vs_weight, "kg"),
        ]

        agent = AnomalyDetectionAgent(use_gestational_ranges=is_gestational_v)
        results_output = []

        for sig_type, val, unit in signals:
            if val > 0:
                record = VitalSignRecord(
                    timestamp=datetime.now().timestamp(),
                    signal_type=SignalType(sig_type),
                    value=float(val),
                    unit=unit,
                    patient_id="PAC-DEMO",
                )
                result = agent.add_record(record)
                if result:
                    results_output.append(result)

        if results_output:
            anomalies = [r for r in results_output if r.is_anomaly]
            if anomalies:
                st.error(f"{len(anomalies)} anomalia(s) detectada(s) em {len(results_output)} sinais!")
                for r in anomalies:
                    st.warning(
                        f"**{r.record.signal_type.value}**: {r.record.value} {r.record.unit} "
                        f"— score: {r.anomaly_score:.2f} [{r.severity.upper()}] "
                        f"(range esperado: {r.expected_range})"
                    )
            else:
                st.success(f"Todos os {len(results_output)} sinais dentro do esperado.")
        else:
            st.info("Preencha pelo menos um sinal vital com valor > 0.")

    st.markdown("---")
    st.subheader("Simular Série Temporal")

    n_points = st.slider("Número de pontos", 10, 200, 50)
    noise_level = st.slider("Nível de ruído", 0.01, 0.5, 0.1)

    if st.button("Gerar e Analisar Série Temporal"):
        from src.agents.anomaly_agent import (
            AnomalyDetectionAgent,
            SignalType,
            VitalSignRecord,
        )

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
