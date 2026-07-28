# Sistema de Monitoramento Multimodal para Saúde da Mulher

**Fase 4 - Tech Challenge**

Sistema de IA para análise e fusão de dados multimodais (vídeo, áudio e sinais vitais) com foco em detecção precoce de riscos em saúde materna e ginecológica.

---

## Funcionalidades

### Análise de Vídeo
- **YOLOv8**: modelo treinado em 11.474 imagens com 30 classes (sangramento, 21 instrumentos cirúrgicos, 8 expressões faciais)
- **Azure AI Vision**: descrição de cena cirúrgica via API (captions e tags)
- Modos: batch com relatório, live com bounding boxes, live com Azure Vision

### Análise de Áudio
- **Azure Speech** (primário) + **Whisper** (fallback): transcrição em português
- **Wav2Vec2**: detecção de emoções na voz (medo, tristeza, raiva, etc.)
- Indicadores textuais de risco: depressão, ansiedade, violência doméstica

### Sinais Vitais
- **Z-Score + Isolation Forest**: detecção de anomalias em 9 indicadores
- Faixas de referência específicas para gestantes

### Fusão Multimodal
- Score unificado: 35% vídeo + 35% áudio + 30% sinais vitais
- Correlação de riscos entre modalidades
- Recomendações automáticas para equipe médica

### Infraestrutura
- **Azure Storage**: upload automático de vídeos, áudios e relatórios
- **Alertas por email**: HTML com scores, correlações e recomendações
- **Streamlit**: interface web interativa com barras de progresso

---

## Estrutura

```
tech-4/
├── main.py                         # CLI (Typer): video, audio, multimodal, live, live-azure, demo
├── train.py                        # Fine-tuning do YOLOv8
├── generate_dataset.py             # Gerador de dataset sintético
├── merge_datasets.py               # Unifica datasets Roboflow (blood + instruments + emotions)
├── convert_obb_dataset.py          # Converte OBB → bbox YOLO
├── download_surgical_model.py      # Download de modelo cirúrgico (Roboflow/HF)
├── requirements.txt
├── .env.example
├── AGENTS.md                       # Arquitetura dos agentes
├── README.md
├── VIDEO_SCRIPT.md                 # Roteiro do vídeo de demonstração
├── src/
│   ├── config.py                   # Configuração centralizada (Pydantic)
│   ├── app.py                      # Interface Streamlit
│   ├── models/
│   │   ├── yolo_detector.py        # YOLOv8: detecção, live, Azure live
│   │   └── audio_analyzer.py       # Whisper + Wav2Vec2 + Azure Speech
│   ├── agents/
│   │   ├── video_agent.py          # Agente de vídeo
│   │   ├── audio_agent.py          # Agente de áudio
│   │   ├── anomaly_agent.py        # Agente de anomalias vitais
│   │   └── fusion_agent.py         # Fusão multimodal
│   ├── pipelines/
│   │   ├── video_pipeline.py
│   │   ├── audio_pipeline.py
│   │   └── multimodal_pipeline.py
│   ├── services/
│   │   ├── azure_services.py       # Azure Speech, Vision, Storage
│   │   └── alert_service.py        # Email + JSON + Azure Storage
│   └── utils/
│       ├── file_handler.py
│       └── report_generator.py
├── data/
│   ├── videos/                     # Vídeos para análise
│   ├── audios/                     # Áudios para análise
│   ├── dataset/                    # Datasets de treino
│   │   ├── blood/                  # 163 imagens — sangramento
│   │   ├── surgery_instruments/    # 1.675 imagens — 21 instrumentos
│   │   ├── emotions/               # 8.459 imagens — 8 emoções
│   │   └── unified/                # Dataset unificado (11.474 img, 30 classes)
│   └── reports/                    # Relatórios gerados
├── models/
│   └── yolov8_custom.pt            # Modelo treinado
└── tests/
```

---

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edite .env com suas chaves
```

---

## Treinamento do Modelo

### Dataset unificado (30 classes)

```bash
# 1. Unificar os 3 datasets
python merge_datasets.py

# 2. Treinar com augmentations para generalização cirúrgica
python train.py --config data/dataset/unified/data.yaml --epochs 30 --batch 8
```

O modelo treinado é salvo em `models/yolov8_custom.pt`.

### Métricas de validação (mAP50 = 0.84)

| Categoria | Classes | Melhor desempenho |
|-----------|---------|-------------------|
| Sangramento | 1 | mAP 0.59 |
| Instrumentos | 21 | bisturi (0.99), tesoura (0.995), pinça (0.97) |
| Emoções | 8 | medo (0.80), feliz (0.88), tristeza (0.70) |

---

## Uso

### CLI

```bash
# Análise batch com relatório
python main.py video data/videos/cesaria.mp4 --type cirurgia

# Vídeo anotado com bounding boxes
python main.py video data/videos/cesaria.mp4 --type cirurgia --annotate

# Live com YOLOv8 (bounding boxes em tempo real)
python main.py live data/videos/cesaria.mp4 --conf 0.20

# Live com Azure Vision (captions e tags)
python main.py live-azure data/videos/cesaria.mp4
python main.py live-azure data/videos/cesaria.mp4 --interval 2.0

# Análise de áudio
python main.py audio data/audios/consulta.wav --type pos_parto

# Pipeline multimodal completo (vídeo + áudio + sinais vitais)
python main.py multimodal PAC-001 --video video.mp4 --audio audio.wav

# Interface Streamlit
python main.py demo
```

### Streamlit

```bash
python main.py demo
```

4 abas: Pipeline Multimodal, Análise de Vídeo (com keyframes), Análise de Áudio (com progresso), Sinais Vitais.

Sidebar: status Azure, bucket browser, configuração de email.

---

## Serviços Azure

| Serviço | Variável .env | Função |
|---------|-------------|--------|
| Speech | `AZURE_SPEECH_KEY` | Transcrição de áudio |
| Vision | `AZURE_VISION_KEY` + `ENDPOINT` | Detecção de cena cirúrgica |
| Storage | `AZURE_STORAGE_CONNECTION_STRING` | Upload de mídia e relatórios |
| OpenAI | `AZURE_OPENAI_API_KEY` | Análise de sentimento |

### Fallback automático

Todos os serviços Azure têm fallback para modelos locais quando indisponíveis:
- Speech → Whisper local
- Vision → YOLOv8 local
- Storage → apenas arquivo local

---

## Alertas por Email

```env
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_FROM=alertas@hospital.com
ALERT_EMAIL_TO=equipe_medica@hospital.com
ALERT_EMAIL_PASSWORD=senha_app
```

Suporte a SSL (porta 465) e TLS (porta 587). Email HTML com scores, correlações e recomendações.

---

## Níveis de Risco

| Nível | Score | Ação |
|-------|-------|------|
| Crítico | > 0.80 | Email URGENTE + protocolo de emergência |
| Alto | > 0.60 | Email de alerta + avaliação em 24h |
| Médio | > 0.30 | Notificação + agendamento |
| Baixo | ≤ 0.30 | Rotina normal |
