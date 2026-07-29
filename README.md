# Sistema de Monitoramento Multimodal para Saúde da Mulher

**Fase 4 - Tech Challenge**

Sistema de IA para análise e fusão de dados multimodais (vídeo, áudio e sinais vitais) com foco em detecção precoce de riscos em saúde materna e ginecológica.

---

## Funcionalidades

### Análise de Vídeo
- **3 modelos YOLOv8 especializados**: emoções (8 classes, 6.586 img), sangramento (1 classe, 82 img), instrumentos cirúrgicos (21 classes, 1.530 img)
- **Azure AI Vision**: descrição de cena cirúrgica via API (captions e tags)
- Modos: batch com relatório, `live` com bounding boxes, `live-azure` com Azure Vision

### Análise de Áudio
- **Azure Speech** (primário) + **Whisper** (fallback): transcrição contínua em português
- **Wav2Vec2**: detecção de emoções na voz (medo, tristeza, raiva, etc.)
- Indicadores textuais de risco: depressão, ansiedade, violência doméstica

### Sinais Vitais
- **Z-Score + Isolation Forest + faixas de referência**: anomalias em 9 indicadores
- Faixas específicas para gestantes

### Fusão Multimodal
- Score unificado: 35% vídeo + 35% áudio + 30% sinais vitais
- Correlação cruzada de riscos entre modalidades
- Recomendações automáticas para equipe médica

### Infraestrutura
- **Azure Storage**: upload de vídeos, áudios e relatórios
- **Alertas por email**: HTML com scores e recomendações (SSL/TLS)
- **Streamlit**: interface interativa com barras de progresso e bucket browser

---

## Estrutura

```
tech-4/
├── main.py                         # CLI: video, audio, multimodal, live, live-azure, demo
├── train.py                        # Fine-tuning (--dataset emotions|blood|instruments)
├── convert_obb.py                  # Converte labels OBB → bbox (instrumentos)
├── requirements.txt
├── .env.example
├── AGENTS.md
├── README.md
├── VIDEO_SCRIPT.md                 # Roteiro do vídeo
├── src/
│   ├── config.py
│   ├── app.py                      # Streamlit
│   ├── models/
│   │   ├── yolo_detector.py        # YOLOv8 multi-modelo + live + Azure live
│   │   └── audio_analyzer.py       # Azure Speech + Whisper + Wav2Vec2
│   ├── agents/
│   │   ├── video_agent.py
│   │   ├── audio_agent.py
│   │   ├── anomaly_agent.py
│   │   └── fusion_agent.py
│   ├── pipelines/
│   │   ├── video_pipeline.py
│   │   ├── audio_pipeline.py
│   │   └── multimodal_pipeline.py
│   ├── services/
│   │   ├── azure_services.py       # Speech, Vision, Storage
│   │   └── alert_service.py        # Email + JSON + Storage
│   └── utils/
│       ├── file_handler.py
│       └── report_generator.py
├── data/
│   ├── videos/
│   ├── audios/
│   ├── dataset/
│   │   ├── emotions/               # 8 classes de expressão facial
│   │   ├── blood/                  # 1 classe — sangramento
│   │   └── surgery_instruments/    # 21 instrumentos cirúrgicos
│   └── reports/
├── models/                         # Modelos treinados (.pt)
├── relatorio/                      # Relatório técnico
└── tests/
```

---

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Treinamento

### Datasets (Roboflow Universe)

| Dataset | Imagens | Classes |
|---------|---------|---------|
| `emotions` | 6.586 treino / 1.873 val | 8 expressões faciais |
| `blood` | 82 treino / 81 val | 1 — sangramento |
| `surgery_instruments` | 1.530 treino / 145 val | 21 instrumentos |

### Treinar os 3 modelos

```bash
# Converter labels OBB → bbox (só precisa uma vez)
python convert_obb.py

# Treinar
python train.py --dataset emotions --epochs 30 --batch 8
python train.py --dataset blood --epochs 50 --batch 8
python train.py --dataset instruments --epochs 30 --batch 8
```

Os modelos são salvos em `models/emotions.pt`, `models/blood.pt` e `models/instruments.pt`. O `YOLODetector` carrega os 3 automaticamente.

### Data Augmentation

Rotação ±15°, variação HSV, scale 0.6, mosaic 80%, mixup 15%, random erasing 20%, flip horizontal/vertical — simulando condições reais de centro cirúrgico.

---

## Uso

### CLI

```bash
# Análise batch
python main.py video data/videos/cesaria.mp4 --type cirurgia

# Vídeo anotado com bounding boxes
python main.py video data/videos/cesaria.mp4 --type cirurgia --annotate

# Live com YOLOv8 (3 modelos)
python main.py live data/videos/cesaria.mp4 --conf 0.20

# Live com Azure Vision
python main.py live-azure data/videos/cesaria.mp4

# Análise de áudio
python main.py audio data/audios/consulta.wav --type pos_parto

# Pipeline multimodal completo
python main.py multimodal PAC-001 --video video.mp4 --audio audio.wav

# Interface Streamlit
python main.py demo
```

### Streamlit

4 abas: Pipeline Multimodal (com sinais vitais), Análise de Vídeo (com keyframes), Análise de Áudio (com barra de progresso), Sinais Vitais.

---

## Serviços Azure

| Serviço | Variável | Função | Fallback |
|---------|----------|--------|----------|
| Speech | `AZURE_SPEECH_KEY` | Transcrição de áudio | Whisper local |
| Vision | `AZURE_VISION_KEY` + `_ENDPOINT` | Descrição de cena | YOLOv8 local |
| Storage | `AZURE_STORAGE_CONNECTION_STRING` | Upload de mídia | Arquivo local |

---

## Alertas por Email

```env
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_FROM=alertas@hospital.com
ALERT_EMAIL_TO=equipe_medica@hospital.com
ALERT_EMAIL_PASSWORD=senha_app
```

Suporte a SSL (465) e TLS (587). Email HTML com scores, correlações e recomendações.

---

## Níveis de Risco

| Nível | Score | Ação |
|-------|-------|------|
| Crítico | > 0.80 | Email URGENTE + protocolo de emergência |
| Alto | > 0.60 | Email de alerta + avaliação em 24h |
| Médio | > 0.30 | Notificação + agendamento |
| Baixo | ≤ 0.30 | Rotina normal |
