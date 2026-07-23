# 🏥 Sistema de Monitoramento Multimodal para Saúde da Mulher

**Fase 4 - Tech Challenge**

Sistema de IA para análise e fusão de dados multimodais (vídeo, áudio e sinais vitais) com foco em detecção precoce de riscos em saúde materna e ginecológica, identificação de sinais de violência doméstica, monitoramento de bem-estar psicológico e alertas em tempo real para equipes médicas.

---

## Funcionalidades Principais

### Análise de Vídeo Clínico
- Detecção de sangramento anômalo em procedimentos ginecológicos
- Identificação de instrumentos cirúrgicos e áreas críticas (útero, ovários, mamas)
- Reconhecimento de expressões faciais de desconforto e medo
- Triagem de objetos suspeitos indicativos de automutilação
- **Modelo:** YOLOv8 customizado (com suporte a fine-tuning)

### Análise de Áudio de Consultas
- Transcrição automática de consultas (português)
- Detecção de emoções na voz (tristeza, medo, raiva, ansiedade)
- Identificação de indicadores verbais de risco (violência, depressão, abuso)
- Análise de hesitação e padrões de silêncio
- **Modelos:** OpenAI Whisper + Wav2Vec2 (HuggingFace)

### Monitoramento de Sinais Vitais
- Detecção de anomalias em tempo real
- Faixas de referência específicas para gestantes
- Análise de prescrições hormonais
- **Métodos:** Z-Score + Rolling Statistics + Isolation Forest

### Fusão Multimodal
- Score de risco unificado (vídeo + áudio + sinais vitais)
- Correlação de indicadores entre modalidades
- Recomendações automáticas para equipe médica

### Integração Azure
- **Speech Services:** Transcrição e análise de sentimentos
- **AI Vision:** Análise complementar de imagens médicas
- **Blob Storage:** Armazenamento seguro e criptografado (LGPD)
- **OpenAI:** Análise avançada de texto clínico

---

## Estrutura do Projeto

```
tech-4/
├── main.py                         # CLI principal (Typer) - ponto de entrada
├── requirements.txt                # Dependências
├── .env.example                    # Template de variáveis de ambiente
├── AGENTS.md                       # Arquitetura de agentes autônomos
├── README.md                       # Este arquivo
├── src/
│   ├── config.py                   # Configuração centralizada (Pydantic)
│   ├── app.py                      # Interface Streamlit
│   ├── models/
│   │   ├── yolo_detector.py        # Detector YOLOv8 customizado
│   │   └── audio_analyzer.py       # Transcrição + análise emocional
│   ├── agents/
│   │   ├── video_agent.py          # Agente de análise de vídeo
│   │   ├── audio_agent.py          # Agente de análise de áudio
│   │   ├── anomaly_agent.py        # Agente de anomalias vitais
│   │   └── fusion_agent.py         # Agente de fusão multimodal
│   ├── pipelines/
│   │   ├── video_pipeline.py       # Pipeline de vídeo
│   │   ├── audio_pipeline.py       # Pipeline de áudio
│   │   └── multimodal_pipeline.py  # Pipeline multimodal completo
│   ├── services/
│   │   ├── azure_services.py       # Azure Cognitive Services
│   │   └── alert_service.py        # Sistema de alertas
│   └── utils/
│       ├── file_handler.py         # Manipulação de arquivos
│       └── report_generator.py     # Gerador de relatórios
├── data/
│   ├── videos/                     # Vídeos para análise
│   ├── audios/                     # Áudios para análise
│   └── reports/                    # Relatórios gerados
│       └── alerts/                 # Logs de alertas
├── tests/
├── models/                         # Modelos treinados
├── notebooks/                      # Jupyter notebooks
└── edital/
    └── fase-4-edital.md
```

---

## Requisitos

- Python 3.10+
- CUDA 11.8+ (opcional, para GPU)
- Chaves Azure (opcional, para serviços cloud)

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd tech-4
```

### 2. Crie o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/Mac
# ou
.venv\Scripts\activate       # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas chaves:

```env
# OpenAI Whisper (obrigatório para transcrição)
OPENAI_API_KEY=sua_chave_aqui

# Azure (opcional - habilita serviços cloud)
AZURE_SPEECH_KEY=sua_chave_azure_speech
AZURE_SPEECH_REGION=brazilsouth
AZURE_VISION_KEY=sua_chave_azure_vision
AZURE_VISION_ENDPOINT=https://seu-endpoint.cognitiveservices.azure.com/

# Alertas por email (opcional)
ALERT_EMAIL_FROM=alertas@hospital.com
ALERT_EMAIL_TO=equipe_medica@hospital.com
ALERT_EMAIL_PASSWORD=sua_senha_app
```

---

## Uso

### CLI (Linha de Comando)

```bash
# Ver ajuda geral
python main.py --help

# Análise de vídeo cirúrgico
python main.py video data/videos/cirurgia.mp4 --type cirurgia

# Análise de áudio de consulta pós-parto
python main.py audio data/audios/consulta.wav --type pos_parto

# Pipeline multimodal completo
python main.py multimodal PAC-001 \
    --video data/videos/consulta.mp4 \
    --audio data/audios/consulta.wav \
    --video-type consulta \
    --audio-type ginecologica

# Interface Streamlit interativa
python main.py demo
```

### Python API

```python
from src.pipelines.multimodal_pipeline import MultimodalPipeline

pipeline = MultimodalPipeline()

assessment = pipeline.run(
    patient_id="PAC-001",
    video_path="data/videos/consulta.mp4",
    audio_path="data/audios/consulta.wav",
    video_type="consulta",
    consultation_type="ginecologica",
)

print(f"Score de risco: {assessment.overall_risk_score:.2f}")
print(f"Nível: {assessment.overall_risk_level}")
print(f"Recomendações: {assessment.recommendations}")
```

### Análise Individual

```python
# Apenas vídeo
from src.pipelines.video_pipeline import VideoPipeline
from src.agents.video_agent import VideoType

pipeline = VideoPipeline()
report, alerts = pipeline.run("video.mp4", VideoType.SURGERY)

# Apenas áudio
from src.pipelines.audio_pipeline import AudioPipeline
from src.agents.audio_agent import AudioConsultationType

pipeline = AudioPipeline()
report, alerts = pipeline.run("audio.wav", AudioConsultationType.POSTPARTUM)

# Apenas sinais vitais
from src.agents.anomaly_agent import AnomalyDetectionAgent, SignalType, VitalSignRecord

agent = AnomalyDetectionAgent(use_gestational_ranges=True)
record = VitalSignRecord(0.0, SignalType.FETAL_HEART_RATE, 175.0, "bpm", "PAC-001")
result = agent.add_record(record)
print(f"Anomalia: {result.is_anomaly}, Score: {result.anomaly_score:.2f}")
```

---

## Tipos de Análise Suportados

### Vídeos
| Tipo | Descrição | Classes Detectadas |
|------|-----------|-------------------|
| `cirurgia` | Cirurgias ginecológicas | Sangramento, instrumentos, áreas críticas |
| `consulta` | Consultas médicas | Expressões faciais, desconforto |
| `fisioterapia` | Sessões de fisioterapia | Movimentos, postura |
| `triagem_violencia` | Triagem de violência | Linguagem corporal, objetos suspeitos |

### Áudios
| Tipo | Descrição | Indicadores |
|------|-----------|-------------|
| `ginecologica` | Consulta ginecológica | Hesitação, desconforto |
| `pre_natal` | Acompanhamento pré-natal | Ansiedade gestacional |
| `pos_parto` | Consulta pós-parto | Depressão pós-parto |
| `vitima_violencia` | Atendimento a vítimas | Trauma vocal, medo |

### Sinais Vitais
| Sinal | Unidade | Faixa Normal | Faixa Gestacional |
|-------|---------|-------------|-------------------|
| Pressão Sistólica | mmHg | 90-140 | 90-135 |
| Pressão Diastólica | mmHg | 60-90 | 60-85 |
| Batimentos Fetais | bpm | 110-160 | 120-160 |
| Batimentos Maternos | bpm | 60-100 | 65-110 |
| Temperatura | °C | 35.5-37.5 | 35.5-37.5 |
| Saturação O2 | % | 95-100 | 95-100 |
| Glicose | mg/dL | 70-140 | 70-130 |

---

## Níveis de Risco e Alertas

| Nível | Score | Ação |
|-------|-------|------|
| **Crítico** | > 0.80 | Notificação imediata + protocolo de emergência |
| **Alto** | > 0.60 | Alerta à equipe + avaliação em 24h |
| **Médio** | > 0.30 | Notificação + agendamento na semana |
| **Baixo** | ≤ 0.30 | Rotina normal |

Alertas são enviados via:
- **Email** (HTML formatado) para equipe médica
- **JSON** (log estruturado) para integração com sistemas hospitalares
- **Log** (console/arquivo) para monitoramento operacional

---

## Relatórios

Relatórios são gerados automaticamente em `data/reports/` nos formatos:

- **JSON:** Estruturado para integração com sistemas (HL7/FHIR)
- **Markdown:** Legível para documentação clínica e auditoria

Tipos de relatório:
- `video_report_*.json/md` - Análise de vídeo
- `audio_report_*.json/md` - Análise de áudio
- `fusion_report_*.json/md` - Fusão multimodal

---

## Fine-tuning do YOLOv8

Para treinar o modelo com dataset customizado de imagens clínicas:

```python
from src.models.yolo_detector import YOLODetector

detector = YOLODetector()
detector.finetune(
    data_yaml="models/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)
```

---

## Demo Streamlit

A interface Streamlit oferece:

- Upload interativo de vídeos e áudios
- Visualização de resultados em tempo real
- Simulação de séries temporais de sinais vitais
- Dashboards com scores e alertas

```bash
streamlit run src/app.py
# ou
python main.py demo
```

---

## Licença

Este projeto é parte do Tech Challenge Fase 4 - FIAP.

---

**Autores:** Equipe Tech Challenge  
**Disciplinas:** Todas as disciplinas da Fase 4  
**Peso:** 90% da nota de todas as disciplinas
# tech-challeng-4-monitoramente
