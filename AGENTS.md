# Sistema de Monitoramento Multimodal para Saúde da Mulher - Fase 4

Arquitetura de agentes autônomos para análise multimodal de dados de saúde da mulher.

## Definição dos Agentes

```
                         +----------------------+
                         |   Dados Multimodais  |
                         |  (Vídeo/Áudio/Vitais)|
                         +----------+-----------+
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
          +-------------+  +-------------+  +---------------+
          | Video Agent |  | Audio Agent |  | Anomaly Agent |
          |  (YOLOv8)   |  | (Whisper +  |  |  (Z-Score +   |
          |             |  |  Wav2Vec2)  |  | Isolation For)|
          +------+------+  +------+------+  +-------+-------+
                 |                |                  |
                 +----------------+------------------+
                                  |
                                  v
                         +--------+--------+
                         |  Fusion Agent   |  → Score unificado de risco
                         +--------+--------+
                                  |
                                  v
                         +--------+--------+
                         |  Alert Service  |  → E-mail + JSON + Log
                         +-----------------+
                                  |
                                  v
                         +--------+--------+
                         | Equipe Médica   |
                         +-----------------+
```

## Estrutura do Projeto

```
tech-4/
├── main.py                         # CLI principal (Typer)
├── requirements.txt                # Dependências do projeto
├── .env.example                    # Template de variáveis de ambiente
├── AGENTS.md                       # Arquitetura de agentes (este arquivo)
├── README.md                       # Documentação detalhada
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuração centralizada (Pydantic Settings)
│   ├── app.py                      # Interface Streamlit para demonstração
│   ├── models/
│   │   ├── yolo_detector.py        # YOLOv8: detecção visual em vídeos clínicos
│   │   └── audio_analyzer.py       # Whisper + Wav2Vec2: transcrição e emoção vocal
│   ├── agents/
│   │   ├── video_agent.py          # Agente de análise de vídeo clínico
│   │   ├── audio_agent.py          # Agente de análise de áudio de consultas
│   │   ├── anomaly_agent.py        # Agente de anomalias em sinais vitais
│   │   └── fusion_agent.py         # Fusão multimodal com score unificado
│   ├── pipelines/
│   │   ├── video_pipeline.py       # Pipeline de vídeo end-to-end
│   │   ├── audio_pipeline.py       # Pipeline de áudio end-to-end
│   │   └── multimodal_pipeline.py  # Pipeline integrado (vídeo + áudio + vitais)
│   ├── services/
│   │   ├── azure_services.py       # Azure Cognitive Services (Speech, Vision, Storage)
│   │   └── alert_service.py        # Serviço de alertas (email, JSON, log)
│   └── utils/
│       ├── file_handler.py         # Manipulação de arquivos multimodais
│       └── report_generator.py     # Gerador de relatórios (JSON + Markdown)
├── data/
│   ├── videos/                     # Vídeos clínicos para análise
│   ├── audios/                     # Áudios de consultas para análise
│   └── reports/                    # Relatórios gerados automaticamente
│       └── alerts/                 # Logs de alertas em JSON
├── tests/                          # Testes automatizados
├── models/                         # Modelos treinados (.pt, .pkl)
├── notebooks/                      # Jupyter notebooks para experimentação
└── edital/
    └── fase-4-edital.md            # Edital da Fase 4
```

## Fluxo de Trabalho (Tasks)

### Task 1: Análise de Vídeo (Video Agent)

- **Agente:** VideoAnalysisAgent
- **Entrada:** Arquivo de vídeo (.mp4, .avi, .mov, .mkv)
- **Modelo:** YOLOv8 customizado para detecção de:
  - Instrumentos cirúrgicos ginecológicos
  - Sangramento anômalo durante procedimentos
  - Áreas críticas (útero, ovários, mamas)
  - Objetos suspeitos (indicativos de automutilação)
  - Expressões faciais de desconforto/medo
- **Saída:** Relatório de análise + alertas de anomalia

### Task 2: Análise de Áudio (Audio Agent)

- **Agente:** AudioAnalysisAgent
- **Entrada:** Arquivo de áudio (.wav, .mp3, .m4a, .ogg)
- **Modelos:** Whisper (transcrição) + Wav2Vec2 (emoção vocal)
- **Detecção de:**
  - Depressão pós-parto
  - Ansiedade gestacional
  - Sinais de violência doméstica
  - Padrões vocais de trauma
  - Fadiga hormonal
- **Saída:** Transcrição + score de risco emocional + alertas

### Task 3: Anomalias em Sinais Vitais (Anomaly Agent)

- **Agente:** AnomalyDetectionAgent
- **Entrada:** Dicionário de sinais vitais (pressão, batimentos fetais, etc.)
- **Métodos:** Z-Score + Rolling Statistics + Isolation Forest
- **Monitoramento de:**
  - Pressão arterial em gestantes
  - Batimentos fetais
  - Níveis hormonais
  - Glicose, temperatura, saturação O2
- **Saída:** DataFrame com anomalias detectadas + severidade

### Task 4: Fusão Multimodal (Fusion Agent)

- **Agente:** MultimodalFusionAgent
- **Entrada:** Resultados dos 3 agentes anteriores
- **Integração:** Ponderação dos scores + correlação de riscos
- **Saída:** Score unificado + recomendações para equipe médica

## Como Executar

### 1. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env com suas chaves (opcional para Azure)
```

### 3. Executar comandos

```bash
# Análise de vídeo
python main.py video data/videos/cirurgia.mp4 --type cirurgia

# Análise de áudio
python main.py audio data/audios/consulta.wav --type pos_parto

# Pipeline multimodal completo
python main.py multimodal PAC-001 --video video.mp4 --audio audio.wav

# Interface interativa Streamlit
python main.py demo
```

### 4. Comandos do CLI

```
python main.py --help
python main.py video --help
python main.py audio --help
python main.py multimodal --help
```

## Critérios do Edital Atendidos

| Critério | Implementação |
|----------|--------------|
| Análise de vídeos (cirurgias, consultas, fisioterapia) | `video_agent.py` + `yolo_detector.py` |
| Processamento de gravações de voz | `audio_agent.py` + `audio_analyzer.py` |
| Detecção de anomalias em sinais vitais | `anomaly_agent.py` |
| Integração Azure Cognitive Services | `azure_services.py` |
| Detecção precoce de riscos em saúde materna | `fusion_agent.py` |
| Identificação de sinais de violência doméstica | `video_agent.py` + `audio_agent.py` |
| Monitoramento de bem-estar psicológico | `audio_agent.py` (emoções vocais) |
| Relatórios automáticos especializados | `report_generator.py` |
| Alertas à equipe médica em tempo real | `alert_service.py` |
| YOLOv8 customizado | `yolo_detector.py` (fine-tuning incluso) |
| Privacidade e segurança de dados sensíveis | `azure_services.py` (Storage criptografado) |

## Tecnologias

| Categoria | Ferramenta |
|-----------|-----------|
| Detecção visual | YOLOv8 (Ultralytics) |
| Transcrição de áudio | OpenAI Whisper |
| Análise de emoção vocal | Wav2Vec2 (HuggingFace) |
| Anomalias em séries temporais | Isolation Forest + Z-Score (Scikit-learn) |
| Processamento de áudio | Librosa + SoundFile |
| Processamento de vídeo | OpenCV + MoviePy |
| Cloud | Azure Speech, Vision, Storage, OpenAI |
| Interface | Typer (CLI) + Streamlit (Web) |
| Configuração | Pydantic Settings + python-dotenv |
| Logging | Loguru |
