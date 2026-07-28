# Roteiro do Vídeo — Sistema de Monitoramento Multimodal (Fase 4)

**Duração:** 15 minutos | **Formato:** Gravação de tela com narração
**Upload:** YouTube / Vimeo (público ou não listado)

---

## Checklist do Edital

| # | Item | Minuto |
|---|------|--------|
| 1 | Processamento multimodal | 00:00–14:00 |
| 2 | Exemplo prático de áudio e vídeo | 02:00–09:00 |
| 3 | Detecção e resposta a anomalias | 05:00–11:00 |
| 4 | Integração Azure | 09:00–12:00 |
| 5 | Fluxo de alerta à equipe médica | 12:00–14:00 |
| — | Encerramento | 14:00–15:00 |

---

## Bloco 1 — Abertura (0:00–1:30)

**Tela:** Terminal + Streamlit sidebar

```bash
source .venv/bin/activate
python main.py version
tree -L 2 src/
python main.py demo &
```

**Código para mostrar:** Sidebar com `Azure Speech ✅`, `Azure Vision ✅`, `Azure Storage ✅`, `Alertas por Email ✅`

**Fala:**
> "Olá, somos o Grupo [NOME]. Este é o sistema de Monitoramento Multimodal
> para Saúde da Mulher — Fase 4 do Tech Challenge. O sistema analisa
> simultaneamente vídeo, áudio e sinais vitais. Quatro agentes autônomos
> integrados com Azure Cognitive Services. Na sidebar, todos os serviços ativos."

---

## Bloco 2 — Arquitetura de Treinamento (1:30–4:00)

**Tela 1 — Estrutura dos datasets**
```bash
ls -la data/dataset/
ls data/dataset/emotions/train/images/ | head -5
ls data/dataset/surgery_instruments/train/images/ | head -5
ls data/dataset/blood/train/images/ | head -5
```

**Tela 2 — data.yaml de cada dataset**
```bash
cat data/dataset/emotions/data.yaml
```
> Mostrar `nc: 8` e as classes: medo, tristeza, raiva, surpresa...

```bash
cat data/dataset/surgery_instruments/data.yaml
```
> Mostrar `nc: 21` e as classes: bisturi, tesoura, pinça, afastador...

```bash
cat data/dataset/blood/data.yaml
```
> Mostrar `nc: 1` — classe única: sangramento

**Tela 3 — Código do treinamento com augmentations**
Abrir `src/models/yolo_detector.py` — método `finetune()` (linha ~427)

```python
# Mostrar parâmetros de data augmentation
def finetune(self, data_yaml, epochs=50, imgsz=640, batch=16, model_name="custom"):
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        hsv_h=0.03, hsv_s=0.5, hsv_v=0.4,    # variação de cor/iluminação
        degrees=15.0,                           # rotação ±15°
        scale=0.6,                              # zoom out
        mosaic=0.8, mixup=0.15,                # combinação de imagens
        erasing=0.2,                            # oclusão parcial
    )
```

**Tela 4 — Comando de treino + métricas**
```bash
python train.py --dataset emotions --epochs 30 --batch 8
python train.py --dataset blood --epochs 50 --batch 8
python train.py --dataset instruments --epochs 30 --batch 8
```

> Mostrar terminal com métricas de validação: `mAP50=0.XX` para cada modelo

**Fala:**
> "Treinamos 3 modelos YOLOv8 especializados com datasets do Roboflow Universe.
> Emoções: 6.586 imagens, 8 classes de expressão facial. Sangramento: 82 imagens
> com augmentations agressivas. Instrumentos: 1.530 imagens, 21 classes convertidas
> de OBB para bbox padrão. Data augmentation inclui rotação, variação de cor,
> mosaic e random erasing para simular condições reais de centro cirúrgico."

---

## Bloco 3 — Código: Carregamento dos 3 Modelos (4:00–5:00)

**Tela:** `src/models/yolo_detector.py` — método `_load_models()` (linha ~100)

```python
def _load_models(self) -> None:
    model_files = {
        "emotions": "models/emotions.pt",
        "blood": "models/blood.pt",
        "instruments": "models/instruments.pt",
    }
    for name, path in model_files.items():
        if Path(path).exists():
            self._models[name] = YOLO(path)
```

**Tela:** `src/models/yolo_detector.py` — método `process_frame()` (linha ~138)

```python
def process_frame(self, frame, frame_number, fps):
    self._load_models()
    all_detections = []
    for name, model in self._models.items():      # 3 modelos
        results = model.predict(source=frame, ...)
        for box in result.boxes:
            all_detections.append(DetectionResult(
                class_name=model.names.get(cls_id),
                confidence=confidence, bbox=bbox, ...
            ))
    return all_detections  # detecções mescladas dos 3 modelos
```

**Fala:**
> "O YOLODetector carrega os 3 modelos e itera sobre cada um no mesmo frame.
> As detecções são mescladas em uma única lista. Como cada modelo é especialista
> no seu domínio, um rosto nunca será confundido com sangramento."

---

## Bloco 4 — Demo de Vídeo: YOLOv8 Live (5:00–7:00)

**Tela:** Terminal + janela OpenCV

```bash
python main.py live data/videos/cesaria.mp4 --conf 0.20
```

> Mostrar a janela com bounding boxes coloridas e labels

**Código:** `src/models/yolo_detector.py` — método `live_detect()` (linha ~219)

```python
def live_detect(self, video_path, sample_every_n_frames=3):
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if frame_idx % sample_every_n_frames == 0:
            last_detections = self.process_frame(frame, frame_idx, fps)
        annotated = self.annotate_frame(frame, last_detections)
        cv2.imshow(window_name, annotated)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'): break
```

**Fala:**
> "Live mode: OpenCV lê o vídeo frame a frame, roda detecção com os 3 modelos
> e desenha bounding boxes. Caixas verdes são detecções normais, cada uma com
> nome da classe e confiança. Controles: ESPAÇO pausa, Q sai."

---

## Bloco 5 — Demo de Vídeo: Azure Vision Live (7:00–8:30)

**Tela:** Terminal + janela OpenCV com texto amarelo sobreposto

```bash
python main.py live-azure data/videos/cesaria.mp4 --interval 1.0
```

**Código:** `src/models/yolo_detector.py` — método `azure_live_detect()` (linha ~269)

```python
def azure_live_detect(self, video_path, interval_seconds=1.0):
    azure = AzureVisionService()
    while True:
        ret, frame = cap.read()
        if frame_idx - last_update >= azure_interval:
            result = azure.analyze_image(tmp_path)
            caption = result.get("caption", "")
            last_result = f"Azure: {caption}"
        # sobrepõe texto amarelo no frame
        cv2.putText(overlay, last_result, (10, 30), ...)
```

**Código:** `src/services/azure_services.py` — classe `AzureVisionService` (linha ~169)

```python
class AzureVisionService:
    def analyze_image(self, image_path):
        client = ImageAnalysisClient(endpoint=self.endpoint, ...)
        result = client.analyze(image_data=image_data, visual_features=[
            VisualFeatures.TAGS, VisualFeatures.CAPTION, ...
        ])
        return {"caption": result.caption.text, "tags": [...], "objects": [...]}
```

**Fala:**
> "Azure Vision Live: envia frames para a nuvem a cada 1 segundo. O caption
> retornado — 'a person in gloves performing surgery' — é sobreposto em
> amarelo no vídeo. O Azure entende o contexto completo da cena, enquanto
> o YOLOv8 detecta objetos individuais. Os dois modos se complementam."

---

## Bloco 6 — Análise de Áudio (8:30–10:30)

**Tela:** Streamlit aba "Análise de Áudio" + upload + resultado

**Código:** `src/models/audio_analyzer.py` — classe `SpeechToText` (linha ~176)

```python
class SpeechToText:
    @property
    def provider(self) -> str:
        if settings.azure.speech_key and len(...) > 30:
            return "azure"     # primário: Azure Speech
        return "whisper"       # fallback: Whisper local

    def transcribe(self, audio_path):
        if self.provider == "azure":
            text = self._azure.transcribe(audio_path)  # Azure Speech API
        else:
            result = self.model.transcribe(audio_path) # Whisper local
```

**Código:** `src/models/audio_analyzer.py` — `EmotionAnalyzer.compute_risk_score()` (linha ~364)

```python
def compute_risk_score(self, emotions, has_text_risk=False, text_indicator_count=0):
    negative_emotions = {"sad", "angry", "fearful", "disgusted"}
    risk_score = sum(e.confidence for e in emotions if e.emotion in negative_emotions) / total
    if has_text_risk:
        text_boost = min(0.5, 0.15 + text_indicator_count * 0.05)
        risk_score = max(risk_score, text_boost)
    return risk_score, "alto" if risk_score > 0.50 else ...
```

**Fala:**
> "Azure Speech como primário, Whisper como fallback. Wav2Vec2 detecta emoções
> a cada 3 segundos. O score combina emoção vocal + indicadores textuais —
> cada palavra de risco no texto aumenta o score mínimo em 0.05."

---

## Bloco 7 — Sinais Vitais e Pipeline Multimodal (10:30–12:30)

**Tela:** Streamlit aba "Pipeline Multimodal" + sinais vitais + resultado

**Código:** `src/agents/anomaly_agent.py` — `AnomalyDetectionAgent` (linha ~190)

```python
class AnomalyDetectionAgent:
    def _zscore_detection(self, record):
        zscore = np.abs(stats.zscore(values))[-1]
        is_anomaly = zscore > self.zscore_threshold
        return AnomalyDetectionResult(is_anomaly=is_anomaly, ...)

    def _range_detection(self, record):
        low, high = GESTATIONAL_REFERENCE_RANGES[record.signal_type]
        in_range = low <= record.value <= high
```

**Código:** `src/agents/fusion_agent.py` — `MultimodalFusionAgent.fuse()` (linha ~78)

```python
class MultimodalFusionAgent:
    def __init__(self):
        self.video_weight = 0.35
        self.audio_weight = 0.35
        self.vitals_weight = 0.30

    def fuse(self, patient_id, video_alerts, audio_alerts, vitals_results):
        video_score = self._compute_video_risk(video_alerts)    # top-10 alertas
        audio_score = self._compute_audio_risk(audio_alerts, audio_report_score)
        vitals_score = self._compute_vitals_risk(vitals_results)
        overall = video_score * 0.35 + audio_score * 0.35 + vitals_score * 0.30
```

**Fala:**
> "Sinais vitais: Z-Score + faixas de referência gestacionais. Fusion Agent:
> score unificado = 35% vídeo + 35% áudio + 30% vitais. Usa top-10 alertas
> de vídeo para não diluir o score com falsos positivos."

---

## Bloco 8 — Email, Storage e Fluxo de Alerta (12:30–14:00)

**Tela:** Email HTML + bucket Azure no sidebar + logs do terminal

**Código:** `src/services/alert_service.py` — `AlertService` (linha ~23)

```python
class AlertService:
    def send_alert(self, assessment):
        if self.email_password:
            self._send_email(assessment, is_alert=True)
        self._save_alert_log(assessment)
        # upload para Azure Storage
        storage.upload_file(filepath, container_name="saude-mulher-alertas")

    def _send_email(self, assessment, is_alert):
        if self.smtp_port == 465:
            server = smtplib.SMTP_SSL(host, port)     # SSL
        else:
            server = smtplib.SMTP(host, port)
            server.starttls()                          # TLS
        server.send_message(msg)  # email HTML com scores e recomendações
```

**Código:** `src/services/azure_services.py` — `AzureStorageService` (linha ~280)

```python
class AzureStorageService:
    def upload_file(self, file_path, container_name="saude-mulher"):
        blob_name = f"{patient_id}/{timestamp}/{file_path.name}"
        container.upload_blob(name=blob_name, data=data, overwrite=True)

    def list_files(self, container_name):
        for blob in container.list_blobs():
            blobs.append({"name": blob.name, "size": blob.size, ...})
```

**Fala:**
> "Alertas: email HTML com scores por modalidade, riscos correlacionados e
> recomendações. Suporte a SSL (465) e TLS (587). Azure Storage: upload
> automático de vídeos, áudios e relatórios organizados por paciente.
> A sidebar do Streamlit lista os arquivos do bucket em tempo real."

---

## Bloco 9 — Encerramento (14:00–15:00)

**Tela:** Checklist ou slide com os critérios atendidos

> "Entregamos um sistema que atende todos os critérios da Fase 4:
>
> - 3 modelos YOLOv8 especializados com data augmentation
> - Azure AI Vision para descrição de cena em tempo real
> - Azure Speech + Whisper para transcrição + Wav2Vec2 para emoções
> - Z-Score + faixas gestacionais em 9 sinais vitais
> - Fusão multimodal com score unificado
> - Alertas por email HTML + Azure Blob Storage
> - Interface Streamlit com barras de progresso e bucket browser
>
> Código-fonte completo no repositório Git. Obrigado."

---

## Referência Rápida de Arquivos para Mostrar

| Minuto | Bloco | Arquivo | Linha |
|--------|-------|---------|-------|
| 1:30 | Datasets | `data/dataset/*/data.yaml` | — |
| 2:30 | Augmentations | `src/models/yolo_detector.py:finetune()` | ~427 |
| 4:00 | 3 modelos | `src/models/yolo_detector.py:_load_models()` | ~100 |
| 4:30 | Mesclagem | `src/models/yolo_detector.py:process_frame()` | ~138 |
| 5:00 | Live YOLO | `src/models/yolo_detector.py:live_detect()` | ~219 |
| 7:00 | Live Azure | `src/models/yolo_detector.py:azure_live_detect()` | ~269 |
| 7:30 | Azure Vision | `src/services/azure_services.py:AzureVisionService` | ~169 |
| 8:30 | Áudio provider | `src/models/audio_analyzer.py:SpeechToText.provider` | ~194 |
| 9:30 | Risk score | `src/models/audio_analyzer.py:compute_risk_score()` | ~364 |
| 10:30 | Anomalias | `src/agents/anomaly_agent.py:_zscore_detection()` | ~190 |
| 11:00 | Fusão | `src/agents/fusion_agent.py:fuse()` | ~78 |
| 12:30 | Email | `src/services/alert_service.py:_send_email()` | ~85 |
| 13:00 | Storage | `src/services/azure_services.py:AzureStorageService` | ~280 |

---

## Preparação Pré-Gravação

### Arquivos de demonstração
```
data/videos/cesaria.mp4          — cesariana (87 MB, ~14 min)
data/audios/consulta.mp3         — áudio de 16s com indicadores de risco
```

### Texto para gravação do áudio
> "Doutor, por favor me ajuda. Eu estou em desespero, sentindo muita dor,
> com depressão profunda e muito medo. Eu sofri violência e não aguento
> mais essa crise e solidão."

### Sinais vitais (preencher no formulário)
- Pressão Sistólica: 150 | Diastólica: 95
- Batimentos Fetais: 175 | Temperatura: 38.2
- ☑ Faixas gestacionais

### Verificação de dependências
```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
python -c "import whisper; print(f'openai-whisper {whisper.__version__}')"
python -c "import azure.cognitiveservices.speech; print('Azure Speech OK')"
python -c "import streamlit; print(f'Streamlit {streamlit.__version__}')"
```
