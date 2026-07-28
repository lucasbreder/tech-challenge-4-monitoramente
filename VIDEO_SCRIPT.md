# Roteiro do Vídeo — Sistema de Monitoramento Multimodal (Fase 4)

**Duração:** 15 minutos | **Formato:** Gravação de tela com narração

---

## Checklist do Edital

| # | Item | Minuto |
|---|------|--------|
| 1 | Processamento multimodal | 00:00–14:00 |
| 2 | Análise de áudio e vídeo | 02:00–08:00 |
| 3 | Detecção e resposta a anomalias | 05:00–11:00 |
| 4 | Integração Azure | 08:00–12:00 |
| 5 | Fluxo de alerta à equipe médica | 12:00–14:00 |
| — | Encerramento | 14:00–15:00 |

---

## Bloco 1 — Abertura (0:00–1:30)

**Tela:** `python main.py version` + `tree -L 2 src/` + sidebar Streamlit

> "Olá, somos o Grupo [NOME]. Este é o sistema de Monitoramento Multimodal
> para Saúde da Mulher — Fase 4 do Tech Challenge.
>
> O sistema analisa vídeo, áudio e sinais vitais para detectar riscos como
> depressão pós-parto, sangramento anômalo e sinais de violência doméstica.
>
> Quatro agentes autônomos: Video, Audio, Anomaly e Fusion — integrados com
> Azure Cognitive Services. Na sidebar vemos todos os serviços ativos."

---

## Bloco 2 — Treinamento do Modelo (1:30–4:00)

**Tela:** `merge_datasets.py` + terminal com métricas de validação

> "O coração da análise de vídeo é um YOLOv8 treinado com 11.474 imagens
> de 3 datasets do Roboflow Universe, unificados em 30 classes:
>
> - **Sangramento**: 163 imagens de blood stains
> - **Instrumentos cirúrgicos**: 1.675 imagens com 21 instrumentos
> - **Expressões faciais**: 8.459 imagens com 8 emoções
>
> Para melhorar a generalização em cenas cirúrgicas reais, aplicamos
> data augmentation agressiva durante o treino:
>
> - **HSV**: variação de matiz, saturação e brilho — simula diferentes
>   condições de iluminação cirúrgica
> - **Rotação de ±15° e shear de ±5°** — simula diferentes ângulos de câmera
> - **Scale de 0.6** — instrumentos em diferentes distâncias
> - **Mosaic 80% e MixUp 15%** — combina múltiplas imagens para ensinar
>   o modelo a lidar com oclusão e contexto
> - **Random Erasing 20%** — corta partes da imagem para simular instrumentos
>   parcialmente visíveis durante a cirurgia
>
> Após 30 épocas, o modelo atingiu **mAP50 de 0.84** na validação.
> Destaque para: bisturi 0.99, tesoura cirúrgica 0.995,
> pinça hemostática 0.97, expressão de medo 0.80."

**Arquivos para mostrar:** `merge_datasets.py`, `src/models/yolo_detector.py` (método `finetune` com parâmetros de augmentation)

---

## Bloco 3 — Demo de Vídeo: YOLOv8 Live (4:00–6:00)

**Tela:** `python main.py live data/videos/cesaria.mp4 --conf 0.20`

> "Com o modelo treinado, podemos rodar detecção em tempo real.
> O comando `live` abre uma janela OpenCV com bounding boxes desenhadas
> sobre cada instrumento, expressão ou sangramento detectado.
>
> Caixas verdes são detecções normais, vermelhas indicam anomalias.
> O nome da classe e a confiança aparecem acima de cada box."

---

## Bloco 4 — Demo de Vídeo: Azure Vision Live (6:00–8:00)

**Tela:** `python main.py live-azure data/videos/cesaria.mp4`

> "Além do YOLOv8 local, integramos o Azure AI Vision como alternativa.
> O comando `live-azure` envia frames para a nuvem a cada 1 segundo e
> sobrepõe a descrição da cena no vídeo:
>
> 'a person in gloves performing surgery on a patient'
> 'a doctor handing a surgical instrument'
>
> O Azure Vision entende o contexto completo da cena — algo que o YOLOv8
> não consegue sozinho. Os dois modos se complementam."

---

## Bloco 5 — Análise de Áudio (8:00–10:00)

**Tela:** Aba "Análise de Áudio" no Streamlit + upload + transcrição

> "A análise de áudio usa Azure Speech Services como primário, com
> reconhecimento contínuo para arquivos acima de 12 segundos.
> Fallback automático para Whisper local se o Azure estiver offline.
>
> O Wav2Vec2 analisa emoções na voz a cada 3 segundos, e o sistema
> cruza os indicadores textuais (palavras como 'depressão', 'violência',
> 'medo') com a análise emocional para calcular o score de risco.
>
> Neste áudio de 16 segundos, a paciente relata depressão, violência e
> desespero. O sistema detectou 7 indicadores de risco e classificou
> como risco ALTO, gerando alerta de possível violência doméstica."

---

## Bloco 6 — Sinais Vitais e Pipeline Multimodal (10:00–12:00)

**Tela:** Aba "Pipeline Multimodal" + sinais vitais + barras de progresso

> "O pipeline multimodal recebe vídeo, áudio e sinais vitais simultaneamente.
> Barras de progresso independentes mostram o avanço do vídeo e do áudio.
>
> Preencho pressão 150/95 e batimentos fetais 175 — fora da faixa
> gestacional. O Z-Score detecta as anomalias em tempo real.
>
> Ao final, o Fusion Agent calcula o score unificado: 35% vídeo,
> 35% áudio, 30% sinais vitais. Os alertas aparecem em accordions
> colapsáveis: vídeo, áudio e vitais, cada um com severidade e descrição."

---

## Bloco 7 — Alertas e Infraestrutura (12:00–14:00)

**Tela:** Email HTML recebido + bucket Azure no sidebar

> "Quando o risco é alto ou crítico, o sistema dispara um email HTML
> com: score unificado e por modalidade, riscos correlacionados e
> recomendações de conduta clínica. Compatível com SMTP SSL e TLS.
>
> Paralelamente, vídeos, áudios e relatórios são enviados automaticamente
> para o Azure Blob Storage, organizados por paciente e timestamp.
> A sidebar do Streamlit permite visualizar os arquivos no bucket.
>
> O Storage também armazena os alertas em JSON para integração com
> sistemas hospitalares HL7 e FHIR."

---

## Bloco 8 — Encerramento (14:00–15:00)

> "Entregamos um sistema que atende todos os critérios da Fase 4:
>
> - YOLOv8 com 30 classes treinado em 11.474 imagens com
>   data augmentation para cenários cirúrgicos reais
> - Azure AI Vision para descrição de cena em tempo real
> - Azure Speech + Whisper para transcrição e Wav2Vec2 para emoções
> - Z-Score + Isolation Forest em 9 sinais vitais com faixas gestacionais
> - Fusão multimodal com score unificado e correlação de riscos
> - Alertas por email HTML + Azure Blob Storage
> - Interface Streamlit com barras de progresso e bucket browser
>
> Código-fonte completo no repositório Git. Obrigado."

---

## Arquivos para Demonstração

| Arquivo | Conteúdo |
|---------|----------|
| `data/videos/cesaria.mp4` | Cirurgia — YOLO live + Azure live |
| `data/audios/consulta_pos_parto.mp3` | Áudio de 16s com indicadores de risco |
| Sinais vitais | Pressão 150/95, Bat. Fetais 175, Temp 38.2 |

## Texto para Gravação do Áudio

> "Doutor, por favor me ajuda. Eu estou em desespero, sentindo muita dor,
> com depressão profunda e muito medo. Eu sofri violência e não aguento
> mais essa crise e solidão."
