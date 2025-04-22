# Ricostruzione tomografica con regolarizzazione Total Variation (TV)

Questo progetto si propone di simulare e risolvere problemi di ricostruzione di immagini in tomografia computerizzata (CT), utilizzando dati sintetici generati a partire da immagini reali (es. dal Mayo Clinic Dataset).  

Il focus attuale è sull'implementazione e analisi della ricostruzione tramite **Filtered Backprojection (FBP)** e successivamente tramite **regolarizzazione Total Variation (TV)**.

## Obiettivi

- Simulare sinogrammi con geometrie reali tramite ASTRA Toolbox.
- Eseguire una ricostruzione di base con FBP.
- Valutare le prestazioni con metriche standard.
- Preparare la struttura per successive estensioni con tecniche avanzate come Plug-and-Play.

## Struttura del progetto

```
computational_imaging/
├── main.py                    # Script principale per esecuzione e test
├── requirements.txt           # Dipendenze Python
├── README.md                  # Documentazione generale
│
├── data/
│   ├── raw/                   # Immagini ground truth (es. dal dataset Mayo)
│   │   ├── train/             # (Per addestramento, se necessario)
│   │   └── test/              # Immagini per la ricostruzione
│   ├── sinograms/             # Sinogrammi simulati
│   └── reconstructions/       # Output delle ricostruzioni
│
├── src/
│   ├── sinogram_simulation.py  # Simulazione sinogrammi con ASTRA
│   ├── total_variation.py      # Ricostruzione (FBP, e TV in fase successiva)
│   ├── utils.py                # Funzioni di utilità (normalizzazione, metriche, rumore)
```

## Requisiti

Tutte le dipendenze sono elencate nel file `requirements.txt`.

Si consiglia fortemente l’utilizzo di un ambiente virtuale gestito con **conda**, in particolare per la compatibilità con la libreria ASTRA Toolbox.

### Esempio di installazione (usando conda)

```bash
conda create -n compImaging python=3.10
conda activate compImaging
pip install -r requirements.txt
```

## Esecuzione

1. Inserire un’immagine `.png` o `.jpg` nella cartella `data/raw/test/`, preferibilmente di dimensioni 512x512.
2. Lanciare il file `main.py` dalla root del progetto:

```bash
python main.py
```

### Il file `main.py` esegue le seguenti operazioni:

- Carica una ground truth da `data/raw/test/`
- Simula i sinogrammi corrispondenti con ASTRA:
  - 30 angoli in [-30°, 30°], con e senza rumore
  - 180 angoli in [-90°, 90°], con e senza rumore
- Applica la ricostruzione FBP
- Calcola le metriche tra immagine originale e ricostruita:
  - Errore relativo (RE)
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
- Visualizza i risultati

## Output

- I sinogrammi vengono salvati nella cartella `data/sinograms/`.
- Le immagini ricostruite possono essere salvate nella cartella `data/reconstructions/`.
- Le metriche vengono stampate in console.

## Estensioni previste

- Integrazione di tecniche di ricostruzione con regolarizzazione Total Variation (TV).
- Confronto sistematico tra FBP e TV su tutto il test set.
- Aggiunta di metodi Plug-and-Play con denoiser neurali (fase successiva).
