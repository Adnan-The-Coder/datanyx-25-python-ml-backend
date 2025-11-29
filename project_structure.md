fastapi-mg-template/
├── data/
│   └── mg_clinical_data.csv          # The main dataset (raw/cleaned for simplicity)
├── training/
│   ├── __init__.py
│   └── trainer.py                    # Contains the actual ML model training, fitting, and saving logic
├── app/
│   ├── main.py                       # App entry point, includes all routers
│   ├── core/
│   │   └── config.py                 # Configuration variables (data paths, model paths)
│   ├── models/                       # Location for all saved .pkl and .h5 files
│   │   ├── diagnosis_rf.pkl
│   │   ├── severity_lgbm.pkl
│   │   ├── warning_lstm.h5
│   │   └── feature_scaler.pkl        # New: Saves the data preprocessor (Scaler/Encoder)
│   ├── schemas/
│   │   ├── input_data.py
│   │   ├── features.py
│   │   └── output_report.py
│   ├── services/
│   │   ├── feature_service.py        # Loads scaler, runs pre-processing
│   │   ├── ensemble_service.py       # Loads models, runs inference
│   │   └── training_service.py       # New: Manages calling the trainer.py script
│   └── api/
│       └── v1/
│           ├── endpoints/
│           │   ├── predict.py        # The prediction endpoint
│           │   └── train.py          # New: The training endpoint
│           └── routers.py
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md