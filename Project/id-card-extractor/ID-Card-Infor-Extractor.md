# ID-Card-Information-Extractor/
id-card-extractor/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py
│   │   ├── extract.py
│   │   ├── display.py
│   │   └── webcam.py         # Endpoint for handling webcam input
│   ├── services/
│   │   ├── __init__.py
│   │   ├── extraction.py
│   │   ├── validation.py
│   │   ├── detection.py      # Logic for detecting ID card in webcam frames
│   │   └── webcam_feed.py    # Logic for capturing and processing webcam feed
│   ├── models/
│   │   ├── __init__.py
│   │   └── id_card.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py
│   │   └── webcam_utils.py   # Helper functions for webcam integration
│   └── schemas/
│       ├── __init__.py
│       └── id_card.py
│
├── gradio_ui/
│   ├── __init__.py
│   ├── app.py
│   ├── components.py
│   └── webcam_component.py   # Gradio component for webcam integration
│
├── tests/
│   ├── __init__.py
│   ├── test_routes.py
│   ├── test_services.py
│   ├── test_webcam.py        # Tests for webcam functionality
│   └── test_ui.py
│
├── static/
│   └── css/
│       └── styles.css
│
├── .env
├── requirements.txt
├── README.md
└── Dockerfile



