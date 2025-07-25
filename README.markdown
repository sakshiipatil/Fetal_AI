# Fetal Health Analyzer - Orange Edition

Created by Sakshi Patil, this Fetal Health Analyzer is a cutting-edge web tool built to assess fetal health using machine learning. Launched on July 25, 2025, at 04:32 PM IST, this version features a vibrant orange theme, a two-column design, and advanced prediction capabilities.

## Introduction

This application leverages data from fetal heart rate (FHR) parameters to predict health outcomes—Normal, Suspect, or Pathological—using Logistic Regression, KNN, and Random Forest models. The orange-themed interface, crafted with a two-column layout, offers an intuitive experience with modern styling.

## Key Highlights

- **Model Options**: Switch between three ML models for tailored predictions.
- **User-Friendly Design**: Two-column layout with tooltips and dynamic loading indicators.
- **Unique Styling**: Orange color scheme, Poppins font, and Font Awesome icons.
- **Extra Feature**: `/health` endpoint for checking system status.
- **Enhanced Output**: Includes a placeholder for prediction confidence.

## File Organization

```
fetal-health-copy5/
├── app.py              # Core Flask app with prediction functionality
├── requirements.txt    # List of required Python packages
├── fetal_health.csv    # Input dataset for model training
└── templates/
    ├── index.html      # Main interface with orange two-column design
    └── output.html     # Template for displaying results
```

## Getting Started

### Setup Instructions

1. **Get the Code**:
   ```bash
   git clone <your-repo-url>
   cd fetal-health-copy5
   ```

2. **Install Libraries**:
   With Python ready, execute:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Dataset**:
   Copy `fetal_health.csv` into the project folder (dataset provided separately).

4. **Launch the App**:
   ```bash
   python app.py
   ```
   Access it at `http://127.0.0.1:5000/`.

## How to Use

- Pick a model from the dropdown.
- Input FHR data into the fields provided.
- Hit "Predict Health Status" to view results, including a confidence score placeholder.
- Reset the form with the "Reset" button.
- Check system health via `/health` (e.g., `http://127.0.0.1:5000/health`).

## Tech Stack

- **Backend**: Flask, Python, Scikit-learn
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Design Elements**: Poppins font, Font Awesome icons
- **Data Source**: `fetal_health.csv`

## Visuals

![Main Interface](https://github.com/your-username/fetal-health-copy5/blob/main/images/interface.png)  
![Result View](https://github.com/your-username/fetal-health-copy5/blob/main/images/result.png)  
*(Update with your actual image URLs.)*

## Contribution Guidelines

Open to enhancements! Fork the repo, tweak the code, and submit a pull request. Ideas for new features or UI improvements are appreciated.

## Licensing

Distributed under the MIT License. Check the `LICENSE` file for more info (create one if needed).

## Get in Touch

Questions? Contact Sakshi Patil at sakshi.patil@example.com.