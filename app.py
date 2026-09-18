import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load XGBoost Pickle Model
MODEL_PATH = "XGmodel.pkl"

def load_xgboost_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' was not found in the current directory.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_xgboost_model()
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Feature Categorical Mappings based on XGmodel.pkl schema
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
BLOOD_TYPE_MAP = {"A+": 0, "A-": 1, "B+": 2, "B-": 3, "AB+": 4, "AB-": 5, "O+": 6, "O-": 7}
MEDICAL_COND_MAP = {"Arthritis": 0, "Asthma": 1, "Cancer": 2, "Diabetes": 3, "Hypertension": 4, "Obesity": 5}
INSURANCE_MAP = {"Aetna": 0, "Blue Cross": 1, "Cigna": 2, "Medicare": 3, "UnitedHealthcare": 4}
ADMISSION_MAP = {"Elective": 0, "Emergency": 1, "Urgent": 2}

CLASS_LABELS = {
    0: {"name": "Low Risk / Stable", "badge": "bg-success", "color": "#10b981", "desc": "Patient indicators suggest normal recovery trajectory."},
    1: {"name": "Moderate Risk / Monitor", "badge": "bg-warning", "color": "#f59e0b", "desc": "Patient requires close observation and follow-up tests."},
    2: {"name": "High Risk / Critical", "badge": "bg-danger", "color": "#ef4444", "desc": "Immediate clinical intervention and ICU evaluation recommended."}
}

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded properly on the server."}), 500

    try:
        data = request.get_json()

        # Extract features in the EXACT order defined in XGmodel.pkl
        age = float(data.get("Age", 0))
        gender = GENDER_MAP.get(data.get("Gender"), 0)
        blood_type = BLOOD_TYPE_MAP.get(data.get("Blood Type"), 0)
        med_condition = MEDICAL_COND_MAP.get(data.get("Medical Condition"), 0)
        insurance = INSURANCE_MAP.get(data.get("Insurance Provider"), 0)
        billing = float(data.get("Billing Amount", 0))
        admission = ADMISSION_MAP.get(data.get("Admission Type"), 0)

        features = np.array([[age, gender, blood_type, med_condition, insurance, billing, admission]])

        # Execute prediction
        probabilities = model.predict_proba(features)[0].tolist()
        prediction_class = int(np.argmax(probabilities))
        class_info = CLASS_LABELS.get(prediction_class, {"name": f"Class {prediction_class}", "badge": "bg-primary", "color": "#3b82f6", "desc": ""})

        return jsonify({
            "status": "success",
            "prediction": class_info["name"],
            "prediction_badge": class_info["badge"],
            "prediction_color": class_info["color"],
            "prediction_desc": class_info["desc"],
            "class_index": prediction_class,
            "probabilities": probabilities,
            "feature_summary": {
                "Age": f"{int(age)} Yrs",
                "Gender": data.get("Gender"),
                "Blood Type": data.get("Blood Type"),
                "Medical Condition": data.get("Medical Condition"),
                "Insurance Provider": data.get("Insurance Provider"),
                "Billing Amount": f"${billing:,.2f}",
                "Admission Type": data.get("Admission Type")
            },
            "normalized_metrics": [
                min(100, int((age / 100) * 100)),
                (gender + 1) * 33,
                (blood_type + 1) * 12,
                (med_condition + 1) * 16,
                (insurance + 1) * 20,
                min(100, int((billing / 50000) * 100)),
                (admission + 1) * 33
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Next-Gen Dashboard Interface Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="emerald">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Healthcare Results Prediction</title>
    
    <!-- Fonts & Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.2/font/bootstrap-icons.min.css">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg-body: #090d16;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-gradient: linear-gradient(135deg, #059669 0%, #0284c7 100%);
            --accent-glow: rgba(5, 150, 105, 0.35);
            --accent-color: #10b981;
            --card-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.7);
        }

        [data-theme="cyberpunk"] {
            --bg-body: #070210;
            --bg-card: rgba(23, 10, 41, 0.75);
            --border-color: rgba(236, 72, 153, 0.2);
            --text-main: #f43f5e;
            --text-muted: #a855f7;
            --accent-gradient: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
            --accent-glow: rgba(236, 72, 153, 0.4);
            --accent-color: #ec4899;
            --card-shadow: 0 0 30px rgba(236, 72, 153, 0.2);
        }

        [data-theme="sunset"] {
            --bg-body: #12090e;
            --bg-card: rgba(36, 15, 26, 0.75);
            --border-color: rgba(244, 63, 94, 0.2);
            --text-main: #fff1f2;
            --text-muted: #fb7185;
            --accent-gradient: linear-gradient(135deg, #f43f5e 0%, #ea580c 100%);
            --accent-glow: rgba(244, 63, 94, 0.35);
            --accent-color: #f43f5e;
            --card-shadow: 0 20px 30px -10px rgba(244, 63, 94, 0.2);
        }

        [data-theme="light-pro"] {
            --bg-body: #f8fafc;
            --bg-card: rgba(255, 255, 255, 0.85);
            --border-color: rgba(0, 0, 0, 0.08);
            --text-main: #0f172a;
            --text-muted: #64748b;
            --accent-gradient: linear-gradient(135deg, #2563eb 0%, #0284c7 100%);
            --accent-glow: rgba(37, 99, 235, 0.25);
            --accent-color: #2563eb;
            --card-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            min-height: 100vh;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        /* Ambient Glow Background Mesh */
        .ambient-mesh {
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            pointer-events: none;
            z-index: -1;
            background: 
                radial-gradient(circle at 10% 20%, var(--accent-glow) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(14, 165, 233, 0.15) 0%, transparent 40%);
            filter: blur(80px);
        }

        /* Glass Cards */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.2);
        }

        .gradient-text {
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .form-control, .form-select {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 14px;
            padding: 0.75rem 1rem;
            transition: all 0.3s ease;
        }

        [data-theme="light-pro"] .form-control, 
        [data-theme="light-pro"] .form-select {
            background: rgba(255, 255, 255, 0.9);
        }

        .form-control:focus, .form-select:focus {
            background: rgba(0, 0, 0, 0.3);
            border-color: var(--accent-color);
            color: var(--text-main);
            box-shadow: 0 0 0 4px var(--accent-glow);
        }

        /* Hero Action Button */
        .btn-predict {
            background: var(--accent-gradient);
            border: none;
            color: #ffffff;
            font-weight: 700;
            padding: 1.1rem 2rem;
            border-radius: 16px;
            width: 100%;
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px var(--accent-glow);
        }

        .btn-predict:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 15px 25px -5px var(--accent-glow);
            color: #ffffff;
        }

        /* Theme Selector Pill */
        .theme-pill {
            cursor: pointer;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            display: inline-block;
            border: 2px solid #fff;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .theme-pill:hover {
            transform: scale(1.25);
        }

        /* Metric Dashboard Highlight Cards */
        .metric-box {
            padding: 1.5rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }

        /* Preset Chips */
        .preset-chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-muted);
            border-radius: 20px;
            padding: 0.35rem 0.85rem;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .preset-chip:hover {
            background: var(--accent-gradient);
            color: #fff;
            border-color: transparent;
        }

        /* Loading Overlay */
        .spinner-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(9, 13, 22, 0.85);
            backdrop-filter: blur(8px);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
            transition: opacity 0.3s ease;
        }

        .pulse-ring {
            animation: pulse 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: .7; transform: scale(1.05); }
        }
    </style>
</head>
<body class="py-4">

<div class="ambient-mesh"></div>

<div class="container-fluid px-lg-5">
    
    <!-- Navigation Header -->
    <header class="d-flex justify-content-between align-items-center mb-4 glass-card p-3 px-4">
        <div class="d-flex align-items-center gap-3">
            <div class="p-2 rounded-4 text-white d-flex align-items-center justify-content-center shadow-sm" style="background: var(--accent-gradient); width: 50px; height: 50px;">
                <i class="bi bi-heart-pulse-fill fs-3"></i>
            </div>
            <div>
                <h4 class="fw-extrabold mb-0 gradient-text">AI Healthcare Results Prediction</h4>
                <small style="color: var(--text-muted)" class="fw-medium">XGBoost Clinical Analytics Engine</small>
            </div>
        </div>
        
        <!-- Multi-Theme Selection Switcher -->
        <div class="d-flex align-items-center gap-2 bg-black bg-opacity-20 p-2 rounded-pill border border-secondary border-opacity-25">
            <span class="me-2 ms-2 fs-7 text-muted fw-semibold"><i class="bi bi-palette2 me-1"></i> Themes:</span>
            <span class="theme-pill" style="background: linear-gradient(135deg, #059669, #0284c7);" onclick="setTheme('emerald')" title="Emerald Bio-Tech"></span>
            <span class="theme-pill" style="background: linear-gradient(135deg, #ec4899, #8b5cf6);" onclick="setTheme('cyberpunk')" title="Cyberpunk Cyber-Med"></span>
            <span class="theme-pill" style="background: linear-gradient(135deg, #f43f5e, #ea580c);" onclick="setTheme('sunset')" title="Sunset Vibrant"></span>
            <span class="theme-pill" style="background: linear-gradient(135deg, #2563eb, #0284c7);" onclick="setTheme('light-pro')" title="Crisp Light Pro"></span>
        </div>
    </header>

    <div class="row g-4">
        <!-- Feature Inputs Panel -->
        <div class="col-lg-5">
            <div class="glass-card p-4 h-100 position-relative">
                
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="fw-bold mb-0 d-flex align-items-center gap-2">
                        <i class="bi bi-sliders2 text-primary"></i> Clinical Features
                    </h5>
                    <span class="fs-8 text-muted fw-medium">7 Input Variables</span>
                </div>

                <!-- Form Quick Presets -->
                <div class="mb-4">
                    <small class="text-muted d-block mb-2 fw-semibold fs-8">POPULATE QUICK DEMO PRESETS:</small>
                    <div class="d-flex flex-wrap gap-2">
                        <span class="preset-chip" onclick="loadPreset('cardiac')"><i class="bi bi-heart-break me-1"></i> Emergency Cardiac</span>
                        <span class="preset-chip" onclick="loadPreset('routine')"><i class="bi bi-check-circle me-1"></i> Standard Checkup</span>
                        <span class="preset-chip" onclick="loadPreset('urgent')"><i class="bi bi-hospital me-1"></i> High Billing Urgent</span>
                    </div>
                </div>

                <form id="predictionForm">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <label class="form-label small fw-semibold text-muted">Patient Age (Years)</label>
                            <input type="number" class="form-control fw-bold" id="Age" value="58" min="1" max="120" required>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label small fw-semibold text-muted">Gender</label>
                            <select class="form-select fw-semibold" id="Gender" required>
                                <option value="Female">Female</option>
                                <option value="Male" selected>Male</option>
                                <option value="Other">Other</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label small fw-semibold text-muted">Blood Type</label>
                            <select class="form-select fw-semibold" id="Blood Type" required>
                                <option value="A+">A+</option>
                                <option value="A-">A-</option>
                                <option value="B+">B+</option>
                                <option value="B-">B-</option>
                                <option value="AB+">AB+</option>
                                <option value="AB-">AB-</option>
                                <option value="O+" selected>O+</option>
                                <option value="O-">O-</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label small fw-semibold text-muted">Medical Condition</label>
                            <select class="form-select fw-semibold" id="Medical Condition" required>
                                <option value="Arthritis">Arthritis</option>
                                <option value="Asthma">Asthma</option>
                                <option value="Cancer">Cancer</option>
                                <option value="Diabetes">Diabetes</option>
                                <option value="Hypertension" selected>Hypertension</option>
                                <option value="Obesity">Obesity</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label small fw-semibold text-muted">Insurance Provider</label>
                            <select class="form-select fw-semibold" id="Insurance Provider" required>
                                <option value="Aetna">Aetna</option>
                                <option value="Blue Cross" selected>Blue Cross</option>
                                <option value="Cigna">Cigna</option>
                                <option value="Medicare">Medicare</option>
                                <option value="UnitedHealthcare">UnitedHealthcare</option>
                            </select>
                        </div>

                        <div class="col-md-6">
                            <label class="form-label small fw-semibold text-muted">Admission Type</label>
                            <select class="form-select fw-semibold" id="Admission Type" required>
                                <option value="Elective">Elective</option>
                                <option value="Emergency" selected>Emergency</option>
                                <option value="Urgent">Urgent</option>
                            </select>
                        </div>

                        <div class="col-12">
                            <label class="form-label small fw-semibold text-muted">Billing Amount ($ USD)</label>
                            <input type="number" step="0.01" class="form-control fw-bold" id="Billing Amount" value="34250.00" required>
                        </div>

                        <div class="col-12 mt-4">
                            <button type="submit" class="btn-predict">
                                <i class="bi bi-cpu me-2"></i> Compute XGBoost Diagnosis
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>

        <!-- Diagnostic Dashboard Output -->
        <div class="col-lg-7">
            <div class="glass-card p-4 h-100 position-relative" id="resultsCard">
                
                <!-- Loading Animation -->
                <div id="loadingOverlay" class="spinner-overlay d-none">
                    <div class="text-center">
                        <div class="spinner-border text-info pulse-ring" role="status" style="width: 3.5rem; height: 3.5rem;"></div>
                        <h6 class="mt-3 text-white fw-bold">Processing Inference Model...</h6>
                        <small class="text-muted">Evaluating Gradient Boosted Decision Trees</small>
                    </div>
                </div>

                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h5 class="fw-bold mb-0 d-flex align-items-center gap-2">
                        <i class="bi bi-activity text-success"></i> Predictive Analysis Dashboard
                    </h5>
                    <span class="badge bg-success bg-opacity-10 text-success border border-success border-opacity-25 px-3 py-2 rounded-pill">
                        <i class="bi bi-broadcast me-1"></i> Model Loaded
                    </span>
                </div>

                <!-- Primary Prediction Metric Callouts -->
                <div class="row g-3 mb-4">
                    <div class="col-md-7">
                        <div class="metric-box">
                            <small class="text-muted text-uppercase fw-bold fs-8">Predicted Diagnostic Result</small>
                            <div class="d-flex align-items-center gap-2 mt-2">
                                <h3 class="fw-extrabold mb-0" id="predClassResult">Awaiting Inference</h3>
                            </div>
                            <small class="text-muted mt-2 d-block" id="predClassDesc">Fill out the patient features and click calculate.</small>
                        </div>
                    </div>
                    <div class="col-md-5">
                        <div class="metric-box">
                            <small class="text-muted text-uppercase fw-bold fs-8">Prediction Probability</small>
                            <h2 class="fw-extrabold mt-2 mb-1" id="confidenceResult" style="color: var(--accent-color);">0.0%</h2>
                            <div class="progress mt-2" style="height: 6px; background: rgba(255,255,255,0.05);">
                                <div id="confidenceBar" class="progress-bar bg-success" style="width: 0%; transition: width 0.8s ease;"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Chart Selector Controls -->
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h6 class="fw-bold small text-muted text-uppercase mb-0">Multiclass Output Probabilities</h6>
                    <div class="btn-group btn-group-sm" role="group">
                        <button type="button" class="btn btn-outline-secondary active" id="btnBarChart" onclick="switchChart('bar')">Bar View</button>
                        <button type="button" class="btn btn-outline-secondary" id="btnRadarChart" onclick="switchChart('radar')">Radar Profile</button>
                    </div>
                </div>

                <!-- Dynamic Chart Container -->
                <div class="mb-4">
                    <div style="height: 230px; position: relative;">
                        <canvas id="analyticsChart"></canvas>
                    </div>
                </div>

                <!-- Evaluated Patient Profile Summary Table -->
                <div>
                    <h6 class="fw-bold small text-muted text-uppercase mb-2">Evaluated Patient Profile</h6>
                    <div class="table-responsive">
                        <table class="table table-sm table-borderless text-light align-middle mb-0" style="font-size: 0.85rem;">
                            <tbody id="featureSummaryTable">
                                <tr><td class="text-muted">No analysis calculated yet.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

            </div>
        </div>
    </div>
</div>

<script>
    let analyticsChart = null;
    let currentChartType = 'bar';
    let lastResponseData = null;

    // Preset Configurations
    const presets = {
        cardiac: { Age: 68, Gender: 'Male', "Blood Type": 'O-', "Medical Condition": 'Hypertension', "Insurance Provider": 'Medicare', "Admission Type": 'Emergency', "Billing Amount": 48200.00 },
        routine: { Age: 29, Gender: 'Female', "Blood Type": 'A+', "Medical Condition": 'Asthma', "Insurance Provider": 'Aetna', "Admission Type": 'Elective', "Billing Amount": 5400.00 },
        urgent: { Age: 52, Gender: 'Male', "Blood Type": 'B+', "Medical Condition": 'Diabetes', "Insurance Provider": 'Blue Cross', "Admission Type": 'Urgent', "Billing Amount": 28900.00 }
    };

    function loadPreset(key) {
        const p = presets[key];
        if (!p) return;
        for (const [id, val] of Object.entries(p)) {
            document.getElementById(id).value = val;
        }
    }

    function setTheme(themeName) {
        document.documentElement.setAttribute('data-theme', themeName);
        if (lastResponseData) renderChart(lastResponseData);
    }

    function switchChart(type) {
        currentChartType = type;
        document.getElementById('btnBarChart').classList.toggle('active', type === 'bar');
        document.getElementById('btnRadarChart').classList.toggle('active', type === 'radar');
        if (lastResponseData) renderChart(lastResponseData);
    }

    // Chart Renderer
    function renderChart(data) {
        const ctx = document.getElementById('analyticsChart').getContext('2d');
        const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-color').trim() || '#10b981';

        if (analyticsChart) {
            analyticsChart.destroy();
        }

        if (currentChartType === 'bar') {
            analyticsChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ["Low Risk (Class 0)", "Moderate Risk (Class 1)", "High Risk (Class 2)"],
                    datasets: [{
                        label: 'Softmax Probability',
                        data: data.probabilities,
                        backgroundColor: [
                            'rgba(16, 185, 129, 0.7)',
                            'rgba(245, 158, 11, 0.7)',
                            'rgba(239, 68, 68, 0.7)'
                        ],
                        borderColor: ['#10b981', '#f59e0b', '#ef4444'],
                        borderWidth: 2,
                        borderRadius: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0,
                            ticks: { color: '#94a3b8' },
                            grid: { color: 'rgba(255,255,255,0.05)' }
                        },
                        x: {
                            ticks: { color: '#94a3b8' },
                            grid: { display: false }
                        }
                    }
                }
            });
        } else {
            // Radar Feature Intensity View
            analyticsChart = new Chart(ctx, {
                type: 'radar',
                data: {
                    labels: ['Age Intensity', 'Gender Scale', 'Blood Index', 'Condition Severity', 'Insurance Tier', 'Billing Magnitude', 'Admission Urgency'],
                    datasets: [{
                        label: 'Normalized Feature Profile',
                        data: data.normalized_metrics,
                        backgroundColor: 'rgba(16, 185, 129, 0.25)',
                        borderColor: accentColor,
                        borderWidth: 2,
                        pointBackgroundColor: accentColor
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255,255,255,0.1)' },
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            pointLabels: { color: '#94a3b8', font: { size: 10 } },
                            ticks: { display: false, beginAtZero: true, max: 100 }
                        }
                    }
                }
            });
        }
    }

    // Form Submit Handler
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.remove('d-none');

        const payload = {
            "Age": document.getElementById('Age').value,
            "Gender": document.getElementById('Gender').value,
            "Blood Type": document.getElementById('Blood Type').value,
            "Medical Condition": document.getElementById('Medical Condition').value,
            "Insurance Provider": document.getElementById('Insurance Provider').value,
            "Billing Amount": document.getElementById('Billing Amount').value,
            "Admission Type": document.getElementById('Admission Type').value
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.status === "success") {
                lastResponseData = result;

                // Update Metric Display
                document.getElementById('predClassResult').innerText = result.prediction;
                document.getElementById('predClassResult').style.color = result.prediction_color;
                document.getElementById('predClassDesc').innerText = result.prediction_desc;

                const maxProb = (Math.max(...result.probabilities) * 100).toFixed(1);
                document.getElementById('confidenceResult').innerText = `${maxProb}%`;
                document.getElementById('confidenceBar').style.width = `${maxProb}%`;

                // Render Chart
                renderChart(result);

                // Render Summary Grid
                const tableBody = document.getElementById('featureSummaryTable');
                tableBody.innerHTML = '';
                
                let keys = Object.keys(result.feature_summary);
                for (let i = 0; i < keys.length; i += 2) {
                    let k1 = keys[i], v1 = result.feature_summary[k1];
                    let k2 = keys[i+1], v2 = k2 ? result.feature_summary[k2] : '';
                    
                    tableBody.innerHTML += `
                        <tr>
                            <td class="text-muted fw-semibold" style="width: 20%;">${k1}</td>
                            <td class="fw-bold text-light" style="width: 30%;">${v1}</td>
                            <td class="text-muted fw-semibold" style="width: 20%;">${k2 || ''}</td>
                            <td class="fw-bold text-light" style="width: 30%;">${v2 || ''}</td>
                        </tr>
                    `;
                }

            } else {
                alert("Prediction Error: " + result.error);
            }
        } catch (err) {
            alert("API Connection Error: " + err.message);
        } finally {
            setTimeout(() => {
                overlay.classList.add('d-none');
            }, 300);
        }
    });

    // Default Initialization
    window.addEventListener('DOMContentLoaded', () => {
        const dummyData = {
            probabilities: [0.25, 0.50, 0.25],
            normalized_metrics: [58, 66, 84, 80, 40, 68, 66]
        };
        renderChart(dummyData);
    });
</script>

</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
