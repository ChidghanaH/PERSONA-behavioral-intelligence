# PERSONA: Contextual Behavioral Intelligence Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Executive Summary

**PERSONA** (Contextual Behavioral Intelligence Engine) is a cutting-edge machine learning system designed to predict, analyze, and model complex human behavioral patterns within organizational contexts. It combines multimodal data fusion, Temporal Graph Neural Networks (TGNNs), and explainable AI to generate real-time behavioral insights for HR analytics, organizational risk assessment, and talent management—all while maintaining strict access controls through **Permission-as-Code (PaC)** architecture.

### Key Differentiator

Every user, organization, and data usage scenario requires **explicit cryptographic authorization** from the creator, ensuring compliance, accountability, and controlled deployment. PERSONA cannot be used without proper permission grants.

---

## 🎯 Vision & Purpose

### Why PERSONA?

Traditional HR analytics tools operate reactively—flagging issues after escalation. PERSONA reimagines this through **predictive behavioral contextualization**: understanding not just *what* happened, but *why* it happened and *what's likely to happen next*.

### Core Purpose

- ✅ Enable ethical, transparent, legally compliant organizational behavioral understanding
- ✅ Provide early warning systems for attrition, disengagement, or performance decline
- ✅ Support data-driven HR decisions while maintaining individual privacy and dignity
- ✅ Create foundational platform for organizational network analysis and team dynamics modeling

### Forward-Looking Features

| Feature | Description |
|---------|-------------|
| **Explainable AI First** | Every prediction includes reasoning trails, not black-box outputs |
| **Privacy-by-Design** | Uses federated learning and differential privacy by default |
| **Dynamic Consent** | Users control exactly what data is analyzed |
| **Real-Time Adaptability** | Models retrain continuously, detecting shifts within hours |
| **Ethical Guardrails** | Built-in bias detection, fairness constraints, regulatory compliance |

---

## 🔑 Core Features

### 1. Permission-as-Code (PaC) System

The cornerstone of PERSONA's controlled access architecture:

```python
from persona.auth import PermissionManager, PermissionScope, AccessLevel

# Initialize permission system
pm = PermissionManager()

# Create cryptographically signed permission
grant = pm.create_permission(
    grantor="system_admin",
    grantee="analyst_001",
    resource="employee_behavioral_data",
    scopes=[
        PermissionScope.DATA_READ,
        PermissionScope.ANALYTICS_VIEW
    ],
    access_level=AccessLevel.CONTRIBUTOR,
    duration=timedelta(days=90),
    conditions={
        "department": "HR",
        "max_records": 10000,
        "anonymization_required": True
    }
)
```

**Three-Tier Authorization:**

1. **Creator-Level**: Define who accesses which org data, retention policies, compliance rules
2. **Organization-Level**: HR teams get granular controls over departments, opt-in/out, insights
3. **Individual-Level**: Employees view their profile, control data types, request explanations

### 2. Temporal Graph Neural Networks (TGNNs)

Core ML architecture for behavioral pattern recognition:

```python
from persona.models import TGNN, TGNNConfig

# Configure model
config = TGNNConfig(
    node_feature_dim=128,
    hidden_dim=256,
    num_gnn_layers=3,
    temporal_window=30  # days
)

model = TGNN(config)

# Get predictions
predictions = model(
    node_features=employee_vectors,
    edge_index=interaction_graph,
    edge_attr=interaction_features
)

print(predictions['attrition_risk'])      # 90-day flight risk
print(predictions['engagement_score'])    # Current engagement
print(predictions['collaboration_score']) # Team collaboration
print(predictions['anomaly_likelihood'])  # Behavioral anomalies
```

**Model Components:**
- Graph Attention Networks (GAT) for relationship encoding
- Temporal attention mechanisms for behavioral evolution
- Multi-task prediction heads for simultaneous outcomes
- Residual connections for stable training

### 3. Explainability Engine

Every prediction comes with:

- **SHAP Values**: Which features drove this prediction?
- **Influence Tracking**: Which past events contributed?
- **Counterfactual Explanations**: "If X had been different, outcome would be Y"
- **Fairness Diagnostics**: Is prediction biased by protected attributes?
- **Confidence Intervals**: How certain is this prediction?

### 4. Federated Learning for Privacy

Organizations train models locally without sharing raw data:

```python
from persona.federated import FederatedTrainer

trainer = FederatedTrainer(
    global_model=model,
    privacy_budget=0.5,  # Differential privacy epsilon
    num_rounds=10
)

# Each org trains locally
for org in organizations:
    local_updates = trainer.local_training(org.data)
    trainer.aggregate_updates(local_updates)
```

---

## 🏗️ Project Structure

```
PERSONA-behavioral-intelligence/
├── src/
│   └── persona/
│       ├── auth/              # Permission-as-Code system
│       │   └── permission_system.py
│       ├── models/            # ML models
│       │   └── tgnn.py       # Temporal GNN implementation
│       ├── explainability/   # Interpretability tools
│       ├── federated/        # Federated learning
│       ├── data/             # Data processing
│       └── api/              # REST API
├── requirements.txt          # Dependencies
├── setup.py                  # Installation
├── README.md                 # This file
└── examples/                 # Usage examples
```

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/ChidghanaH/PERSONA-behavioral-intelligence.git
cd PERSONA-behavioral-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PERSONA
pip install -e .
```

---

## 📊 Real-World Use Cases

### Use Case 1: Attrition Risk Prediction

**Scenario**: Mid-size tech company identifies flight-risk talent 90 days before resignation

**Indicators Analyzed**:
- Communication frequency decline
- Calendar engagement reduction
- Shift from collaborative to solo work
- Increased skill-development activities
- Performance review sentiment trajectory

**Outcome**: Proactive career development discussions prevent $150K+ replacement costs

### Use Case 2: Team Psychological Safety

**Scenario**: Consulting firm maintains team safety during remote transitions

**Metrics**:
- Voice equity (contribution equality)
- Cross-team collaboration patterns
- Response latency to peer messages
- Meeting participation diversity

**Impact**: Catches burnout early, prevents team dysfunction

### Use Case 3: Organizational Capability Planning

**Scenario**: Enterprise plans AI/ML initiative staffing

**Analysis**:
- Current skill graph and expertise mapping
- Skill diffusion patterns
- Project success correlation with skills
- Attrition risk by role

**ROI**: Prevent $2.4M in replacement/ramp-up costs

---

## 🔒 Permission & Security

### How PaC Works

1. **Permission Creation**: Admin creates cryptographically signed grant
2. **Signature Verification**: System validates signature before each access
3. **Expiration Check**: Permissions have time-limited validity
4. **Audit Trail**: All access logged immutably
5. **Revocation**: Permissions can be instantly revoked

### Privacy Guarantees

- **No raw data sharing** in federated mode
- **Differential privacy** adds statistical noise
- **K-anonymity** ensures no individual re-identification
- **GDPR/CCPA compliant** by design
- **Individual data deletion** honored within 30 days

---

## 💻 Quick Start Example

```python
from persona import PERSONA

# Initialize system
system = PERSONA(
    permission_key_path="./keys/creator_private.pem",
    model_config={"hidden_dim": 256}
)

# Grant permission to analyst
system.auth.create_permission(
    grantor="admin",
    grantee="analyst_jane",
    resource="org_behavioral_data",
    scopes=["read", "analyze"],
    duration_days=90
)

# Analyst runs prediction (with valid permission)
results = system.predict(
    user_id="analyst_jane",
    data_source="org_behavioral_data",
    prediction_type="attrition_risk"
)

print(results.summary())
print(results.explanation())  # SHAP values, counterfactuals
```

---

## 🧪 Technical Architecture

### ML Pipeline

1. **Data Ingestion** → Multi-modal data aggregation (communication, calendar, projects)
2. **Graph Construction** → Temporal interaction graphs
3. **Feature Engineering** → Node/edge embeddings
4. **Model Training** → TGNN with multi-task learning
5. **Prediction** → Behavioral outcomes with explanations
6. **Monitoring** → Drift detection and retraining

### Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | PyTorch, PyTorch Geometric |
| Graph Processing | NetworkX, DGL |
| Explainability | SHAP, LIME, Alibi |
| Privacy | Opacus, DiffPrivLib |
| Federated Learning | Flower, PySyft |
| Cryptography | Python Cryptography |
| API | FastAPI, Pydantic |
| Database | PostgreSQL, Redis |

---

## 📄 License & Usage

**MIT License** - See LICENSE file

### ⚠️ Important Usage Restriction

PERSONA requires explicit permission from the creator for any usage. To request access:

1. Contact: [your-email@domain.com]
2. Provide: Organization details, use case, data governance plan
3. Receive: Cryptographically signed permission grant
4. Use: Within granted scope and duration

**Unauthorized usage will be cryptographically blocked.**

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with description

---

## 📞 Contact & Support

- **Creator**: Chidghana Hemantharaju
- **GitHub**: [@ChidghanaH](https://github.com/ChidghanaH)
- **Issues**: [GitHub Issues](https://github.com/ChidghanaH/PERSONA-behavioral-intelligence/issues)

---

## 🌟 Why PERSONA Represents the Future

1. **Proactive vs Reactive**: Predicts issues before escalation
2. **Privacy-First**: Federated learning keeps data local
3. **Explainable**: No black-box decisions
4. **Permission-Controlled**: Cryptographic access control
5. **Ethical AI**: Built-in fairness and bias detection
6. **Cutting-Edge ML**: TGNNs for temporal behavioral analysis

---

**Built with ❤️ for ethical, transparent, and effective organizational intelligence**
