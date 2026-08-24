# Held-Out Evaluation Scenario Ground Truth Definitions

These scenario definitions represent the locked evaluation protocol for testing both the object detector and the temporal event engine.

> **Evaluation Isolation Rule:** Ground truth scenario labels must remain frozen and must never be accessed or optimized during training or threshold sweeps.

---

## Scenario Taxonomy

| Scenario ID | Name | Visual Behavior | Expected Observable Event | Expected Diagnostic |
|---|---|---|---|---|
| **S01** | Normal Baseline | Examinee seated upright, looking at screen. | None (`INACTIVE`) | `OK` |
| **S02** | Natural Micro-movement | Natural head pitch/yaw within $\pm 10^\circ$. | None (`INACTIVE`) | `OK` |
| **S03** | Brief Glance Away | Quick glance at keyboard (<1.0s). | None (filtered by persistence) | `OK` |
| **S04** | Sustained Yaw Away | Sustained head turn left/right $>25^\circ$ for $>1.5\text{s}$. | `LOOKING_AWAY` | `OK` |
| **S05** | Sustained Pitch Away | Sustained head tilt up/down $>20^\circ$ for $>1.5\text{s}$. | `LOOKING_AWAY` | `OK` |
| **S06** | Phone Visible on Desk | Mobile phone resting on desk in view $>0.5\text{s}$. | `PHONE_DETECTED` | `OK` |
| **S07** | Phone in Hand | Mobile phone held and partially occluded. | `PHONE_DETECTED` | `OK` |
| **S08** | Second Person Present | Second individual standing in webcam view $>0.75\text{s}$. | `MULTIPLE_PERSONS` | `OK` |
| **S09** | Transient Bystander | Person walks quickly across background ($<0.4\text{s}$). | None (filtered by persistence) | `OK` |
| **S10** | Face Unavailable | Examinee leaves frame or covers face. | None (`INACTIVE`) | `POSE_UNAVAILABLE` |

---

## Ground Truth JSON Schema

```json
{
  "scenario_id": "S04",
  "session_id": "sess_webcam_001",
  "duration_seconds": 30.0,
  "ground_truth_intervals": [
    {
      "event_type": "LOOKING_AWAY",
      "start_sec": 5.0,
      "end_sec": 12.5,
      "notes": "Turned head 40 deg right to talk"
    }
  ]
}
```
