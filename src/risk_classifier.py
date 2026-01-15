import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
# ... rest of your code

from config import LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD

def classify_risk(score):
    if score < LOW_RISK_THRESHOLD:
        return "LOW"
    elif score < HIGH_RISK_THRESHOLD:
        return "MEDIUM"
    else:
        return "HIGH"
