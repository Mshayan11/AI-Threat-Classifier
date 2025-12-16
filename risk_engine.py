from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import ipaddress
import re
import os
import json
from unsw_ml import ml_attack_probability_from_log

# Robust import: works with both old and new openai package versions
try:
    import openai  # type: ignore
except ImportError:
    openai = None  

IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass
class LogEntry:
    timestamp: str
    src_ip: str
    dst_ip: str
    event_type: str
    details: str


@dataclass
class ClassifiedLog:
    log: LogEntry
    severity: str  # "RED", "YELLOW", "GREEN"
    reason: str




def infer_event_type_from_text(text: str) -> str:
    """
    Try to guess an event_type from a free-text log line.
    """
    t = text.lower()

    # HIGH SEVERITY / MALWARE INDICATORS
    if any(k in t for k in [
        "virus:",
        " virus ",
        "malware",
        "trojan",
        "ransomware",
        "worm ",
        "rootkit",
        "backdoor",
        "eicar",  # EICAR test file
        "threat detected",
        "threat found",
        "detected malware",
        "pua:",
        "potentially unwanted",
    ]):
        return "MALWARE"

    # Data exfiltration style wording
    if any(k in t for k in ["data exfil", "exfiltration", "large outbound transfer"]):
        return "DATA_EXFIL"

    # MEDIUM SEVERITY INDICATORS
    if any(k in t for k in [
        "brute force",
        "failed password",
        "failed login",
        "multiple login failures",
        "invalid user",
        "account lockout",
    ]):
        return "BRUTE_FORCE"

    if any(k in t for k in ["portscan", "port scan", "nmap scan"]):
        return "PORT_SCAN"

    if any(k in t for k in ["sql injection", "sqli"]):
        return "SQLI"

    if any(k in t for k in ["xss", "cross-site scripting", "cross site scripting"]):
        return "XSS"

    # LOW SEVERITY / INFO
    if "phish" in t:
        return "PHISHING"

    if "login success" in t or "authentication succeeded" in t:
        return "LOGIN_SUCCESS"

    return "UNKNOWN"

#  Parsing logs from text

def parse_logs_from_text(text: str) -> List[LogEntry]:
    
    logs: List[LogEntry] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # CSV with commas
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            while len(parts) < 5:
                parts.append("")
            timestamp, src_ip, dst_ip, event_type, details = parts[:5]

            if not event_type:
                event_type = infer_event_type_from_text(details)

            logs.append(LogEntry(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                event_type=event_type.upper(),
                details=details or line,
            ))
            continue

        # Free-text line
        ips = IP_REGEX.findall(line)
        src_ip = ips[0] if len(ips) >= 1 else ""
        dst_ip = ips[1] if len(ips) >= 2 else ""

        event_type = infer_event_type_from_text(line)

        logs.append(LogEntry(
            timestamp="",
            src_ip=src_ip,
            dst_ip=dst_ip,
            event_type=event_type.upper(),
            details=line,
        ))

    return logs


def is_external_ip(ip: str) -> bool:
    if not ip:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_reserved
        or ip_obj.is_link_local
    )

#  ML hybrid helper

def _apply_ml_hybrid_decision(
    log: LogEntry,
    severity: str,
    reason: str
) -> Tuple[str, str]:
    
    prob_attack = ml_attack_probability_from_log(log)

    if prob_attack is None:
        return severity, reason

    ml_text = f"UNSW-trained ML model attack probability: {prob_attack:.2f}."
    reason = (reason + " " + ml_text).strip() if reason else ml_text

    if prob_attack >= 0.8 and severity != "RED":
        severity = "RED"
        reason += " ML model is highly confident this is an attack."
    elif 0.4 <= prob_attack < 0.8 and severity == "GREEN":
        severity = "YELLOW"
        reason += " ML model indicates moderate attack likelihood."

    return severity, reason

#  Rule-based scoring (fallback)

def score_log(log: LogEntry) -> ClassifiedLog:
    
    event = log.event_type
    details_lower = log.details.lower()

    score = 0
    reasons: List[str] = []

    high_events = ["MALWARE", "RANSOMWARE", "ROOTKIT", "C2_BEACON", "DATA_EXFIL"]
    medium_events = ["BRUTE_FORCE", "PORT_SCAN", "SQLI", "XSS", "PRIV_ESC"]
    low_events = ["PHISHING", "LOGIN_SUCCESS", "NORMAL_TRAFFIC"]

    if event in high_events:
        score += 90
        reasons.append(f"Event type '{event}' is typically critical.")
    elif event in medium_events:
        score += 60
        reasons.append(f"Event type '{event}' is typically medium risk.")
    elif event in low_events:
        score += 20
        reasons.append(f"Event type '{event}' is typically low risk.")
    else:
        score += 30
        reasons.append(f"Unknown event type '{event}' treated as suspicious until verified.")

    if is_external_ip(log.src_ip):
        score += 15
        reasons.append("Source IP appears to be external to the organisation.")

    critical_indicators = [
        "malware", "ransomware", "trojan", "exploit", "c2",
        "data exfiltration", "data exfil", "virus:", "eicar"
    ]
    medium_indicators = [
        "failed login", "failed password", "multiple login failures",
        "password guessing", "port scan", "portscan"
    ]

    if any(term in details_lower for term in critical_indicators):
        score += 30
        reasons.append("Details contain indicators of serious compromise.")
    elif any(term in details_lower for term in medium_indicators):
        score += 15
        reasons.append("Details contain indicators of suspicious activity.")

    score = max(0, min(score, 100))

    if score >= 70:
        severity = "RED"
        reasons.append("Overall score ≥ 70: classified as HIGH severity.")
    elif score >= 40:
        severity = "YELLOW"
        reasons.append("Overall score between 40–69: classified as MEDIUM severity.")
    else:
        severity = "GREEN"
        reasons.append("Overall score < 40: classified as LOW severity.")

    reason_str = " ".join(reasons)

    # Hybrid step: adjust severity + explanation using UNSW ML model
    severity, reason_str = _apply_ml_hybrid_decision(log, severity, reason_str)

    return ClassifiedLog(log=log, severity=severity, reason=reason_str)

#  LLM-based classification

def _prepare_log_for_classification(log: LogEntry) -> LogEntry:
    
    if not log.event_type or log.event_type.upper() == "UNKNOWN":
        inferred = infer_event_type_from_text(log.details)
        if inferred and inferred != "UNKNOWN":
            return LogEntry(
                timestamp=log.timestamp,
                src_ip=log.src_ip,
                dst_ip=log.dst_ip,
                event_type=inferred.upper(),
                details=log.details,
            )
    return log


def classify_with_llm(log: LogEntry) -> ClassifiedLog:
    
    if openai is None:
        raise RuntimeError("openai package is not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    payload = {
        "timestamp": log.timestamp,
        "src_ip": log.src_ip,
        "dst_ip": log.dst_ip,
        "event_type_hint": log.event_type,
        "details": log.details,
    }

    system_message = (
        "You are an experienced SOC Level 2 analyst. "
        "Given a single log event, decide the security severity and explain why.\n"
        "Valid severities: LOW, MEDIUM, HIGH.\n"
        "Respond ONLY as a JSON object with keys: severity, reason, event_type.\n"
        "event_type should be a short label like MALWARE, BRUTE_FORCE, PORT_SCAN, PHISHING, DATA_EXFIL, etc."
    )

    user_message = "Classify the following log event:\n\n" + json.dumps(payload, ensure_ascii=False)

    # New-style client (openai>=1.0.0) exposes 'OpenAI'
    if hasattr(openai, "OpenAI"):
        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content
    else:
        # Old-style client – backwards compatible API
        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        content = completion.choices[0].message["content"]

    data = json.loads(content)

    sev_str = str(data.get("severity", "LOW")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    event_type = str(data.get("event_type", "")).strip().upper() or log.event_type

    if sev_str.startswith("high"):
        severity = "RED"
    elif sev_str.startswith("med"):
        severity = "YELLOW"
    else:
        severity = "GREEN"

    if not reason:
        reason = "Classified by LLM based on the log content."
    else:
        reason += " (classified by LLM model)."

    if not event_type or event_type == "UNKNOWN":
        event_type = infer_event_type_from_text(log.details).upper()

    new_log = LogEntry(
        timestamp=log.timestamp,
        src_ip=log.src_ip,
        dst_ip=log.dst_ip,
        event_type=event_type,
        details=log.details,
    )

    return ClassifiedLog(log=new_log, severity=severity, reason=reason)


def classify_log(log: LogEntry, use_llm: bool = False) -> ClassifiedLog:
    
    prepared = _prepare_log_for_classification(log)

    if use_llm:
        try:
            result = classify_with_llm(prepared)
        except Exception as e:
            fallback = score_log(prepared)
            fallback.reason = f"{fallback.reason} (LLM unavailable: {e})"
            return fallback

        # Apply ML hybrid decision on top of LLM output
        new_severity, new_reason = _apply_ml_hybrid_decision(
            result.log, result.severity, result.reason
        )
        return ClassifiedLog(log=result.log, severity=new_severity, reason=new_reason)

    else:
        # Rule-based path already calls _apply_ml_hybrid_decision inside score_log()
        return score_log(prepared)


def categorise_logs(logs: List[LogEntry], use_llm: bool = False) -> Dict[str, Any]:
    
    classified: List[ClassifiedLog] = [
        classify_log(log, use_llm=use_llm) for log in logs
    ]

    summary = {
        "total": len(classified),
        "red": sum(1 for c in classified if c.severity == "RED"),
        "yellow": sum(1 for c in classified if c.severity == "YELLOW"),
        "green": sum(1 for c in classified if c.severity == "GREEN"),
    }

    return {
        "summary": summary,
        "classified_logs": classified,
    }
