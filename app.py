from flask import Flask, render_template, request, jsonify
from collections import deque
from datetime import datetime
from risk_engine import (
    parse_logs_from_text,
    categorise_logs,
    LogEntry,
    classify_log,
)

app = Flask(__name__)
live_events = deque(maxlen=100)

@app.route("/", methods=["GET", "POST"])
def index():
    analysis_result = None
    raw_logs = ""

    if request.method == "POST":
        raw_logs = request.form.get("raw_logs", "").strip()

        file = request.files.get("log_file")
        if file and file.filename:
            try:
                file_content = file.read().decode("utf-8", errors="ignore")
            except AttributeError:
                file_content = file.read()

            if raw_logs:
                raw_logs = raw_logs + "\n" + file_content
            else:
                raw_logs = file_content

        if not raw_logs:
            from sample_logs import SAMPLE_LOG_TEXT
            raw_logs = SAMPLE_LOG_TEXT

        logs = parse_logs_from_text(raw_logs)

        analysis_result = categorise_logs(logs, use_llm=False)

    return render_template(
        "index.html",
        analysis=analysis_result,
        raw_logs=raw_logs,
        live_events=list(live_events),
    )


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
   
    data = request.get_json(force=True) or {}

    log = LogEntry(
        timestamp=data.get("timestamp", ""),
        src_ip=data.get("src_ip", ""),
        dst_ip=data.get("dst_ip", ""),
        event_type=data.get("event_type", "UNKNOWN"),
        details=data.get("details", ""),
    )

    classified = classify_log(log, use_llm=True)

    ts = log.timestamp or datetime.utcnow().isoformat(timespec="seconds") + "Z"

    live_events.append({
        "timestamp": ts,
        "src_ip": classified.log.src_ip,
        "dst_ip": classified.log.dst_ip,
        "event_type": classified.log.event_type,
        "details": classified.log.details,
        "severity": classified.severity,
        "reason": classified.reason,
    })

    return jsonify({
        "severity": classified.severity,
        "reason": classified.reason,
    })


@app.route("/api/live-events", methods=["GET"])
def api_live_events():
 
    return jsonify(list(live_events))


if __name__ == "__main__":
    app.run(debug=True)

