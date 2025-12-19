import time
import requests
API_URL = "http://127.0.0.1:5000/api/ingest"
LOG_FILE = r"scanner.log"
LOG_ENCODING = "utf-8"
READ_FROM_START_ONCE = True

def stream_from_start(path: str):
    with open(path, "r", encoding=LOG_ENCODING, errors="ignore") as f:
        for line in f:
            yield line.strip()
def follow(path: str):
    
    with open(path, "r", encoding=LOG_ENCODING, errors="ignore") as f:
        f.seek(0, 2) 
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            yield line.strip()
def build_event_from_line(line: str) -> dict:
    return {
        "timestamp": "",
        "src_ip": "",
        "dst_ip": "",
        "event_type": "",
        "details": line,
    }
def process_line(line: str):
    if not line:
        return
    event = build_event_from_line(line)
    try:
        r = requests.post(API_URL, json=event, timeout=5)
        r.raise_for_status()
        result = r.json()
        severity = result.get("severity", "UNKNOWN")
        reason = result.get("reason", "")
        print(f"[{severity}] {line}")
        print(f"  → {reason}\n")
    except Exception as e:
        print("Error sending to API:", e)
def main():
    print(f"Agent watching log file: {LOG_FILE}")
    print(f"Using encoding: {LOG_ENCODING}")
    print(f"Sending events to API: {API_URL}")
    if READ_FROM_START_ONCE:
        print("Mode: READ_FROM_START_ONCE (processing existing lines)\n")
        for line in stream_from_start(LOG_FILE):
            process_line(line)
        print("Finished existing lines, switching to LIVE TAIL mode.\n")
    print("Mode: LIVE TAIL (only new lines)\n")
    for line in follow(LOG_FILE):
        process_line(line)


if __name__ == "__main__":
    main()

