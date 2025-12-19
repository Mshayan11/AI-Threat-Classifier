from risk_engine import LogEntry, classify_log, categorise_logs
malware_log = LogEntry(
    timestamp="2025-12-11T12:00:00Z",
    src_ip="10.0.0.8",
    dst_ip="10.0.0.15",
    event_type="MALWARE",
    details="Virus:DOS/EICAR_Test_File detected on C:\\Users\\You\\Desktop\\eicar.com"
)
brute_log = LogEntry(
    timestamp="2025-12-11T12:05:00Z",
    src_ip="203.0.113.10",
    dst_ip="10.0.0.20",
    event_type="BRUTE_FORCE",
    details="Multiple login failures detected from external IP. Possible password guessing."
)
ok_log = LogEntry(
    timestamp="2025-12-11T12:10:00Z",
    src_ip="10.0.0.5",
    dst_ip="10.0.0.10",
    event_type="LOGIN_SUCCESS",
    details="User john.doe successfully authenticated."
)

logs = [malware_log, brute_log, ok_log]

print("=== Individual classify_log (rule-based + ML) ===")
for i, log in enumerate(logs, start=1):
    result = classify_log(log, use_llm=False)
    print(f"\nLog {i}:")
    print(f"  Severity: {result.severity}")
    print(f"  Reason:   {result.reason}")

print("\n=== categorise_logs summary ===")
summary = categorise_logs(logs, use_llm=False)
print(summary["summary"])

