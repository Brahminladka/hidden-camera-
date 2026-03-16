import argparse
import heapq
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260307
GENERATED_AT = 1772832000
DEFAULT_PACK = r"C:\Users\amani\Downloads\camguard_global_pack_500k.json"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "intelligence"
DEFAULT_RUNTIME_BRAIN = PROJECT_ROOT / "data" / "brains" / "runtime_brain.json"

PROFILES = {
    "clock_spy": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["indoor", "indoor", "mixed"], "hints": ["hidden", "clock", "wifi"], "visual": (0.70, 0.93), "lens": (0.58, 0.88), "network": (0.62, 0.90), "magnetic": (0.56, 0.84), "ir": (0.10, 0.48), "aliases": ["Clock Spy Camera", "Hidden Spy Camera", "Alarm Clock Camera"]},
    "button_pinhole": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["indoor", "mixed", "indoor"], "hints": ["pinhole", "button", "microcam"], "visual": (0.64, 0.87), "lens": (0.63, 0.93), "network": (0.52, 0.82), "magnetic": (0.50, 0.78), "ir": (0.06, 0.28), "aliases": ["Button Pinhole Camera", "Pinhole Spy Camera", "Button Camera"]},
    "usb_charger": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["indoor", "indoor", "mixed"], "hints": ["usb", "charger", "hidden"], "visual": (0.68, 0.92), "lens": (0.52, 0.82), "network": (0.62, 0.90), "magnetic": (0.56, 0.84), "ir": (0.08, 0.30), "aliases": ["USB Charger Spy Camera", "Wall Charger Camera", "USB Adapter Camera"]},
    "smoke_detector": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["indoor", "indoor"], "hints": ["smoke", "ceiling", "camera"], "visual": (0.74, 0.95), "lens": (0.60, 0.86), "network": (0.68, 0.94), "magnetic": (0.60, 0.88), "ir": (0.10, 0.36), "aliases": ["Smoke Detector Camera", "Ceiling Sensor Camera"]},
    "light_bulb": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "indoor", "mixed"], "hints": ["bulb", "tuya", "panorama"], "visual": (0.72, 0.94), "lens": (0.50, 0.80), "network": (0.74, 0.96), "magnetic": (0.58, 0.86), "ir": (0.10, 0.32), "aliases": ["Light Bulb Camera", "Panoramic Bulb Camera"]},
    "onvif_ip": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed", "outdoor"], "hints": ["onvif", "rtsp", "device_service"], "visual": (0.76, 0.96), "lens": (0.56, 0.84), "network": (0.80, 0.98), "magnetic": (0.62, 0.90), "ir": (0.12, 0.40), "aliases": ["ONVIF IP Camera", "RTSP Security Camera", "IP Surveillance Camera"]},
    "hisilicon_soc": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed"], "hints": ["hisilicon", "hisi", "xmeye"], "visual": (0.72, 0.92), "lens": (0.46, 0.74), "network": (0.80, 0.98), "magnetic": (0.62, 0.90), "ir": (0.08, 0.22), "aliases": ["HiSilicon Surveillance SoC", "HiSilicon Camera Board", "XMeye Embedded Camera"]},
    "tuya_general": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "indoor", "mixed"], "hints": ["tuya", "smartlife", "iot"], "visual": (0.70, 0.91), "lens": (0.46, 0.76), "network": (0.76, 0.96), "magnetic": (0.56, 0.84), "ir": (0.08, 0.24), "aliases": ["Tuya Smart Camera", "SmartLife Camera", "Tuya Security Camera"]},
    "esp32_hidden": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed", "indoor"], "hints": ["esp32", "espressif", "ai-thinker"], "visual": (0.64, 0.86), "lens": (0.48, 0.78), "network": (0.74, 0.96), "magnetic": (0.52, 0.80), "ir": (0.06, 0.18), "aliases": ["ESP32 Hidden IoT Camera", "ESP32-CAM Module", "Espressif Camera Node"]},
    "tuya_hidden": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed", "indoor"], "hints": ["tuya", "smartlife", "iot_camera"], "visual": (0.64, 0.86), "lens": (0.42, 0.70), "network": (0.76, 0.96), "magnetic": (0.54, 0.82), "ir": (0.06, 0.18), "aliases": ["Tuya Hidden IoT Camera", "Smart Socket Camera", "SmartLife Hidden Camera"]},
    "onvif_hidden": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed"], "hints": ["onvif", "rtsp", "hidden"], "visual": (0.66, 0.88), "lens": (0.44, 0.72), "network": (0.80, 0.98), "magnetic": (0.56, 0.84), "ir": (0.06, 0.18), "aliases": ["ONVIF Hidden Micro Camera", "RTSP Micro Camera"]},
    "hisilicon_hidden": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed"], "hints": ["hisilicon", "xmeye", "hidden"], "visual": (0.66, 0.88), "lens": (0.42, 0.70), "network": (0.80, 0.98), "magnetic": (0.56, 0.84), "ir": (0.06, 0.18), "aliases": ["HiSilicon Hidden IoT Camera", "XMeye Hidden IPC"]},
    "battery_spy": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["indoor", "mixed", "outdoor"], "hints": ["battery", "portable", "wifi"], "visual": (0.62, 0.86), "lens": (0.48, 0.80), "network": (0.54, 0.82), "magnetic": (0.50, 0.78), "ir": (0.08, 0.22), "aliases": ["Battery Spy Cam", "Rechargeable Spy Camera", "Power Bank Camera", "Portable Battery Camera"]},
    "rf_bug": {"category": "rf_transmitter", "deviceType": "rf_bug", "env": ["indoor", "mixed", "indoor"], "hints": ["rf", "transmitter", "wireless"], "visual": (0.50, 0.76), "lens": (0.24, 0.56), "network": (0.68, 0.94), "magnetic": (0.46, 0.74), "ir": (0.02, 0.10), "aliases": ["RF Transmitter Camera", "2.4GHz AV Bug Camera", "5.8GHz Wireless Camera Bug", "Long-Range RF Camera"]},
    "spy_mic": {"category": "microphone", "deviceType": "microphone", "env": ["indoor", "mixed", "indoor"], "hints": ["audio", "mic", "listen"], "visual": (0.40, 0.68), "lens": (0.01, 0.08), "network": (0.62, 0.92), "magnetic": (0.42, 0.68), "ir": (0.00, 0.04), "aliases": ["Wireless Spy Microphone", "GSM Listening Device", "Wi-Fi Voice Bug", "Power Adapter Microphone", "Wall Socket Microphone"]},
    "smartphone": {"category": "reference", "deviceType": "reference_device", "env": ["indoor", "mixed"], "hints": ["android", "iphone", "mobile"], "visual": (0.18, 0.44), "lens": (0.18, 0.40), "network": (0.08, 0.24), "magnetic": (0.14, 0.30), "ir": (0.00, 0.04), "aliases": ["Smartphone"]},
    "laptop": {"category": "reference", "deviceType": "reference_device", "env": ["indoor", "mixed"], "hints": ["laptop", "workstation", "wifi"], "visual": (0.16, 0.42), "lens": (0.14, 0.32), "network": (0.08, 0.24), "magnetic": (0.16, 0.34), "ir": (0.00, 0.04), "aliases": ["Laptop"]},
    "wifi_router": {"category": "reference", "deviceType": "reference_device", "env": ["indoor", "mixed"], "hints": ["router", "gateway", "dhcp"], "visual": (0.10, 0.24), "lens": (0.00, 0.04), "network": (0.10, 0.26), "magnetic": (0.12, 0.28), "ir": (0.00, 0.02), "aliases": ["WiFi Router"]},
    "television": {"category": "reference", "deviceType": "reference_device", "env": ["indoor"], "hints": ["tv", "display", "electronics"], "visual": (0.10, 0.26), "lens": (0.00, 0.04), "network": (0.08, 0.20), "magnetic": (0.12, 0.28), "ir": (0.00, 0.02), "aliases": ["Television"]},
    "bluetooth_speaker": {"category": "reference", "deviceType": "reference_device", "env": ["indoor", "mixed"], "hints": ["bluetooth", "speaker", "audio"], "visual": (0.10, 0.24), "lens": (0.00, 0.04), "network": (0.08, 0.20), "magnetic": (0.12, 0.28), "ir": (0.00, 0.02), "aliases": ["Bluetooth Speaker"]},
    "electric_fan": {"category": "reference", "deviceType": "reference_device", "env": ["indoor", "mixed"], "hints": ["fan", "motor", "appliance"], "visual": (0.08, 0.20), "lens": (0.00, 0.04), "network": (0.04, 0.16), "magnetic": (0.10, 0.26), "ir": (0.00, 0.02), "aliases": ["Electric Fan"]},
    "power_adapter": {"category": "reference", "deviceType": "reference_device", "env": ["indoor"], "hints": ["adapter", "charger", "power"], "visual": (0.08, 0.22), "lens": (0.00, 0.04), "network": (0.02, 0.12), "magnetic": (0.10, 0.26), "ir": (0.00, 0.02), "aliases": ["Power Adapter"]},
    "led_lamp": {"category": "reference", "deviceType": "reference_device", "env": ["indoor"], "hints": ["lamp", "led", "light"], "visual": (0.08, 0.22), "lens": (0.00, 0.04), "network": (0.04, 0.16), "magnetic": (0.10, 0.26), "ir": (0.00, 0.02), "aliases": ["LED Lamp"]},
    "nanny_cam": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["indoor", "indoor", "mixed"], "hints": ["babycam", "audio", "motion"], "visual": (0.66, 0.88), "lens": (0.46, 0.76), "network": (0.60, 0.86), "magnetic": (0.54, 0.80), "ir": (0.08, 0.22), "ai": (0.74, 0.94), "magValue": (52.0, 132.0), "span": (6.0, 16.0), "aliases": ["Nanny Cam", "Toy Camera", "Speaker Nanny Camera"]},
    "dash_cam": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["mixed", "outdoor"], "hints": ["dashcam", "vehicle", "recorder"], "visual": (0.60, 0.84), "lens": (0.44, 0.74), "network": (0.46, 0.74), "magnetic": (0.50, 0.76), "ir": (0.06, 0.18), "ai": (0.73, 0.93), "magValue": (70.0, 158.0), "span": (8.0, 18.0), "aliases": ["Dash Cam", "Rear Mirror Camera", "Vehicle Recorder"]},
    "body_cam": {"category": "surveillance", "deviceType": "wifi_camera", "env": ["mixed", "outdoor"], "hints": ["bodycam", "wearable", "record"], "visual": (0.58, 0.82), "lens": (0.48, 0.80), "network": (0.40, 0.66), "magnetic": (0.48, 0.72), "ir": (0.04, 0.14), "ai": (0.73, 0.92), "magValue": (42.0, 112.0), "span": (5.0, 12.0), "aliases": ["Body Worn Camera", "Wearable Camera", "Clip Body Camera"]},
    "dvr_cam": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed"], "hints": ["dvr", "nvr", "cctv"], "visual": (0.72, 0.94), "lens": (0.48, 0.74), "network": (0.78, 0.98), "magnetic": (0.60, 0.88), "ir": (0.08, 0.20), "ai": (0.78, 0.97), "magValue": (118.0, 218.0), "span": (10.0, 24.0), "aliases": ["Surveillance DVR Camera", "NVR Camera Node", "CCTV Recorder Head"]},
    "hikvision": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed", "outdoor"], "hints": ["hikvision", "isapi", "onvif"], "visual": (0.76, 0.96), "lens": (0.56, 0.84), "network": (0.82, 0.98), "magnetic": (0.62, 0.90), "ir": (0.10, 0.24), "ai": (0.80, 0.98), "magValue": (110.0, 202.0), "span": (10.0, 22.0), "aliases": ["Hikvision Camera", "Hikvision Mini Dome", "Hikvision Turret Camera"]},
    "dahua": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed", "outdoor"], "hints": ["dahua", "magicbox", "onvif"], "visual": (0.76, 0.96), "lens": (0.56, 0.84), "network": (0.82, 0.98), "magnetic": (0.62, 0.90), "ir": (0.10, 0.24), "ai": (0.80, 0.98), "magValue": (110.0, 202.0), "span": (10.0, 22.0), "aliases": ["Dahua Camera", "Dahua Eyeball Camera", "Dahua Cube Camera"]},
    "raspi": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed"], "hints": ["raspi", "libcamera", "mjpg-streamer"], "visual": (0.64, 0.88), "lens": (0.44, 0.72), "network": (0.70, 0.94), "magnetic": (0.52, 0.80), "ir": (0.06, 0.16), "ai": (0.75, 0.95), "magValue": (48.0, 118.0), "span": (5.0, 12.0), "aliases": ["Raspberry Pi Camera Node", "Raspberry Pi Streamer", "Pi Camera Module"]},
    "arducam": {"category": "surveillance", "deviceType": "iot_camera", "env": ["indoor", "mixed"], "hints": ["arducam", "spi-cam", "module"], "visual": (0.64, 0.88), "lens": (0.46, 0.74), "network": (0.64, 0.88), "magnetic": (0.52, 0.80), "ir": (0.06, 0.16), "ai": (0.75, 0.95), "magValue": (44.0, 112.0), "span": (5.0, 12.0), "aliases": ["Arducam Module", "Arducam SPI Module", "Arducam Camera Board"]},
    "gaming_console": {"category": "reference", "deviceType": "reference_device", "env": ["indoor"], "hints": ["console", "gaming", "media"], "visual": (0.10, 0.26), "lens": (0.00, 0.04), "network": (0.08, 0.20), "magnetic": (0.12, 0.30), "ir": (0.00, 0.02), "ai": (0.70, 0.90), "magValue": (84.0, 184.0), "span": (10.0, 22.0), "aliases": ["Gaming Console", "PlayStation Console", "Xbox Console", "Nintendo Console"]},
}

PACK_JOBS = [
    ("clock_spy", "Clock Spy Camera", 12000), ("button_pinhole", "Button Pinhole Camera", 14000),
    ("usb_charger", "USB Charger Spy Camera", 14000), ("smoke_detector", "Smoke Detector Camera", 12000),
    ("light_bulb", "Light Bulb Camera", 12000), ("onvif_ip", "ONVIF IP Camera", 10000),
    ("hisilicon_soc", "HiSilicon Surveillance SoC", 9000), ("tuya_general", "Tuya Smart Camera", 9000),
    ("esp32_hidden", "ESP32 Hidden IoT Camera", 10000), ("tuya_hidden", "Tuya Smart Camera", 6000),
    ("onvif_hidden", "ONVIF IP Camera", 2000), ("hisilicon_hidden", "HiSilicon Surveillance SoC", 2000),
    ("battery_spy", "Battery Spy Cam", 20000), ("rf_bug", "RF Transmitter Camera", 20000),
    ("spy_mic", "Spy Microphone", 20000), ("smartphone", "Smartphone", 5500),
    ("laptop", "Laptop", 5500), ("wifi_router", "WiFi Router", 5500),
    ("television", "Television", 5500), ("bluetooth_speaker", "Bluetooth Speaker", 5500),
    ("electric_fan", "Electric Fan", 5500), ("power_adapter", "Power Adapter", 5500),
    ("led_lamp", "LED Lamp", 5500),
]

SYNTH_JOBS = [
    ("nanny_cam", 5000), ("dash_cam", 3000), ("body_cam", 3000), ("dvr_cam", 4000),
    ("hikvision", 4000), ("dahua", 4000), ("raspi", 2500), ("arducam", 2500),
    ("gaming_console", 6000),
]


def r2(value):
    return round(value, 2)


def clamp(value, low, high):
    return max(low, min(high, value))


def allocate_quota(jobs, target):
    total = sum(count for _, count in jobs)
    raw = [(count / total) * target for _, count in jobs]
    base = [int(value) for value in raw]
    remainder = target - sum(base)
    order = sorted(
        range(len(jobs)),
        key=lambda idx: raw[idx] - int(raw[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        base[idx] += 1
    return {jobs[idx][0]: base[idx] for idx in range(len(jobs))}


def choose_env(rng, options):
    return rng.choice(options)


def base_label(label):
    if " Variant " in label:
        return label.split(" Variant ")[0]
    if " Ref " in label:
        return label.split(" Ref ")[0]
    return label


def alias_label(profile, source_label, idx, rng):
    aliases = profile.get("aliases") or [base_label(source_label)]
    alias = rng.choice(aliases)
    if " Variant " in source_label:
        suffix = source_label.split(" Variant ")[1]
        return f"{alias} Variant {suffix}"
    if " Ref " in source_label:
        suffix = source_label.split(" Ref ")[1]
        return f"{alias} Ref {suffix}"
    return f"{alias} Unit {idx:05d}"


def unique_hints(existing, extra):
    merged = list(existing) + list(extra)
    out = []
    seen = set()
    for item in merged:
        if item not in seen:
            seen.add(item)
            out.append(item)
        if len(out) == 5:
            break
    return out


def compute_scores(rng, profile, ai_confidence):
    visual = r2(clamp(ai_confidence + rng.uniform(-0.10, 0.06), *profile["visual"]))
    lens = r2(rng.uniform(*profile["lens"]))
    network = rng.uniform(*profile["network"])
    magnetic = rng.uniform(*profile["magnetic"])
    ir = rng.uniform(*profile["ir"])
    fusion = r2(
        visual * 0.35
        + lens * 0.25
        + network * 0.20
        + magnetic * 0.15
        + ir * 0.05
    )
    return visual, lens, fusion


def assign_risk(profile, fusion):
    if profile["category"] == "reference":
        return "medium" if fusion >= 0.36 else "low"
    if fusion >= 0.85:
        return "critical"
    if fusion >= 0.72:
        return "high"
    if fusion >= 0.56:
        return "medium"
    return "low"


def normalize_pack_entry(item, profile_key, idx, rng):
    profile = PROFILES[profile_key]
    ai_conf = r2(clamp(item["conf"], 0.70, 0.98))
    visual, lens, fusion = compute_scores(rng, profile, ai_conf)
    timestamp = int(item["ts"] / 1000) if item["ts"] > 10_000_000_000 else int(item["ts"])
    entry = {
        "label": alias_label(profile, item["label"], idx, rng),
        "category": profile["category"],
        "risk": assign_risk(profile, fusion),
        "magValue": r2(item["mag"]),
        "magSignature": [r2(item["sig"][0]), r2(item["sig"][1])],
        "aiConfidence": ai_conf,
        "visualScore": visual,
        "lensScore": lens,
        "fusionScore": fusion,
        "networkHints": unique_hints(item.get("net", []), profile["hints"]),
        "deviceType": profile["deviceType"],
        "environment": choose_env(rng, profile["env"]),
        "timestamp": timestamp,
    }
    return entry


def synthetic_label(profile, idx, rng):
    alias = rng.choice(profile["aliases"])
    region = rng.choice(["US", "EU", "AP", "LAT", "MEA"])
    rev = rng.choice(["R1", "R2", "R3", "MK2", "MK3"])
    return f"{alias} {region}-{rev}-{idx:05d}"


def build_synth_entry(profile_key, idx, rng):
    profile = PROFILES[profile_key]
    mag_value = r2(rng.uniform(*profile["magValue"]))
    span = rng.uniform(*profile["span"])
    ai_conf = r2(rng.uniform(*profile["ai"]))
    visual, lens, fusion = compute_scores(rng, profile, ai_conf)
    risk = assign_risk(profile, fusion)
    return {
        "label": synthetic_label(profile, idx, rng),
        "category": profile["category"],
        "risk": risk,
        "magValue": mag_value,
        "magSignature": [r2(max(0.1, mag_value - span)), r2(mag_value + span)],
        "aiConfidence": ai_conf,
        "visualScore": visual,
        "lensScore": lens,
        "fusionScore": fusion,
        "networkHints": profile["hints"][:],
        "deviceType": profile["deviceType"],
        "environment": choose_env(rng, profile["env"]),
        "timestamp": GENERATED_AT - 86400 + idx * 19,
    }


def write_wrapper(path, dataset_type, counts, distribution, entries):
    payload = {
        "datasetType": dataset_type,
        "schemaVersion": 1,
        "seed": SEED,
        "generatedAt": GENERATED_AT,
        "entryCount": len(entries),
        "counts": counts,
        "distribution": distribution,
        "knowledgeBase": entries,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2 if len(entries) <= 20000 else None, separators=(",", ":"))


def write_runtime_brain(path, entries):
    payload = {
        "version": 12,
        "totalScans": 0,
        "totalDetections": len(entries),
        "confirmedDetections": 0,
        "falseAlarms": 0,
        "adaptedGlintRatio": 1.6,
        "adaptedRiskThreshold": 35,
        "adaptedAIThreshold": 0.60,
        "generatedAt": GENERATED_AT,
        "knowledgeBase": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, separators=(",", ":"))


def load_buckets(pack_path):
    with pack_path.open("r", encoding="utf-8") as handle:
        knowledge_base = json.load(handle)["knowledgeBase"]
    buckets = defaultdict(list)
    for item in knowledge_base:
        buckets[base_label(item["label"])].append(item)
    rng = random.Random(SEED)
    for items in buckets.values():
        rng.shuffle(items)
    return buckets


def emit_global(output_path, pack_buckets):
    rng = random.Random(SEED)
    cursors = Counter()
    job_counts = [(f"pack:{profile}", count) for profile, _, count in PACK_JOBS] + [(f"synth:{profile}", count) for profile, count in SYNTH_JOBS]
    extended_quota = allocate_quota(job_counts, 20000)
    extended_entries = []
    extended_counts = Counter()
    core_heap = []
    fallback_heap = []
    global_counts = Counter()
    distribution = {
        "surveillance_general": 120000,
        "hidden_iot": 20000,
        "battery_powered": 20000,
        "rf_transmitter": 20000,
        "microphone": 20000,
        "reference_devices": 50000,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        handle.write('  "datasetType":"global",\n')
        handle.write('  "schemaVersion":1,\n')
        handle.write(f'  "seed":{SEED},\n')
        handle.write(f'  "generatedAt":{GENERATED_AT},\n')
        handle.write('  "entryCount":250000,\n')
        handle.write('  "counts":{"surveillance":160000,"rf_transmitter":20000,"microphone":20000,"reference":50000},\n')
        handle.write(f'  "distribution":{json.dumps(distribution, separators=(",", ":"))},\n')
        handle.write('  "knowledgeBase":[\n')

        first = True
        local_index = Counter()
        for profile_key, source_label, count in PACK_JOBS:
            items = pack_buckets[source_label]
            start = cursors[source_label]
            end = start + count
            if len(items) < end:
                raise ValueError(f"Not enough source entries for {source_label}: need {end}, found {len(items)}")
            cursors[source_label] = end
            job_key = f"pack:{profile_key}"
            for item in items[start:end]:
                local_index[profile_key] += 1
                entry = normalize_pack_entry(item, profile_key, local_index[profile_key], rng)
                global_counts[entry["category"]] += 1
                if not first:
                    handle.write(",\n")
                handle.write(json.dumps(entry, separators=(",", ":")))
                first = False
                if local_index[profile_key] <= extended_quota[job_key]:
                    extended_entries.append(entry)
                    extended_counts[entry["category"]] += 1
                if entry["category"] != "reference" and entry["risk"] == "critical":
                    key = (entry["fusionScore"], entry["aiConfidence"], entry["label"])
                    if len(core_heap) < 1000:
                        heapq.heappush(core_heap, (key, entry))
                    elif key > core_heap[0][0]:
                        heapq.heapreplace(core_heap, (key, entry))
                if entry["category"] != "reference":
                    key = (entry["fusionScore"], entry["aiConfidence"], entry["label"])
                    if len(fallback_heap) < 1000:
                        heapq.heappush(fallback_heap, (key, entry))
                    elif key > fallback_heap[0][0]:
                        heapq.heapreplace(fallback_heap, (key, entry))

        for profile_key, count in SYNTH_JOBS:
            job_key = f"synth:{profile_key}"
            for idx in range(1, count + 1):
                entry = build_synth_entry(profile_key, idx, rng)
                global_counts[entry["category"]] += 1
                handle.write(",\n")
                handle.write(json.dumps(entry, separators=(",", ":")))
                if idx <= extended_quota[job_key]:
                    extended_entries.append(entry)
                    extended_counts[entry["category"]] += 1
                if entry["category"] != "reference" and entry["risk"] == "critical":
                    key = (entry["fusionScore"], entry["aiConfidence"], entry["label"])
                    if len(core_heap) < 1000:
                        heapq.heappush(core_heap, (key, entry))
                    elif key > core_heap[0][0]:
                        heapq.heapreplace(core_heap, (key, entry))
                if entry["category"] != "reference":
                    key = (entry["fusionScore"], entry["aiConfidence"], entry["label"])
                    if len(fallback_heap) < 1000:
                        heapq.heappush(fallback_heap, (key, entry))
                    elif key > fallback_heap[0][0]:
                        heapq.heapreplace(fallback_heap, (key, entry))

        handle.write("\n  ]\n")
        handle.write("}\n")

    core_map = {item[1]["label"]: dict(item[1]) for item in core_heap}
    if len(core_map) < 1000:
        for _, entry in sorted(fallback_heap, key=lambda pair: pair[0], reverse=True):
            if entry["label"] in core_map:
                continue
            promoted = dict(entry)
            promoted["risk"] = "critical"
            core_map[promoted["label"]] = promoted
            if len(core_map) == 1000:
                break
    core_entries = sorted(core_map.values(), key=lambda item: (item["fusionScore"], item["aiConfidence"], item["label"]), reverse=True)[:1000]
    core_counts = Counter(entry["category"] for entry in core_entries)
    write_wrapper(
        output_path.with_name("core_signatures.json"),
        "core",
        dict(core_counts),
        {"critical_devices": 1000},
        core_entries,
    )
    write_wrapper(
        output_path.with_name("extended_signatures.json"),
        "extended",
        dict(extended_counts),
        distribution,
        extended_entries,
    )
    write_runtime_brain(output_path.parents[1] / "brains" / "runtime_brain.json", core_entries)
    return {
        "global_counts": dict(global_counts),
        "extended_count": len(extended_entries),
        "core_count": len(core_entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize and augment CamGuard intelligence datasets.")
    parser.add_argument("--pack", default=DEFAULT_PACK, help="Source reference pack JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated JSON files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = Path(args.pack)
    buckets = load_buckets(pack_path)
    summary = emit_global(output_dir / "global_intelligence.json", buckets)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
