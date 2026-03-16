import base64
import json
from pathlib import Path


MODEL_JSON = Path("models/tfjs/model.json")
OUTPUT_JS = Path("assets/custom_model_embed.js")


def embed_tfjs_model():
    if not MODEL_JSON.exists():
        raise SystemExit(f"Missing TFJS model: {MODEL_JSON}")

    model = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    model_topology = model.get("modelTopology")
    weights_manifest = model.get("weightsManifest", [])
    if not weights_manifest:
        raise SystemExit("No weightsManifest in model.json")

    weights_block = weights_manifest[0]
    weight_specs = weights_block.get("weights", [])
    weight_paths = weights_block.get("paths", [])

    shards_b64 = []
    for rel in weight_paths:
        shard_path = MODEL_JSON.parent / rel
        data = shard_path.read_bytes()
        shards_b64.append(base64.b64encode(data).decode("ascii"))

    js = []
    js.append("// Auto-generated. Embeds TFJS model weights for offline use.")
    js.append("window.CAMGUARD_EMBEDDED_MODEL = {")
    js.append("  modelTopology: " + json.dumps(model_topology) + ",")
    js.append("  weightSpecs: " + json.dumps(weight_specs) + ",")
    js.append("  weightShards: [")
    for b64 in shards_b64:
        js.append('    "' + b64 + '",')
    js.append("  ],")
    js.append("  getHandler: function () {")
    js.append("    if (!window.tf || !tf.io || !tf.io.fromMemory) {")
    js.append('      throw new Error("TensorFlow.js IO handler not available");')
    js.append("    }")
    js.append("    function b64ToBytes(b64) {")
    js.append("      const bin = atob(b64);")
    js.append("      const len = bin.length;")
    js.append("      const bytes = new Uint8Array(len);")
    js.append("      for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);")
    js.append("      return bytes;")
    js.append("    }")
    js.append("    const parts = this.weightShards.map(b64ToBytes);")
    js.append("    let total = 0;")
    js.append("    parts.forEach(p => total += p.length);")
    js.append("    const merged = new Uint8Array(total);")
    js.append("    let offset = 0;")
    js.append("    parts.forEach(p => { merged.set(p, offset); offset += p.length; });")
    js.append("    return tf.io.fromMemory(this.modelTopology, this.weightSpecs, merged.buffer);")
    js.append("  }")
    js.append("};")

    OUTPUT_JS.write_text("\n".join(js), encoding="utf-8")
    return OUTPUT_JS


if __name__ == "__main__":
    out = embed_tfjs_model()
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
