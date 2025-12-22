from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os
import io
import base64
import math
import random
from collections import Counter
from PIL import Image

app = Flask(__name__)

# ==============================================================================
# 1. DATABASE & CONFIG
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILENAME = os.path.join(BASE_DIR, 'sbox_data_full.json')
SBOX_CANDIDATES = []


def load_database():
    global SBOX_CANDIDATES
    if os.path.exists(DB_FILENAME):
        try:
            with open(DB_FILENAME, 'r') as f:
                data = json.load(f)
                SBOX_CANDIDATES = data.get('candidates', [])
        except:
            SBOX_CANDIDATES = []


load_database()


# ==============================================================================
# 2. CORE LOGIC & METRICS
# ==============================================================================

# --- GF(2^8) & AFFINE ---
def gf_inverse(val):
    if val == 0: return 0
    res = 1;
    base = val;
    exp = 254
    for _ in range(8):
        if (exp & 1): res = gf_mult(res, base)
        base = gf_mult(base, base)
        exp >>= 1
    return res


def gf_mult(a, b):
    p = 0
    for _ in range(8):
        if (b & 1): p ^= a
        hi_bit = (a & 0x80)
        a <<= 1
        if hi_bit: a ^= 0x11B
        b >>= 1
    return p & 0xFF


def apply_affine(byte_val, matrix_flat, constant_flat):
    bits = [(byte_val >> i) & 1 for i in range(8)]
    output_bits = [0] * 8
    for row in range(8):
        val = constant_flat[7 - row]
        for col in range(8):
            mat_bit = matrix_flat[(row * 8) + (7 - col)]
            val ^= (mat_bit & bits[col])
        output_bits[row] = val
    res = 0
    for i in range(8):
        if output_bits[i]: res |= (1 << i)
    return res


# --- S-BOX METRICS ---
def fast_walsh_transform(func):
    n = len(func);
    wf = list(func)
    if all(x in [0, 1] for x in wf): wf = [1 if x == 0 else -1 for x in wf]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = wf[j];
                y = wf[j + h]
                wf[j] = x + y;
                wf[j + h] = x - y
        h *= 2
    return wf


def get_bit_count(n): return bin(n).count('1')


def calculate_nl(sbox):
    min_nl = 256
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(256)]
        spec = fast_walsh_transform(func)
        nl = (256 - max(abs(v) for v in spec)) // 2
        if nl < min_nl: min_nl = nl
    return int(min_nl)


def calculate_sac(sbox):
    total_sac = 0;
    count = 0
    for i in range(256):
        out = sbox[i]
        for b in range(8):
            diff = out ^ sbox[i ^ (1 << b)]
            total_sac += get_bit_count(diff)
            count += 1
    return (total_sac / count) / 8.0


def calculate_bic_nl(sbox):
    min_bic = 256
    for i in range(8):
        for j in range(i + 1, 8):
            func = [((sbox[x] >> i) & 1) ^ ((sbox[x] >> j) & 1) for x in range(256)]
            spec = fast_walsh_transform(func)
            nl = (256 - max(abs(v) for v in spec)) // 2
            if nl < min_bic: min_bic = nl
    return int(min_bic)


def calculate_bic_sac(sbox):
    total = 0;
    cnt = 0
    for i in range(8):
        for j in range(i + 1, 8):
            for inp in range(256):
                val = ((sbox[inp] >> i) & 1) ^ ((sbox[inp] >> j) & 1)
                for b in range(8):
                    diff_inp = inp ^ (1 << b)
                    val_diff = ((sbox[diff_inp] >> i) & 1) ^ ((sbox[diff_inp] >> j) & 1)
                    if val != val_diff: total += 1
                    cnt += 1
    return total / cnt if cnt > 0 else 0


def calculate_lap(sbox):
    max_bias = 0
    for i in range(1, 256):
        func = [(get_bit_count(sbox[x] & i) % 2) for x in range(256)]
        spec = fast_walsh_transform(func)
        curr = max(abs(v) for v in spec)
        if curr > max_bias: max_bias = curr
    return max_bias / 512.0


def calculate_dap(sbox):
    max_diff = 0
    for dx in range(1, 256):
        counts = Counter([sbox[x] ^ sbox[x ^ dx] for x in range(256)])
        curr = max(counts.values())
        if curr > max_diff: max_diff = curr
    return max_diff / 256.0


def calculate_du(sbox): return int(calculate_dap(sbox) * 256)


def calculate_ad(sbox):
    max_deg = 0
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(256)]
        anf = list(func)
        for step in range(1, 256):
            if step & (step - 1) == 0:
                for j in range(0, 256, step * 2):
                    for k in range(j, j + step): anf[k + step] ^= anf[k]
        deg = 0
        for x in range(256):
            if anf[x] == 1 and get_bit_count(x) > deg: deg = get_bit_count(x)
        if deg > max_deg: max_deg = deg
    return int(max_deg)


def calculate_to(sbox):
    N = 256;
    max_to_val = 0
    for beta in range(1, N):
        func = [(get_bit_count(sbox[x] & beta) % 2) for x in range(N)]
        W = fast_walsh_transform(func)
        P = [w ** 2 for w in W]
        AC_raw = fast_walsh_transform(P)
        AC = [val // N for val in AC_raw]
        sum_abs_ac = sum(abs(val) for val in AC[1:])
        val = N - (sum_abs_ac / (N - 1))
        norm_val = val / N
        if norm_val > max_to_val: max_to_val = norm_val
    return round(max_to_val, 5)


def calculate_ci(sbox):
    min_ci = 8
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(256)]
        spec = fast_walsh_transform(func)
        curr = 0
        for m in range(1, 9):
            if any(spec[w] != 0 for w in range(1, 256) if get_bit_count(w) == m): break
            curr = m
        if curr < min_ci: min_ci = curr
    return int(min_ci)


def calculate_fixed_points(sbox): return sum(1 for x in range(256) if sbox[x] == x)


def calculate_cycles(sbox):
    visited = [False] * 256;
    cycles = []
    for i in range(256):
        if not visited[i]:
            curr = i;
            length = 0
            while not visited[curr]: visited[curr] = True; curr = sbox[curr]; length += 1
            cycles.append(length)
    return {"max": max(cycles) if cycles else 0}


def calculate_strength_value(nl, sac, bic_nl, bic_sac):
    sv = (120 - nl) + abs(0.5 - sac) + (120 - bic_nl) + abs(0.5 - bic_sac)
    return round(sv, 6)


def calculate_all_metrics(sbox):
    sbox = [int(x) for x in sbox]
    val_nl = calculate_nl(sbox)
    val_sac = calculate_sac(sbox)
    val_bic_nl = calculate_bic_nl(sbox)
    val_bic_sac = calculate_bic_sac(sbox)
    return {
        "NL": val_nl, "SAC": val_sac, "BIC_NL": val_bic_nl, "BIC_SAC": val_bic_sac,
        "LAP": calculate_lap(sbox), "DAP": calculate_dap(sbox), "DU": calculate_du(sbox),
        "AD": calculate_ad(sbox), "TO": calculate_to(sbox), "CI": calculate_ci(sbox),
        "FIXED": calculate_fixed_points(sbox), "CYC_MAX": calculate_cycles(sbox)['max'],
        "SV_PAPER": calculate_strength_value(val_nl, val_sac, val_bic_nl, val_bic_sac)
    }


# ==============================================================================
# 3. ADVANCED IMAGE ANALYTICS (ENTROPY, CORRELATION, NPCR, UACI)
# ==============================================================================

def calculate_entropy(data_bytes):
    """Shannon Entropy"""
    if not data_bytes: return 0
    counts = Counter(data_bytes)
    total_len = len(data_bytes)
    entropy = 0
    for count in counts.values():
        p = count / total_len
        if p > 0: entropy -= p * math.log2(p)
    return round(entropy, 5)


def calculate_correlation(data_bytes, w, h):
    """Calculates Adjacent Pixel Correlation (Horizontal, Vertical, Diagonal)"""
    # Convert to numpy array for speed
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    # Resize if array size matches w*h (handle potential padding/headers issue simply)
    if len(arr) != w * h:
        # If size mismatch (e.g. RGB flattened), take first channel or grayscale equivalent
        # For simplicity in this demo, we limit to first N pixels that fit
        size = w * h
        if len(arr) > size:
            arr = arr[:size]
        elif len(arr) < size:
            return {"H": 0, "V": 0, "D": 0}

    img_matrix = arr.reshape((h, w))

    # Helper for correlation coef
    def get_corr(x, y):
        if len(x) == 0: return 0
        return np.corrcoef(x, y)[0, 1]

    # Select N random pairs (limit to 3000 pairs for performance)
    N = 3000

    # Horizontal
    x_h, y_h = [], []
    for _ in range(N):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 2)
        x_h.append(img_matrix[r, c])
        y_h.append(img_matrix[r, c + 1])

    # Vertical
    x_v, y_v = [], []
    for _ in range(N):
        r = random.randint(0, h - 2)
        c = random.randint(0, w - 1)
        x_v.append(img_matrix[r, c])
        y_v.append(img_matrix[r + 1, c])

    # Diagonal
    x_d, y_d = [], []
    for _ in range(N):
        r = random.randint(0, h - 2)
        c = random.randint(0, w - 2)
        x_d.append(img_matrix[r, c])
        y_d.append(img_matrix[r + 1, c + 1])

    return {
        "H": round(get_corr(x_h, y_h), 5),
        "V": round(get_corr(x_v, y_v), 5),
        "D": round(get_corr(x_d, y_d), 5)
    }


def calculate_npcr_uaci(orig_bytes, cipher_bytes):
    """Calculates NPCR and UACI between two images"""
    arr1 = np.frombuffer(orig_bytes, dtype=np.uint8)
    arr2 = np.frombuffer(cipher_bytes, dtype=np.uint8)

    # Ensure same length
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    # NPCR: Rate of pixel change
    diff_count = np.sum(arr1 != arr2)
    npcr = (diff_count / min_len) * 100

    # UACI: Unified Average Changing Intensity
    abs_diff = np.sum(np.abs(arr1.astype(int) - arr2.astype(int)))
    uaci = (abs_diff / (255 * min_len)) * 100

    return round(npcr, 4), round(uaci, 4)


# ==============================================================================
# 4. ROUTING
# ==============================================================================

@app.route('/')
def home(): return render_template('index.html')


@app.route('/api/sbox_list')
def sbox_list():
    lst = [{"index": i, "name": c.get('name'), "id": c.get('id')} for i, c in enumerate(SBOX_CANDIDATES)]
    return jsonify(lst)


@app.route('/api/get_sbox/<int:idx>')
def get_sbox(idx):
    if 0 <= idx < len(SBOX_CANDIDATES):
        return jsonify({"status": "success", "sbox": SBOX_CANDIDATES[idx].get('sbox', []),
                        "matrix": SBOX_CANDIDATES[idx].get('matrix', None)})
    return jsonify({"error": "Not found"}), 404


@app.route('/api/generate_custom', methods=['POST'])
def generate_custom():
    try:
        d = request.json
        matrix = d.get('matrix');
        constant = d.get('constant')
        if not matrix or len(matrix) != 64 or not constant or len(constant) != 8: return jsonify(
            {"error": "Invalid input"}), 400
        gen_sbox = [apply_affine(gf_inverse(i), matrix, constant) for i in range(256)]
        return jsonify({"status": "success", "sbox": gen_sbox})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        sbox = request.json.get('sbox')
        if not sbox or len(sbox) != 256: return jsonify({"error": "Invalid Data"}), 400
        return jsonify({"status": "success", "metrics": calculate_all_metrics(sbox)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def encrypt_cbc_bytes(data, sbox):
    out = bytearray();
    prev = 0
    for b in data: val = sbox[b ^ prev]; out.append(val); prev = val
    return out


def decrypt_cbc_bytes(data, sbox):
    inv = [0] * 256
    for i, v in enumerate(sbox): inv[v] = i
    out = bytearray();
    prev = 0
    for b in data: val = inv[b] ^ prev; out.append(val); prev = b
    return out


@app.route('/api/process_text', methods=['POST'])
def process_text():
    try:
        d = request.json;
        mode = d['mode'];
        sbox = d['sbox']
        if mode == 'encrypt':
            res = encrypt_cbc_bytes(d['text'].encode(), sbox).hex().upper()
        else:
            try:
                b = bytes.fromhex(d['text'].replace(' ', '')); res = decrypt_cbc_bytes(b, sbox).decode(errors='ignore')
            except:
                res = "[Error: Invalid Hex]"
        return jsonify({"status": "success", "result": res})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        f = request.files['image']
        sbox_str = request.form.get('sbox')
        if not sbox_str: return jsonify({"error": "S-Box missing"}), 400
        sbox = [int(x) for x in sbox_str.split(',')]
        mode = request.form.get('mode')

        img = Image.open(f).convert('L')  # Convert to Grayscale for simpler analysis (common in papers)
        # OR keep RGB but flatten. Let's use Grayscale ('L') to strictly follow standard crypto metrics on pixel values.
        # If you want RGB, change 'L' to 'RGB'.

        orig_bytes = img.tobytes()
        w, h = img.size

        if mode == 'encrypt':
            proc_bytes = encrypt_cbc_bytes(orig_bytes, sbox)
            # Calculate Metrics only on Encryption
            entropy = calculate_entropy(proc_bytes)
            corr = calculate_correlation(proc_bytes, w, h)
            npcr, uaci = calculate_npcr_uaci(orig_bytes, proc_bytes)

            stats = {
                "entropy": entropy,
                "corr": corr,
                "npcr": npcr,
                "uaci": uaci
            }
        else:
            proc_bytes = decrypt_cbc_bytes(orig_bytes, sbox)
            stats = None

        res_img = Image.frombytes('L', (w, h), bytes(proc_bytes))
        buf = io.BytesIO()
        res_img.save(buf, 'PNG')
        buf.seek(0)

        return jsonify({
            "status": "success",
            "image_b64": base64.b64encode(buf.getvalue()).decode(),
            "stats": stats
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
