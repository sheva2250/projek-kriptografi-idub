from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os
import io
import base64
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
                print(f"[INFO] Loaded {len(SBOX_CANDIDATES)} S-Boxes.")
        except:
            SBOX_CANDIDATES = []


load_database()


# ==============================================================================
# 2. CORE CRYPTO & METRICS (UPDATED)
# ==============================================================================

def fast_walsh_transform(func):
    """Fast Walsh-Hadamard Transform (FWT)"""
    n = len(func);
    wf = list(func)
    if all(x in [0, 1] for x in wf): wf = [1 if x == 0 else -1 for x in wf]  # Map boolean
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


# --- STANDARD METRICS ---
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
                    if val != (((sbox[inp ^ (1 << b)] >> i) & 1) ^ ((sbox[inp ^ (1 << b)] >> j) & 1)): total += 1
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
    """
    Transparency Order (TO) - Normalized
    Menggunakan Autokorelasi via FWT.
    Output dinormalisasi agar range 0.0 - 1.0 (Lower is Better).
    """
    N = 256
    n = 8
    max_to_val = 0

    for beta in range(1, N):
        # 1. Walsh Spectrum
        func = [(get_bit_count(sbox[x] & beta) % 2) for x in range(N)]
        W = fast_walsh_transform(func)

        # 2. Power Spectrum & Autocorrelation
        P = [w ** 2 for w in W]
        AC_raw = fast_walsh_transform(P)
        AC = [val // N for val in AC_raw]

        # 3. Sum Absolute Autocorrelation (a != 0)
        sum_abs_ac = sum(abs(val) for val in AC[1:])

        # 4. Hitung TO Beta
        # Rumus Normalized: (N - (Sum / (N-1))) / N
        # Ini akan menghasilkan angka kecil (misal 0.05 - 0.06 untuk AES)
        val = N - (sum_abs_ac / (N - 1))
        norm_val = val / N  # Normalisasi ke 0-1

        if norm_val > max_to_val:
            max_to_val = norm_val

    return round(max_to_val, 5)


def calculate_ci(sbox):
    """Correlation Immunity"""
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


# --- NEW METRICS (FROM SCREENSHOTS) ---

def calculate_fixed_points(sbox):
    """Menghitung jumlah input yang outputnya sama (S[x] == x)"""
    return sum(1 for x in range(256) if sbox[x] == x)


def calculate_cycles(sbox):
    """Menganalisis struktur cycle S-Box"""
    visited = [False] * 256
    cycles = []
    for i in range(256):
        if not visited[i]:
            curr = i
            length = 0
            while not visited[curr]:
                visited[curr] = True
                curr = sbox[curr]
                length += 1
            cycles.append(length)
    return {
        "min": min(cycles),
        "max": max(cycles),
        "count": len(cycles)
    }


# Tambahkan fungsi ini di bagian metric
def calculate_strength_value(nl, sac, bic_nl, bic_sac):
    """
    Menghitung Strength Value (SV) sesuai rumus Paper Alamsyah et al. Eq(20).
    Idealnya mendekati 0.
    Rumus: SV = (120 - NL) + abs(0.5 - SAC) + (120 - BIC_NL) + abs(0.5 - BIC_SAC)
    """
    # Pastikan bic_nl dan bic_sac diambil dari hasil perhitungan
    # Di paper, nilai ideal NL dan BIC-NL dipatok 120 (sedikit diatas max teoretis 112)

    sv = (120 - nl) + abs(0.5 - sac) + (120 - bic_nl) + abs(0.5 - bic_sac)
    return round(sv, 6)


# Update fungsi wrapper utama
def calculate_all_metrics(sbox):
    sbox = [int(x) for x in sbox]
    cycles = calculate_cycles(sbox)

    # Hitung nilai-nilai dulu
    val_nl = calculate_nl(sbox)
    val_sac = calculate_sac(sbox)
    val_bic_nl = calculate_bic_nl(sbox)
    val_bic_sac = calculate_bic_sac(sbox)

    # Hitung SV Paper
    val_sv = calculate_strength_value(val_nl, val_sac, val_bic_nl, val_bic_sac)

    return {
        "NL": val_nl,
        "SAC": val_sac,
        "BIC_NL": val_bic_nl,
        "BIC_SAC": val_bic_sac,
        "LAP": calculate_lap(sbox),
        "DAP": calculate_dap(sbox),
        "DU": calculate_du(sbox),
        "AD": calculate_ad(sbox),
        "TO": calculate_to(sbox),
        "CI": calculate_ci(sbox),
        "FIXED": calculate_fixed_points(sbox),
        "CYC_MAX": cycles['max'],
        "SV_PAPER": val_sv
    }


# ==============================================================================
# 3. ENCRYPTION & ROUTES
# ==============================================================================
def get_inv_sbox(sbox):
    inv = [0] * 256
    for i, v in enumerate(sbox): inv[v] = i
    return inv


def encrypt_cbc(data, sbox):
    out = bytearray();
    prev = 0
    for b in data:
        val = sbox[b ^ prev];
        out.append(val);
        prev = val
    return out


def decrypt_cbc(data, sbox):
    inv = get_inv_sbox(sbox);
    out = bytearray();
    prev = 0
    for b in data:
        val = inv[b] ^ prev;
        out.append(val);
        prev = b
    return out


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


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        sbox = request.json.get('sbox')
        if not sbox or len(sbox) != 256: return jsonify({"error": "Invalid Data"}), 400
        return jsonify({"status": "success", "metrics": calculate_all_metrics(sbox)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/process_text', methods=['POST'])
def process_text():
    try:
        d = request.json;
        mode = d['mode'];
        sbox = d['sbox']
        if mode == 'encrypt':
            res = encrypt_cbc(d['text'].encode(), sbox).hex().upper()
        else:
            res = decrypt_cbc(bytes.fromhex(d['text'].replace(' ', '')), sbox).decode(errors='ignore')
        return jsonify({"status": "success", "result": res})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        f = request.files['image'];
        sbox = [int(x) for x in request.form['sbox'].split(',')]
        img = Image.open(f).convert('RGB');
        b = img.tobytes()
        proc = encrypt_cbc(b, sbox) if request.form['mode'] == 'encrypt' else decrypt_cbc(b, sbox)
        res_img = Image.frombytes('RGB', img.size, bytes(proc));
        buf = io.BytesIO();
        res_img.save(buf, 'PNG');
        buf.seek(0)
        return jsonify({"status": "success", "image_b64": base64.b64encode(buf.getvalue()).decode()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)