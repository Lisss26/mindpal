"""
MindPal v2.0 - Sistem Layanan Bimbingan Konseling Online Berbasis AI
SMP Negeri 16 Kota Pontianak

Backend : Flask + Groq API (LLaMA 3.3 70B)
Database: SQLite (WAL Mode)

FITUR v2.0:
  - 3 akun Guru BK, masing-masing mengampu 1 angkatan (VII, VIII, IX)
  - Setiap Guru BK hanya melihat data siswanya sendiri
  - 1 akun Super Admin (kepala / koordinator BK) melihat semua data
  - Import data siswa massal via file CSV
  - Password siswa auto-generate dari pola kelas (misal: mindpal9f)
  - Password di-hash (aman)
  - SQLite WAL mode (tahan concurrent users)
  - Rate limiting built-in (tanpa library tambahan)
  - Secret key stabil (disimpan di file .secret_key)
"""

from flask import (Flask, render_template, request,
                   redirect, url_for, session, jsonify, Response)
from werkzeug.security import generate_password_hash, check_password_hash
from groq import Groq
import sqlite3, os, csv, io, time, textwrap
from functools import wraps

# ─────────────────────────────────────────
# SECRET KEY STABIL
# ─────────────────────────────────────────
def load_or_create_secret_key():
    # Di production (Railway), gunakan SECRET_KEY dari environment variable
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key
    # Di lokal (development), simpan ke file
    path = ".secret_key"
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    import secrets
    key = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(key)
    print("🔑 Secret key baru dibuat → .secret_key")
    return key

app = Flask(__name__)
app.secret_key = load_or_create_secret_key()

# ─────────────────────────────────────────
# KONFIGURASI GROQ API
# ─────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"
client       = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────
# RATE LIMITER (sliding window, tanpa library)
# ─────────────────────────────────────────
_rate_store  = {}
RATE_LIMIT   = 15   # pesan per menit per user
RATE_WINDOW  = 60   # detik

def is_rate_limited(user_id):
    now     = time.time()
    key     = str(user_id)
    history = [t for t in _rate_store.get(key, []) if now - t < RATE_WINDOW]
    if len(history) >= RATE_LIMIT:
        _rate_store[key] = history
        return True
    history.append(now)
    _rate_store[key] = history
    return False

# ─────────────────────────────────────────
# SYSTEM PROMPT KONSELOR BK
# ─────────────────────────────────────────
SYSTEM_PROMPT = """
Kamu adalah MindPal, konselor bimbingan konseling (BK) virtual yang bertugas di SMP Negeri 16 Kota Pontianak.

IDENTITASMU:
- Nama: MindPal
- Peran: Konselor BK Virtual yang hangat, empatik, dan profesional
- Pendekatan: Person-Centered Therapy (Carl Rogers) — mendengarkan tanpa menghakimi

CARA BERKOMUNIKASI:
- Gunakan bahasa Indonesia yang ramah, santai namun tetap sopan
- Sesuaikan bahasa dengan usia siswa SMP (12-15 tahun) — tidak terlalu formal
- Selalu validasi perasaan siswa sebelum memberikan saran
- Ajukan pertanyaan terbuka untuk menggali lebih dalam
- Berikan respons yang hangat, tidak kaku seperti robot
- Gunakan kalimat pendek dan mudah dipahami remaja

FOKUS LAYANAN BK:
1. Masalah belajar & akademik (nilai, motivasi belajar, cara belajar)
2. Masalah pertemanan & hubungan sosial (konflik teman, bullying, pergaulan)
3. Masalah keluarga ringan (komunikasi dengan orang tua, adik/kakak)
4. Perencanaan masa depan & karir (cita-cita, minat bakat)
5. Manajemen emosi & stres (ujian, tekanan, kecemasan)
6. Kepercayaan diri & harga diri

BATASAN ETIS YANG WAJIB DIPATUHI:
- Jika siswa menunjukkan tanda krisis (ingin menyakiti diri sendiri, berbicara tentang
  kematian, situasi berbahaya), SEGERA sarankan menemui konselor manusia atau orang tua
- JANGAN pernah mendiagnosa gangguan mental apapun
- JANGAN berperan sebagai dokter atau psikolog klinis
- Selalu ingatkan bahwa konselor BK sekolah siap membantu secara langsung
- Jaga kerahasiaan dan privasi siswa

STRUKTUR RESPONS YANG BAIK:
1. Validasi perasaan ("Aku mengerti kamu merasa...")
2. Refleksi / pertanyaan menggali ("Boleh cerita lebih lanjut tentang...?")
3. Insight atau perspektif baru (jika sudah tepat waktunya)
4. Langkah konkret atau saran praktis (jika diminta)

PENTING: Kamu adalah jembatan pertama — bukan pengganti konselor manusia.
"""

# ─────────────────────────────────────────
# HELPER: generate password default siswa
# Format: mindpal + angkatan + huruf_kelas (lowercase)
# Contoh: kelas "IX F" → "mindpal9f"
# ─────────────────────────────────────────
ANGKATAN_MAP = {"VII": "7", "VIII": "8", "IX": "9"}

def generate_student_password(kelas: str) -> str:
    """
    kelas  : string seperti 'VII A', 'VIII F', 'IX I'
    return : plain-text password default, e.g. 'mindpal7a'
    """
    parts = kelas.strip().upper().split()
    if len(parts) >= 2:
        angkatan = ANGKATAN_MAP.get(parts[0], parts[0])
        huruf    = parts[1].lower()
    elif len(parts) == 1:
        angkatan = ANGKATAN_MAP.get(parts[0], parts[0])
        huruf    = ""
    else:
        angkatan, huruf = "0", "a"
    return f"mindpal{angkatan}{huruf}"

def angkatan_from_kelas(kelas: str) -> str:
    """Ekstrak angkatan Roman dari string kelas, e.g. 'VIII F' → 'VIII'"""
    return kelas.strip().upper().split()[0] if kelas else ""

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
DATABASE = os.environ.get("DATABASE", "mindpal.db")

def get_db():
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_db()
    c    = conn.cursor()

    # ── Tabel counselors (guru BK + super admin) ──
    c.execute("""
        CREATE TABLE IF NOT EXISTS counselors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            username   TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            angkatan   TEXT,           -- 'VII' | 'VIII' | 'IX' | NULL (super admin)
            role       TEXT    NOT NULL DEFAULT 'counselor'
                               CHECK(role IN ('counselor','superadmin'))
        )
    """)

    # ── Tabel siswa ───────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            nisn       TEXT    UNIQUE NOT NULL,
            kelas      TEXT    NOT NULL,
            angkatan   TEXT    NOT NULL,   -- 'VII' | 'VIII' | 'IX'
            password   TEXT    NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Tabel sesi chat ───────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            topic      TEXT    DEFAULT 'Sesi Konseling Baru',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id)
        )
    """)

    # ── Tabel pesan ───────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )
    """)

    # ── Tabel catatan konselor ────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS counselor_notes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id   INTEGER NOT NULL,
            counselor_id INTEGER NOT NULL,
            note         TEXT    NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id)   REFERENCES students(id),
            FOREIGN KEY (counselor_id) REFERENCES counselors(id)
        )
    """)

    # ── Tabel permintaan hubungi konselor (escalation) ───────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS escalations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            session_id INTEGER,
            is_read    INTEGER NOT NULL DEFAULT 0,   -- 0 = belum dibaca, 1 = sudah
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id),
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )
    """)

    # ── Seed: counselors ──────────────────────────
    c.execute("SELECT COUNT(*) FROM counselors")
    if c.fetchone()[0] == 0:
        seed_counselors = [
            # (name, username, password, angkatan, role)
            ("Super Admin BK",     "superadmin", "superadmin123", None,   "superadmin"),
            ("Guru BK Kelas VII",  "bk7",        "bkguru7123",   "VII",  "counselor"),
            ("Guru BK Kelas VIII", "bk8",        "bkguru8123",   "VIII", "counselor"),
            ("Guru BK Kelas IX",   "bk9",        "bkguru9123",   "IX",   "counselor"),
        ]
        for name, uname, pwd, angkatan, role in seed_counselors:
            c.execute(
                "INSERT INTO counselors (name,username,password,angkatan,role) VALUES (?,?,?,?,?)",
                (name, uname, generate_password_hash(pwd), angkatan, role)
            )

    # ── Seed: siswa contoh ────────────────────────
    c.execute("SELECT COUNT(*) FROM students")
    if c.fetchone()[0] == 0:
        samples = [
            ("Andi Pratama", "0011223344", "VIII A"),
            ("Sari Dewi",    "0055667788", "VII B"),
            ("Budi Santoso", "0099887766", "IX C"),
        ]
        for name, nisn, kelas in samples:
            plain = generate_student_password(kelas)
            c.execute(
                "INSERT INTO students (name,nisn,kelas,angkatan,password) VALUES (?,?,?,?,?)",
                (name, nisn, kelas, angkatan_from_kelas(kelas),
                 generate_password_hash(plain))
            )

    conn.commit()
    conn.close()
    print("✅ Database MindPal v2 berhasil diinisialisasi.")
    print("─" * 50)
    print("  Super Admin → superadmin / superadmin123")
    print("  Guru BK VII  → bk7 / bkguru7123")
    print("  Guru BK VIII → bk8 / bkguru8123")
    print("  Guru BK IX   → bk9 / bkguru9123")
    print("  Siswa (NISN sebagai username, password auto dari kelas)")
    print("─" * 50)

# ─────────────────────────────────────────
# DECORATORS
# ─────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def counselor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session or session.get("role") not in ("counselor", "superadmin"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def superadmin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "superadmin":
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────
# HELPER DB: filter siswa berdasarkan sesi login
# ─────────────────────────────────────────
def student_filter_clause():
    """
    Kembalikan (WHERE clause, params) untuk memfilter siswa
    sesuai angkatan yang diampu counselor yang sedang login.
    Super admin melihat semua.
    """
    if session.get("role") == "superadmin":
        return "", []
    return "WHERE s.angkatan = ?", [session.get("angkatan")]

# ─────────────────────────────────────────
# ROUTES: AUTH
# ─────────────────────────────────────────
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    role = session.get("role")
    if role in ("counselor", "superadmin"):
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("chat"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role_tab = request.form.get("role", "student")   # 'student' | 'counselor'
        conn     = get_db()

        if role_tab == "counselor":
            user = conn.execute(
                "SELECT * FROM counselors WHERE username=?", (username,)
            ).fetchone()
            if user and check_password_hash(user["password"], password):
                session.update({
                    "user_id"  : user["id"],
                    "user_name": user["name"],
                    "role"     : user["role"],        # 'counselor' | 'superadmin'
                    "angkatan" : user["angkatan"],    # 'VII'|'VIII'|'IX'|None
                })
                conn.close()
                return redirect(url_for("admin_dashboard"))
        else:
            # Siswa login menggunakan NISN sebagai username
            user = conn.execute(
                "SELECT * FROM students WHERE nisn=?", (username,)
            ).fetchone()
            if user and check_password_hash(user["password"], password):
                session.update({
                    "user_id"  : user["id"],
                    "user_name": user["name"],
                    "role"     : "student",
                    "kelas"    : user["kelas"],
                    "angkatan" : user["angkatan"],
                })
                conn.close()
                return redirect(url_for("chat"))

        conn.close()
        error = "Username / NISN atau password salah. Silakan coba lagi."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────
# ROUTES: STUDENT — CHAT
# ─────────────────────────────────────────
@app.route("/chat")
@login_required
def chat():
    if session.get("role") != "student":
        return redirect(url_for("admin_dashboard"))
    conn     = get_db()
    sessions = conn.execute("""
        SELECT cs.*, COUNT(m.id) as msg_count
        FROM chat_sessions cs
        LEFT JOIN messages m ON m.session_id = cs.id AND m.role = 'user'
        WHERE cs.student_id = ?
        GROUP BY cs.id
        ORDER BY cs.started_at DESC
    """, (session["user_id"],)).fetchall()
    conn.close()
    return render_template("chat.html",
                           user_name=session["user_name"],
                           kelas=session.get("kelas", ""),
                           sessions=sessions)

@app.route("/chat/new", methods=["POST"])
@login_required
def new_session():
    conn = get_db()
    c    = conn.cursor()
    c.execute("INSERT INTO chat_sessions (student_id) VALUES (?)",
              (session["user_id"],))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return jsonify({"session_id": sid})

@app.route("/chat/send", methods=["POST"])
@login_required
def send_message():
    if is_rate_limited(session["user_id"]):
        return jsonify({
            "error": f"Kamu mengirim pesan terlalu cepat. "
                     f"Tunggu sebentar ya 😊 (maks. {RATE_LIMIT} pesan/menit)"
        }), 429

    data       = request.get_json()
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id")

    if not user_msg or not session_id:
        return jsonify({"error": "Pesan atau sesi tidak valid."}), 400
    if len(user_msg) > 2000:
        return jsonify({"error": "Pesan terlalu panjang (maks. 2000 karakter)."}), 400

    conn = get_db()
    chat_sess = conn.execute(
        "SELECT * FROM chat_sessions WHERE id=? AND student_id=?",
        (session_id, session["user_id"])
    ).fetchone()
    if not chat_sess:
        conn.close()
        return jsonify({"error": "Sesi tidak ditemukan."}), 403

    conn.execute("INSERT INTO messages (session_id,role,content) VALUES (?,'user',?)",
                 (session_id, user_msg))
    conn.commit()

    history = conn.execute("""
        SELECT role, content FROM messages
        WHERE session_id=? ORDER BY timestamp ASC LIMIT 20
    """, (session_id,)).fetchall()

    try:
        resp     = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        *[{"role": r["role"], "content": r["content"]} for r in history]],
            max_tokens  = 1024,
            temperature = 0.7,
        )
        ai_reply = resp.choices[0].message.content
    except Exception as e:
        conn.close()
        print(f"❌ Groq Error: {e}")
        return jsonify({"error": "MindPal sedang gangguan. Coba lagi sebentar 🙏"}), 500

    conn.execute("INSERT INTO messages (session_id,role,content) VALUES (?,'assistant',?)",
                 (session_id, ai_reply))

    if chat_sess["topic"] == "Sesi Konseling Baru" and len(history) == 1:
        conn.execute("UPDATE chat_sessions SET topic=? WHERE id=?",
                     (user_msg[:50] + ("..." if len(user_msg) > 50 else ""), session_id))

    conn.commit()
    conn.close()
    return jsonify({"reply": ai_reply})

@app.route("/chat/history/<int:session_id>")
@login_required
def chat_history(session_id):
    conn = get_db()
    msgs = conn.execute("""
        SELECT role, content, timestamp FROM messages
        WHERE session_id=? ORDER BY timestamp ASC
    """, (session_id,)).fetchall()
    conn.close()
    return jsonify([dict(m) for m in msgs])


# ─────────────────────────────────────────
# ROUTE: SISWA — HUBUNGI KONSELOR MANUSIA (ESCALATION)
# ─────────────────────────────────────────
@app.route("/chat/escalate", methods=["POST"])
def chat_escalate():
    """Simpan permintaan siswa untuk dihubungi konselor manusia ke tabel escalations."""
    if session.get("role") != "student":
        return jsonify({"error": "Unauthorized"}), 403

    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    student_id = session["user_id"]

    conn = get_db()
    conn.execute(
        "INSERT INTO escalations (student_id, session_id) VALUES (?,?)",
        (student_id, session_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ─────────────────────────────────────────
# ROUTE: KONSELOR — TANDAI ESCALATION SUDAH DIBACA
# ─────────────────────────────────────────
@app.route("/admin/escalation/<int:escalation_id>/read", methods=["POST"])
@counselor_required
def mark_escalation_read(escalation_id):
    conn = get_db()
    conn.execute("UPDATE escalations SET is_read=1 WHERE id=?", (escalation_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ─────────────────────────────────────────
# ROUTES: COUNSELOR — DASHBOARD
# ─────────────────────────────────────────
@app.route("/admin")
@counselor_required
def admin_dashboard():
    conn    = get_db()
    role    = session.get("role")
    angkatan = session.get("angkatan")

    # Filter berdasarkan angkatan atau tampilkan semua (superadmin)
    if role == "superadmin":
        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
        total_messages = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role='user'"
        ).fetchone()[0]

        recent_sessions = conn.execute("""
            SELECT cs.id, cs.topic, cs.started_at,
                   s.id as student_id, s.name as student_name,
                   s.kelas, s.angkatan,
                   COUNT(m.id) as msg_count
            FROM chat_sessions cs
            JOIN students s ON s.id = cs.student_id
            LEFT JOIN messages m ON m.session_id=cs.id AND m.role='user'
            GROUP BY cs.id ORDER BY cs.started_at DESC LIMIT 15
        """).fetchall()

        students = conn.execute("""
            SELECT s.*, COUNT(cs.id) as session_count
            FROM students s
            LEFT JOIN chat_sessions cs ON cs.student_id=s.id
            GROUP BY s.id ORDER BY s.angkatan, s.kelas, s.name
        """).fetchall()

        # Statistik per angkatan untuk super admin
        stats_per_angkatan = conn.execute("""
            SELECT s.angkatan,
                   COUNT(DISTINCT s.id)  as jml_siswa,
                   COUNT(DISTINCT cs.id) as jml_sesi,
                   COUNT(m.id)           as jml_pesan
            FROM students s
            LEFT JOIN chat_sessions cs ON cs.student_id=s.id
            LEFT JOIN messages m ON m.session_id=cs.id AND m.role='user'
            GROUP BY s.angkatan ORDER BY s.angkatan
        """).fetchall()

    else:
        total_students = conn.execute(
            "SELECT COUNT(*) FROM students WHERE angkatan=?", (angkatan,)
        ).fetchone()[0]
        total_sessions = conn.execute("""
            SELECT COUNT(*) FROM chat_sessions cs
            JOIN students s ON s.id=cs.student_id
            WHERE s.angkatan=?
        """, (angkatan,)).fetchone()[0]
        total_messages = conn.execute("""
            SELECT COUNT(*) FROM messages m
            JOIN chat_sessions cs ON cs.id=m.session_id
            JOIN students s ON s.id=cs.student_id
            WHERE s.angkatan=? AND m.role='user'
        """, (angkatan,)).fetchone()[0]

        recent_sessions = conn.execute("""
            SELECT cs.id, cs.topic, cs.started_at,
                   s.id as student_id, s.name as student_name,
                   s.kelas, s.angkatan,
                   COUNT(m.id) as msg_count
            FROM chat_sessions cs
            JOIN students s ON s.id=cs.student_id
            LEFT JOIN messages m ON m.session_id=cs.id AND m.role='user'
            WHERE s.angkatan=?
            GROUP BY cs.id ORDER BY cs.started_at DESC LIMIT 15
        """, (angkatan,)).fetchall()

        students = conn.execute("""
            SELECT s.*, COUNT(cs.id) as session_count
            FROM students s
            LEFT JOIN chat_sessions cs ON cs.student_id=s.id
            WHERE s.angkatan=?
            GROUP BY s.id ORDER BY s.kelas, s.name
        """, (angkatan,)).fetchall()

        stats_per_angkatan = []

    # ── Escalations belum dibaca (semua role) ──────────────────────────
    if role == "superadmin":
        escalations = conn.execute("""
            SELECT e.id, e.created_at, e.is_read, e.session_id,
                   s.id as student_id, s.name as student_name, s.kelas, s.angkatan
            FROM escalations e
            JOIN students s ON s.id=e.student_id
            WHERE e.is_read=0
            ORDER BY e.created_at DESC
        """).fetchall()
    else:
        escalations = conn.execute("""
            SELECT e.id, e.created_at, e.is_read, e.session_id,
                   s.id as student_id, s.name as student_name, s.kelas, s.angkatan
            FROM escalations e
            JOIN students s ON s.id=e.student_id
            WHERE e.is_read=0 AND s.angkatan=?
            ORDER BY e.created_at DESC
        """, (angkatan,)).fetchall()

    conn.close()
    return render_template("admin.html",
                           admin_name   = session["user_name"],
                           role         = role,
                           angkatan     = angkatan,
                           total_students  = total_students,
                           total_sessions  = total_sessions,
                           total_messages  = total_messages,
                           recent_sessions = recent_sessions,
                           students        = students,
                           stats_per_angkatan = stats_per_angkatan,
                           escalations     = escalations)

# ─────────────────────────────────────────
# ROUTES: COUNSELOR — DETAIL SISWA
# ─────────────────────────────────────────
@app.route("/admin/student/<int:student_id>")
@counselor_required
def student_detail(student_id):
    conn    = get_db()
    student = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()

    if not student:
        conn.close()
        return "Siswa tidak ditemukan.", 404

    # Counselor biasa hanya boleh akses siswa angkatannya sendiri
    if session.get("role") == "counselor" and student["angkatan"] != session.get("angkatan"):
        conn.close()
        return "Akses ditolak.", 403

    sessions = conn.execute("""
        SELECT cs.*, COUNT(m.id) as msg_count
        FROM chat_sessions cs
        LEFT JOIN messages m ON m.session_id=cs.id AND m.role='user'
        WHERE cs.student_id=?
        GROUP BY cs.id ORDER BY cs.started_at DESC
    """, (student_id,)).fetchall()

    notes = conn.execute("""
        SELECT cn.*, c.name as counselor_name
        FROM counselor_notes cn
        JOIN counselors c ON c.id=cn.counselor_id
        WHERE cn.student_id=?
        ORDER BY cn.created_at DESC
    """, (student_id,)).fetchall()

    # Escalations siswa ini (semua, termasuk yang sudah dibaca)
    student_escalations = conn.execute("""
        SELECT id, created_at, is_read, session_id
        FROM escalations
        WHERE student_id=?
        ORDER BY created_at DESC
    """, (student_id,)).fetchall()

    conn.close()
    return render_template("student_detail.html",
                           student    = student,
                           sessions   = sessions,
                           notes      = notes,
                           admin_name = session["user_name"],
                           role       = session.get("role"),
                           student_escalations = student_escalations)

@app.route("/admin/student/<int:student_id>/session/<int:session_id>")
@counselor_required
def view_session(student_id, session_id):
    conn = get_db()
    msgs = conn.execute("""
        SELECT role, content, timestamp FROM messages
        WHERE session_id=? ORDER BY timestamp ASC
    """, (session_id,)).fetchall()
    conn.close()
    return jsonify([dict(m) for m in msgs])

@app.route("/admin/student/<int:student_id>/note", methods=["POST"])
@counselor_required
def add_note(student_id):
    data = request.get_json()
    note = data.get("note", "").strip()
    if not note:
        return jsonify({"error": "Catatan tidak boleh kosong."}), 400
    conn = get_db()
    conn.execute(
        "INSERT INTO counselor_notes (student_id,counselor_id,note) VALUES (?,?,?)",
        (student_id, session["user_id"], note)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ─────────────────────────────────────────
# ROUTES: TAMBAH SISWA MANUAL
# ─────────────────────────────────────────
@app.route("/admin/student/add", methods=["POST"])
@counselor_required
def add_student():
    data    = request.get_json()
    name    = data.get("name", "").strip()
    nisn    = data.get("nisn", "").strip()
    kelas   = data.get("kelas", "").strip()

    if not all([name, nisn, kelas]):
        return jsonify({"error": "Semua field wajib diisi."}), 400

    angkatan = angkatan_from_kelas(kelas)
    if not angkatan:
        return jsonify({"error": "Format kelas tidak valid. Contoh: VII A"}), 400

    # Counselor biasa hanya boleh tambah siswa angkatannya
    if session.get("role") == "counselor" and angkatan != session.get("angkatan"):
        return jsonify({"error": f"Kamu hanya bisa menambah siswa angkatan {session.get('angkatan')}."}), 403

    plain_pwd = generate_student_password(kelas)
    conn      = get_db()
    try:
        conn.execute(
            "INSERT INTO students (name,nisn,kelas,angkatan,password) VALUES (?,?,?,?,?)",
            (name, nisn, kelas, angkatan, generate_password_hash(plain_pwd))
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True, "password_info": plain_pwd})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "NISN sudah terdaftar."}), 409

# ─────────────────────────────────────────
# ROUTES: IMPORT CSV MASSAL
# ─────────────────────────────────────────
@app.route("/admin/import-csv", methods=["GET", "POST"])
@counselor_required
def import_csv():
    """
    GET  → halaman upload CSV
    POST → proses file CSV

    Format CSV yang diterima (header wajib ada):
        nama,nisn,kelas
        Budi Santoso,0123456789,IX F
        Sari Dewi,0987654321,VII B

    Password di-generate otomatis dari kelas:
        VII B  → mindpal7b
        VIII A → mindpal8a
        IX F   → mindpal9f
    """
    if request.method == "GET":
        return render_template("import_csv.html",
                               admin_name=session["user_name"],
                               role=session.get("role"),
                               angkatan=session.get("angkatan"))

    # ── POST: proses file ──────────────────
    if "file" not in request.files:
        return jsonify({"error": "File tidak ditemukan."}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Hanya file CSV yang diterima."}), 400

    content  = f.read().decode("utf-8-sig")   # utf-8-sig agar BOM Excel hilang

    # Auto-detect pemisah: titik koma (Excel Indonesia) atau koma (standar)
    delimiter = ";" if ";" in content.split("\n")[0] else ","
    reader    = csv.DictReader(io.StringIO(content), delimiter=delimiter)

    # Normalisasi header — abaikan spasi & beda huruf
    required = {"nama", "nisn", "kelas"}
    headers  = {h.strip().lower() for h in (reader.fieldnames or [])}
    if not required.issubset(headers):
        return jsonify({
            "error": f"Header CSV kurang. Butuh: nama, nisn, kelas. "
                     f"Ditemukan: {', '.join(reader.fieldnames or [])}"
        }), 400

    conn        = get_db()
    sukses      = 0
    gagal       = []
    duplikat    = []

    for i, row in enumerate(reader, start=2):   # baris 2 = data pertama
        try:
            name    = row.get("nama",  row.get("Nama",  "")).strip()
            nisn    = row.get("nisn",  row.get("NISN",  "")).strip()
            kelas   = row.get("kelas", row.get("Kelas", "")).strip()

            if not all([name, nisn, kelas]):
                gagal.append(f"Baris {i}: data tidak lengkap ({name}, {nisn}, {kelas})")
                continue

            angkatan  = angkatan_from_kelas(kelas)
            if not angkatan:
                gagal.append(f"Baris {i}: format kelas tidak valid → '{kelas}'")
                continue

            # Counselor biasa hanya bisa import angkatannya sendiri
            if session.get("role") == "counselor" and angkatan != session.get("angkatan"):
                gagal.append(f"Baris {i}: {name} ({kelas}) bukan angkatan {session.get('angkatan')}")
                continue

            plain_pwd = generate_student_password(kelas)
            conn.execute(
                "INSERT INTO students (name,nisn,kelas,angkatan,password) VALUES (?,?,?,?,?)",
                (name, nisn, kelas, angkatan, generate_password_hash(plain_pwd))
            )
            sukses += 1

        except sqlite3.IntegrityError:
            duplikat.append(f"Baris {i}: NISN {nisn} sudah ada")
        except Exception as ex:
            gagal.append(f"Baris {i}: error — {ex}")

    conn.commit()
    conn.close()

    return jsonify({
        "sukses"  : sukses,
        "duplikat": duplikat,
        "gagal"   : gagal,
        "pesan"   : f"{sukses} siswa berhasil diimport."
                    + (f" {len(duplikat)} NISN duplikat dilewati." if duplikat else "")
                    + (f" {len(gagal)} baris error." if gagal else "")
    })

# ─────────────────────────────────────────
# ROUTES: RESET PASSWORD SISWA
# ─────────────────────────────────────────
@app.route("/admin/student/<int:student_id>/reset-password", methods=["POST"])
@counselor_required
def reset_password(student_id):
    conn    = get_db()
    student = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    if not student:
        conn.close()
        return jsonify({"error": "Siswa tidak ditemukan."}), 404
    if session.get("role") == "counselor" and student["angkatan"] != session.get("angkatan"):
        conn.close()
        return jsonify({"error": "Akses ditolak."}), 403

    plain_pwd = generate_student_password(student["kelas"])
    conn.execute("UPDATE students SET password=? WHERE id=?",
                 (generate_password_hash(plain_pwd), student_id))
    conn.commit()
    conn.close()
    return jsonify({"success": True, "password_baru": plain_pwd})

# ─────────────────────────────────────────
# ROUTE: RINGKASAN AI PER SESI (manual trigger oleh guru BK)
# ─────────────────────────────────────────
@app.route("/admin/student/<int:student_id>/session/<int:session_id>/summarize", methods=["POST"])
@counselor_required
def summarize_session(student_id, session_id):
    """
    Generate ringkasan 1 paragraf dari percakapan satu sesi menggunakan Groq API.
    Hanya dipanggil saat guru BK menekan tombol 'Ringkasan' — tidak otomatis.
    """
    conn    = get_db()
    student = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    if not student:
        conn.close()
        return jsonify({"error": "Siswa tidak ditemukan."}), 404

    # Validasi akses angkatan
    if session.get("role") == "counselor" and student["angkatan"] != session.get("angkatan"):
        conn.close()
        return jsonify({"error": "Akses ditolak."}), 403

    msgs = conn.execute("""
        SELECT role, content FROM messages
        WHERE session_id=? ORDER BY timestamp ASC
    """, (session_id,)).fetchall()
    conn.close()

    if not msgs:
        return jsonify({"error": "Belum ada pesan dalam sesi ini."}), 400

    # Susun transkrip untuk dikirim ke AI
    transcript_lines = []
    for m in msgs:
        label = "Siswa" if m["role"] == "user" else "MindPal AI"
        transcript_lines.append(f"{label}: {m['content']}")
    transcript = "\n".join(transcript_lines)

    summary_prompt = f"""Berikut adalah transkrip percakapan konseling antara siswa SMP dan MindPal AI:

{transcript}

Buatkan SATU paragraf ringkasan singkat (3-5 kalimat) dalam Bahasa Indonesia yang:
- Menjelaskan topik utama yang dibahas siswa
- Mencatat kondisi emosional siswa secara umum
- Menyebutkan saran atau respons yang diberikan AI
- Ditulis dengan bahasa profesional dan objektif, cocok untuk dibaca guru BK
- JANGAN menyebut nama siswa, cukup gunakan kata "siswa"

Langsung tulis paragraf ringkasannya saja, tanpa judul atau label tambahan."""

    try:
        resp = client.chat.completions.create(
            model      = GROQ_MODEL,
            messages   = [{"role": "user", "content": summary_prompt}],
            max_tokens = 300,
            temperature= 0.4,
        )
        summary = resp.choices[0].message.content.strip()
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"❌ Groq summarize error: {e}")
        return jsonify({"error": "Gagal menghubungi AI. Coba lagi."}), 500


# ─────────────────────────────────────────
# ROUTE: DOWNLOAD LAPORAN PERCAKAPAN SEBAGAI PDF
# ─────────────────────────────────────────
@app.route("/admin/student/<int:student_id>/download-pdf")
@counselor_required
def download_student_pdf(student_id):
    """
    Generate dan download file PDF berisi seluruh riwayat percakapan siswa.
    PDF dibuat menggunakan HTML-to-text lalu dikemas sebagai file yang bisa
    dibuka di semua perangkat (HP maupun laptop/PC).
    Menggunakan ReportLab jika tersedia, fallback ke plain-text UTF-8 jika tidak.
    """
    conn    = get_db()
    student = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    if not student:
        conn.close()
        return "Siswa tidak ditemukan.", 404

    if session.get("role") == "counselor" and student["angkatan"] != session.get("angkatan"):
        conn.close()
        return "Akses ditolak.", 403

    sessions_data = conn.execute("""
        SELECT cs.id, cs.topic, cs.started_at, COUNT(m.id) as msg_count
        FROM chat_sessions cs
        LEFT JOIN messages m ON m.session_id=cs.id AND m.role='user'
        WHERE cs.student_id=?
        GROUP BY cs.id ORDER BY cs.started_at ASC
    """, (student_id,)).fetchall()

    all_sessions = []
    for s in sessions_data:
        msgs = conn.execute("""
            SELECT role, content, timestamp FROM messages
            WHERE session_id=? ORDER BY timestamp ASC
        """, (s["id"],)).fetchall()
        all_sessions.append({"session": s, "messages": msgs})

    notes = conn.execute("""
        SELECT cn.note, cn.created_at, c.name as counselor_name
        FROM counselor_notes cn
        JOIN counselors c ON c.id=cn.counselor_id
        WHERE cn.student_id=? ORDER BY cn.created_at DESC
    """, (student_id,)).fetchall()
    conn.close()

    student_name = student["name"]
    student_kelas = student["kelas"]
    student_nisn  = student["nisn"]

    # ── Coba generate PDF dengan ReportLab ──────────────────────────────
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        style_title   = ParagraphStyle("Title2",   parent=styles["Heading1"],  fontSize=16, textColor=colors.HexColor("#0D9488"), spaceAfter=4)
        style_h2      = ParagraphStyle("H2",       parent=styles["Heading2"],  fontSize=12, textColor=colors.HexColor("#1E293B"), spaceAfter=4)
        style_h3      = ParagraphStyle("H3",       parent=styles["Heading3"],  fontSize=10, textColor=colors.HexColor("#334155"), spaceAfter=2)
        style_normal  = ParagraphStyle("Normal2",  parent=styles["Normal"],    fontSize=9,  leading=14, spaceAfter=2)
        style_muted   = ParagraphStyle("Muted",    parent=styles["Normal"],    fontSize=8,  textColor=colors.HexColor("#64748B"), spaceAfter=2)
        style_bubble_user = ParagraphStyle("BubbleUser", parent=styles["Normal"], fontSize=9, leading=14,
                                           backColor=colors.HexColor("#0D9488"), textColor=colors.white,
                                           borderPadding=6, spaceAfter=4)
        style_bubble_ai   = ParagraphStyle("BubbleAI",   parent=styles["Normal"], fontSize=9, leading=14,
                                           backColor=colors.HexColor("#F1F5F9"), textColor=colors.HexColor("#0F172A"),
                                           borderPadding=6, spaceAfter=4)
        story = []

        # Header laporan
        story.append(Paragraph("Laporan Percakapan Konseling", style_title))
        story.append(Paragraph(f"MindPal AI · SMPN 16 Kota Pontianak", style_muted))
        story.append(Spacer(1, 8))

        # Info siswa
        data_tabel = [
            ["Nama Siswa", student_name],
            ["Kelas",      student_kelas],
            ["NISN",       student_nisn],
            ["Total Sesi", str(len(all_sessions))],
        ]
        tbl = Table(data_tabel, colWidths=[4*cm, 12*cm])
        tbl.setStyle(TableStyle([
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("TEXTCOLOR",   (0,0), (0,-1), colors.HexColor("#64748B")),
            ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#F8FAFC"), colors.white]),
            ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID",   (0,0), (-1,-1), 0.25, colors.HexColor("#E2E8F0")),
            ("TOPPADDING",  (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 14))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2DD4BF")))
        story.append(Spacer(1, 10))

        # Sesi percakapan
        for idx, item in enumerate(all_sessions, 1):
            s    = item["session"]
            msgs = item["messages"]
            topic    = s["topic"] or "Sesi Konseling"
            started  = str(s["started_at"])[:16].replace("T", " ")
            msg_count = s["msg_count"]

            story.append(Paragraph(f"Sesi {idx}: {topic}", style_h2))
            story.append(Paragraph(f"{started} · {msg_count} pesan siswa", style_muted))
            story.append(Spacer(1, 6))

            if not msgs:
                story.append(Paragraph("(Belum ada pesan)", style_muted))
            else:
                for m in msgs:
                    ts    = str(m["timestamp"] or "")[:16].replace("T", " ")
                    label = "Siswa" if m["role"] == "user" else "MindPal AI"
                    style = style_bubble_user if m["role"] == "user" else style_bubble_ai
                    # Escape karakter HTML di konten pesan
                    content_safe = (m["content"]
                                    .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                                    .replace("\n","<br/>"))
                    story.append(Paragraph(f"<b>{label}</b> [{ts}]<br/>{content_safe}", style))

            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
            story.append(Spacer(1, 8))

        # Catatan konselor
        if notes:
            story.append(Paragraph("Catatan Konselor", style_h2))
            story.append(Spacer(1, 4))
            for n in notes:
                ts   = str(n["created_at"])[:16].replace("T", " ")
                counselor = n["counselor_name"]
                note_safe = (n["note"]
                             .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                             .replace("\n","<br/>"))
                story.append(Paragraph(note_safe, style_normal))
                story.append(Paragraph(f"{ts} · {counselor}", style_muted))
                story.append(Spacer(1, 6))

        # Footer disclaimer
        story.append(Spacer(1, 12))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Dokumen ini bersifat rahasia dan hanya untuk keperluan bimbingan konseling. "
            "Digenerate otomatis oleh MindPal AI · SMPN 16 Kota Pontianak.",
            style_muted
        ))

        doc.build(story)
        buf.seek(0)

        filename = f"laporan_{student_name.replace(' ','_')}_{student_nisn}.pdf"
        return Response(
            buf.getvalue(),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except ImportError:
        # ── Fallback: plain-text UTF-8 jika ReportLab tidak ada ─────────
        lines = []
        lines.append("=" * 60)
        lines.append("LAPORAN PERCAKAPAN KONSELING - MINDPAL")
        lines.append("SMPN 16 Kota Pontianak")
        lines.append("=" * 60)
        lines.append(f"Nama Siswa : {student_name}")
        lines.append(f"Kelas      : {student_kelas}")
        lines.append(f"NISN       : {student_nisn}")
        lines.append(f"Total Sesi : {len(all_sessions)}")
        lines.append("")

        for idx, item in enumerate(all_sessions, 1):
            s    = item["session"]
            msgs = item["messages"]
            lines.append("-" * 60)
            lines.append(f"SESI {idx}: {s['topic'] or 'Sesi Konseling'}")
            lines.append(f"Tanggal : {str(s['started_at'])[:16].replace('T',' ')}")
            lines.append("-" * 60)
            if not msgs:
                lines.append("(Belum ada pesan)")
            else:
                for m in msgs:
                    ts    = str(m["timestamp"] or "")[:16].replace("T", " ")
                    label = "SISWA" if m["role"] == "user" else "MINDPAL AI"
                    lines.append(f"\n[{ts}] {label}:")
                    wrapped = textwrap.fill(m["content"], width=70, subsequent_indent="  ")
                    lines.append(wrapped)
            lines.append("")

        if notes:
            lines.append("=" * 60)
            lines.append("CATATAN KONSELOR")
            lines.append("=" * 60)
            for n in notes:
                ts = str(n["created_at"])[:16].replace("T", " ")
                lines.append(f"[{ts}] {n['counselor_name']}:")
                lines.append(textwrap.fill(n["note"], width=70, subsequent_indent="  "))
                lines.append("")

        lines.append("=" * 60)
        lines.append("Dokumen ini bersifat rahasia untuk keperluan bimbingan konseling.")
        lines.append("Digenerate oleh MindPal AI - SMPN 16 Kota Pontianak")

        content = "\n".join(lines)
        filename = f"laporan_{student_name.replace(' ','_')}_{student_nisn}.txt"
        return Response(
            content.encode("utf-8"),
            mimetype="text/plain; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print(f"\n🚀 MindPal v2 berjalan di http://localhost:5000")
    print(f"🛡️  Rate limit: {RATE_LIMIT} pesan/{RATE_WINDOW}detik per user\n")
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=5000)