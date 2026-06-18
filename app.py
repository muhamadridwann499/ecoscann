"""
app.py - EcoScan: Sistem Klasifikasi Sampah Berbasis CNN
Tugas Akhir 2025
"""

import os, uuid, json, time, base64
from datetime import datetime
from functools import wraps

import numpy as np
from PIL import Image
import mysql.connector
from flask import (Flask, render_template, request, jsonify, redirect,
                   url_for, session, flash, send_from_directory)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

try:
    import bcrypt
except ImportError:
    bcrypt = None

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR    = os.path.join(BASE_DIR, 'static', 'uploads')
UMKM_DIR      = os.path.join(BASE_DIR, 'static', 'umkm_uploads')
MODEL_DIR     = os.path.join(BASE_DIR, 'model')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(UMKM_DIR,   exist_ok=True)

ALLOWED_EXT   = {'jpg','jpeg','png','webp','gif','bmp'}
IMG_SIZE      = (128, 128)
CLASSES       = ['cardboard','glass','metal','paper','plastic','trash']

CLASS_INFO = {
    'cardboard': {'label':'Kardus',      'icon':'📦','color':'#F4A460','recyclable':True},
    'glass':     {'label':'Kaca',        'icon':'🫙','color':'#87CEEB','recyclable':True},
    'metal':     {'label':'Logam',       'icon':'🥫','color':'#C0C0C0','recyclable':True},
    'paper':     {'label':'Kertas',      'icon':'📄','color':'#DEB887','recyclable':True},
    'plastic':   {'label':'Plastik',     'icon':'♻️','color':'#FF6B6B','recyclable':True},
    'trash':     {'label':'Sampah Umum', 'icon':'🗑️','color':'#808080','recyclable':False},
}

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'ecoscan_secret_2025_change_in_production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    'host':'localhost','port':3306,'user':'root','password':'',
    'database':'waste_classifier','charset':'utf8mb4','autocommit':True,
}

def get_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        app.logger.error(f"DB Error: {e}")
        return None

def db_query(sql, params=None, fetchall=False, fetchone=False):
    conn = get_db()
    if not conn: return None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params or ())
        if fetchall:  return cur.fetchall()
        if fetchone:  return cur.fetchone()
        conn.commit(); return cur.lastrowid
    except Exception as e:
        app.logger.error(f"Query error: {e}"); return None
    finally:
        try: cur.close(); conn.close()
        except: pass

def init_database():
    sql_path = os.path.join(BASE_DIR, 'database', 'waste_db.sql')
    if not os.path.exists(sql_path): return
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'],
            password=DB_CONFIG['password'], port=DB_CONFIG['port'], autocommit=True)
        cur = conn.cursor()
        with open(sql_path, encoding='utf-8') as f:
            raw = f.read()
        # strip delimiter blocks
        raw = raw.replace('DELIMITER //','').replace('DELIMITER ;','')
        stmts, buf, in_proc = [], [], False
        for line in raw.split('\n'):
            ls = line.strip()
            if ls.startswith('--'): continue
            if 'CREATE PROCEDURE' in ls: in_proc = True
            if in_proc:
                if 'END //' in ls or ls == 'END;': in_proc = False
                continue
            buf.append(line)
            if ';' in line and not in_proc:
                s = '\n'.join(buf).strip()
                if s: stmts.append(s)
                buf = []
        for s in stmts:
            if not s or s.startswith('--'): continue
            try: cur.execute(s)
            except mysql.connector.Error as e:
                if 'already' not in str(e).lower() and 'duplicate' not in str(e).lower():
                    app.logger.debug(f"SQL: {e}")
        cur.close(); conn.close()
        app.logger.info("✅ Database init OK")
    except Exception as e:
        app.logger.error(f"DB init error: {e}")

def ensure_users_schema():
    conn = get_db()
    if not conn:
        app.logger.error("Users schema check error: unable to connect to database")
        return
    cur = None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SHOW COLUMNS FROM users")
        existing = {row['Field'] for row in cur.fetchall()}

        column_additions = []
        if 'email' not in existing:
            column_additions.append("ADD COLUMN email VARCHAR(100) NULL AFTER password_hash")
        if 'full_name' not in existing:
            column_additions.append("ADD COLUMN full_name VARCHAR(100) NULL AFTER email")
        if 'role' not in existing:
            column_additions.append("ADD COLUMN role ENUM('admin','user') DEFAULT 'user' AFTER full_name")
        if 'is_active' not in existing:
            column_additions.append("ADD COLUMN is_active BOOLEAN DEFAULT TRUE AFTER role")
        if 'created_at' not in existing:
            column_additions.append("ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP AFTER is_active")
        if 'last_login' not in existing:
            column_additions.append("ADD COLUMN last_login TIMESTAMP NULL AFTER created_at")

        for clause in column_additions:
            cur.execute(f"ALTER TABLE users {clause}")

        cur.execute("UPDATE users SET is_active=1 WHERE is_active IS NULL OR is_active<>1")
        conn.commit()
        app.logger.info("✅ Users schema verified")
    except mysql.connector.Error as e:
        app.logger.error(f"Users schema error: {e}")
    except Exception as e:
        app.logger.error(f"Users schema check error: {e}")
    finally:
        try:
            if cur:
                cur.close()
            conn.close()
        except Exception:
            pass

def activate_all_users():
    conn = get_db()
    if not conn:
        app.logger.error("User activation error: unable to connect to database")
        return
    try:
        cur = conn.cursor()
        try:
            cur.execute("UPDATE users SET is_active=1")
        except mysql.connector.Error as e:
            if getattr(e, 'errno', None) == 1054:
                cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE")
                cur.execute("UPDATE users SET is_active=1")
            else:
                raise
        conn.commit()
        app.logger.info("✅ All users activated")
    except Exception as e:
        app.logger.error(f"User activation error: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

# ── Model CNN ─────────────────────────────────────────────────────────────────
_model = _labels = None

def load_model():
    global _model, _labels
    mp = os.path.join(MODEL_DIR,'waste_classifier.keras')
    lp = os.path.join(MODEL_DIR,'class_labels.json')
    if os.path.exists(mp):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(mp)
        except Exception as e:
            app.logger.error(f"Model load error: {e}"); _model = None
    _labels = ({v:k for k,v in json.load(open(lp)).items()} if os.path.exists(lp)
               else {i:c for i,c in enumerate(CLASSES)})

def predict_image(path):
    if _model is None: return _demo_predict()
    img = Image.open(path).convert('RGB').resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img,'float32')/255.0, 0)
    t0 = time.time(); preds = _model.predict(arr,verbose=0)[0]; elapsed = time.time()-t0
    idx = int(np.argmax(preds))
    return {'predicted_class':_labels.get(idx,CLASSES[idx]),'confidence':float(preds[idx]),
            'probabilities':{_labels.get(i,CLASSES[i]):float(v) for i,v in enumerate(preds)},
            'processing_time':round(elapsed,3),'demo_mode':False}

def _demo_predict():
    import random
    p = [random.uniform(.01,1) for _ in CLASSES]; t=sum(p); p=[v/t for v in p]
    i=p.index(max(p))
    return {'predicted_class':CLASSES[i],'confidence':p[i],
            'probabilities':dict(zip(CLASSES,p)),'processing_time':0.0,'demo_mode':True}

# ── Auth helpers ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def deco(*a,**kw):
        if 'user_id' not in session: return redirect(url_for('login'))
        return f(*a,**kw)
    return deco

def admin_required(f):
    @wraps(f)
    def deco(*a,**kw):
        if 'user_id' not in session: return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Akses ditolak. Hanya admin yang dapat mengakses halaman ini.','error')
            return redirect(url_for('index'))
        return f(*a,**kw)
    return deco

def allowed_file(fn): return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXT

def verify_password_hash(password, stored_hash):
    if not stored_hash:
        return False
    try:
        if stored_hash.startswith(('$2a$', '$2b$', '$2y$')) and bcrypt is not None:
            return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        return check_password_hash(stored_hash, password)
    except Exception:
        return False

def save_upload(file, dest=None):
    dest = dest or UPLOAD_DIR
    orig = secure_filename(file.filename)
    ext  = orig.rsplit('.',1)[-1].lower() if '.' in orig else 'jpg'
    name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(dest, name)
    file.save(path)
    return name, orig, path

# ── UMKM Data ─────────────────────────────────────────────────────────────────
PLATFORM_URLS = {
    'Shopee':               'https://shopee.co.id',
    'Tokopedia':            'https://tokopedia.com',
    'TikTok Shop':          'https://shop.tiktok.com',
    'Instagram':            'https://instagram.com',
    'Facebook Marketplace': 'https://facebook.com/marketplace',
    'WhatsApp Business':    'https://business.whatsapp.com',
    'Pinterest':            'https://pinterest.com',
    'Etsy':                 'https://etsy.com',
    'Bukalapak':            'https://bukalapak.com',
    'Lazada':               'https://lazada.co.id',
    'Kaskus FJB':           'https://kaskus.co.id/forum/45',
    'Facebook Group':       'https://facebook.com/groups',
    'Discord':              'https://discord.com',
    'YouTube':              'https://youtube.com',
    'Pasar kreatif':        'https://www.google.com/search?q=pasar+kreatif+terdekat',
    'Grab/Gojek':           'https://gofood.co.id',
}

UMKM_LIST = [
    # ── Kardus ──
    {'icon':'🪑','name':'Furnitur Kardus','waste_icon':'📦','waste_label':'Kardus',
     'desc':'Kursi, rak buku, dan meja dari kardus laminasi berlapis. Kuat, ringan, estetik.','profit':'Rp 3–8 jt'},
    {'icon':'🪴','name':'Pot & Organizer Kardus','waste_icon':'📦','waste_label':'Kardus',
     'desc':'Pot tanaman dan organizer meja dari kardus gulung. Laku di marketplace.','profit':'Rp 1–3 jt'},
    {'icon':'🧩','name':'Puzzle & Mainan Edukatif','waste_icon':'📦','waste_label':'Kardus',
     'desc':'Puzzle dan mainan anak dari kardus tebal, dicat dan dilaminasi tahan lama.','profit':'Rp 2–5 jt'},
    {'icon':'🖼️','name':'Bingkai Foto Kardus','waste_icon':'📦','waste_label':'Kardus',
     'desc':'Bingkai foto artistik dan dekorasi dinding dari lapisan kardus bertekstur.','profit':'Rp 2–4 jt'},
    # ── Kaca ──
    {'icon':'💡','name':'Lampu Hias Botol Kaca','waste_icon':'🫙','waste_label':'Kaca',
     'desc':'Botol kaca bekas jadi lampu gantung artistik untuk kafe dan rumah.','profit':'Rp 4–10 jt'},
    {'icon':'🏺','name':'Vas Bunga & Dekorasi','waste_icon':'🫙','waste_label':'Kaca',
     'desc':'Botol kaca dihias cat dan tali menjadi vas bunga premium artisan.','profit':'Rp 2–5 jt'},
    {'icon':'🍯','name':'Jual Toples Artisan','waste_icon':'🫙','waste_label':'Kaca',
     'desc':'Toples kaca bersih dijual ke produsen selai, madu, dan pickle artisan.','profit':'Rp 1–3 jt'},
    {'icon':'🎨','name':'Mosaik Seni Kaca','waste_icon':'🫙','waste_label':'Kaca',
     'desc':'Pecahan kaca berwarna dibuat mosaik seni dinding dan aksesori dekoratif.','profit':'Rp 2–6 jt'},
    # ── Logam ──
    {'icon':'🪔','name':'Kerajinan Kaleng Artistik','waste_icon':'🥫','waste_label':'Logam',
     'desc':'Kaleng bekas jadi tempat pensil, pot bunga, lampu taman berlubang.','profit':'Rp 2–5 jt'},
    {'icon':'💍','name':'Aksesori & Suvenir Logam','waste_icon':'🥫','waste_label':'Logam',
     'desc':'Pengecoran aluminium bekas jadi gelang, cincin, dan suvenir kustom.','profit':'Rp 3–8 jt'},
    {'icon':'🗿','name':'Miniatur & Patung Logam','waste_icon':'🥫','waste_label':'Logam',
     'desc':'Patung dan miniatur dari kaleng dirangkai dan dicat jadi karya bernilai seni.','profit':'Rp 2–7 jt'},
    {'icon':'🏠','name':'Panel Dekorasi Bangunan','waste_icon':'🥫','waste_label':'Logam',
     'desc':'Kaleng dipress jadi panel dekoratif untuk atap gazebo dan dinding taman.','profit':'Rp 3–6 jt'},
    # ── Kertas ──
    {'icon':'🎭','name':'Paper Mache & Kerajinan','waste_icon':'📄','waste_label':'Kertas',
     'desc':'Bubur kertas dibentuk jadi topeng, patung, dekorasi bernilai seni tinggi.','profit':'Rp 2–5 jt'},
    {'icon':'📓','name':'Buku Catatan Handmade','waste_icon':'📄','waste_label':'Kertas',
     'desc':'Kertas daur ulang dijilid jadi buku catatan premium dengan cover artistik.','profit':'Rp 3–7 jt'},
    {'icon':'🎁','name':'Wrapping Paper Custom','waste_icon':'📄','waste_label':'Kertas',
     'desc':'Kertas koran dicetak ulang dengan motif unik untuk pembungkus kado artisan.','profit':'Rp 1–3 jt'},
    {'icon':'🌱','name':'Pot Semai dari Koran','waste_icon':'📄','waste_label':'Kertas',
     'desc':'Pot semai biodegradable dari koran gulung — diminati hobi tanam dan petani.','profit':'Rp 1–4 jt'},
    # ── Plastik ──
    {'icon':'🧱','name':'Ecobrick','waste_icon':'♻️','waste_label':'Plastik',
     'desc':'Botol PET diisi plastik lunak jadi bata konstruksi furnitur dan taman.','profit':'Rp 500rb–2 jt'},
    {'icon':'🖨️','name':'Filamen 3D Printer','waste_icon':'♻️','waste_label':'Plastik',
     'desc':'Botol PET diolah jadi filamen printer 3D. Nilai sangat tinggi di komunitas maker.','profit':'Rp 5–15 jt'},
    {'icon':'👜','name':'Tas & Aksesori Upcycle','waste_icon':'♻️','waste_label':'Plastik',
     'desc':'Kemasan plastik berwarna dianyam jadi tas, dompet, aksesori eco fashion.','profit':'Rp 3–8 jt'},
    {'icon':'🏗️','name':'Paving Block Plastik','waste_icon':'♻️','waste_label':'Plastik',
     'desc':'Plastik dicampur pasir dicetak jadi paving block ringan dan tahan air.','profit':'Rp 5–12 jt'},
    # ── Trash / Sampah Umum ──
    {'icon':'🌿','name':'Kompos & Pupuk Organik','waste_icon':'🗑️','waste_label':'Sampah Umum',
     'desc':'Sisa makanan dan organik diolah jadi kompos. Jual ke petani, komunitas urban farming.','profit':'Rp 500rb–2 jt'},
    {'icon':'⚡','name':'Biogas dari Sampah Organik','waste_icon':'🗑️','waste_label':'Sampah Umum',
     'desc':'Sampah organik difermentasi menghasilkan gas bakar untuk kebutuhan rumah tangga.','profit':'Rp 1–3 jt'},
    {'icon':'🎪','name':'Seni Instalasi Sampah','waste_icon':'🗑️','waste_label':'Sampah Umum',
     'desc':'Sampah non-daur ulang dijadikan instalasi seni edukatif. Diminati sekolah dan event.','profit':'Rp 2–8 jt'},
]

UMKM_GUIDES = {
    'Furnitur Kardus': {
        'icon':'🪑','waste':'Kardus','desc':'Membuat furnitur dari kardus laminasi: kursi, rak buku, meja ringan.',
        'modal':'Rp 300rb–2 jt','profit':'Rp 3–8 jt/bln','waktu':'1–2 bln',
        'steps':[
            {'title':'Kumpulkan kardus tebal','desc':'Minta gratis ke supermarket, toko elektronik, atau ekspedisi. Target 50–100 lembar tebal.'},
            {'title':'Beli alat dasar','desc':'Cutter besar (Rp 50rb), lem UHU (Rp 100rb), penggaris besi, amplas. Total ±Rp 300rb.'},
            {'title':'Teknik laminasi','desc':'Susun lapisan kardus bersilang-seling (serat tegak lurus) dan lem tiap lapisan.'},
            {'title':'Buat produk pertama','desc':'Mulai dari yang sederhana: tempat pensil, rak kecil, bingkai foto.'},
            {'title':'Finishing anti air','desc':'Lapisi cat akrilik + vernish. Produk jadi tahan air dan terlihat premium.'},
            {'title':'Foto & jual online','desc':'Upload ke Shopee/Tokopedia. Harga kursi kardus Rp 150rb–400rb tergantung ukuran.'},
        ],
        'checklist':['Kardus bekas minimal 50 lembar tebal','Cutter besar dan penggaris besi','Lem putih/UHU kering kuat','Cat akrilik dan kuas','Vernish transparan','Amplas halus (220 grit)'],
        'platforms':['Shopee','Tokopedia','Instagram','TikTok Shop','Facebook Marketplace'],
        'tips':['Kardus gelombang 3 lapis lebih kuat dari kardus mie biasa','Video proses di TikTok bisa viral dan mendatangkan banyak pesanan','Terima pesanan custom dengan harga premium +30%'],
    },
    'Pot & Organizer Kardus': {
        'icon':'🪴','waste':'Kardus','desc':'Pot tanaman dan organizer meja dari kardus gulung.',
        'modal':'Rp 100rb–500rb','profit':'Rp 1–3 jt/bln','waktu':'1–2 minggu',
        'steps':[
            {'title':'Pilih kardus gulungan','desc':'Kardus bekas toilet, paper towel, atau roll kain jadi bahan utama pot dan organizer.'},
            {'title':'Desain dan potong','desc':'Potong gulungan sesuai ukuran. Susun vertikal dan rekatkan jadi organizer.'},
            {'title':'Lapisi dalam pot','desc':'Lapisi bagian dalam pot dengan plastik tipis agar tahan air untuk tanaman.'},
            {'title':'Cat dan dekorasi','desc':'Cat dengan warna menarik, tambahkan decoupage atau washi tape untuk estetik.'},
            {'title':'Foto produk jadi','desc':'Foto dengan tanaman kecil atau alat tulis di dalamnya untuk konten yang menarik.'},
        ],
        'checklist':['Kardus gulungan/tabung bekas','Lem tembak atau UHU','Plastik tipis untuk lapisan pot','Cat akrilik warna-warni','Kuas dan spons'],
        'platforms':['Shopee','Tokopedia','TikTok Shop','Instagram'],
        'tips':['Set organizer isi 5–10 slot lebih laku dari satuan','Pot ukuran kecil cocok untuk sukulen — target pasar hobi tanaman hias','Foto flat lay dengan latar kayu atau kain linen meningkatkan daya tarik'],
    },
    'Puzzle & Mainan Edukatif': {
        'icon':'🧩','waste':'Kardus','desc':'Puzzle dan mainan anak dari kardus tebal yang dicat dan dilaminasi.',
        'modal':'Rp 200rb–800rb','profit':'Rp 2–5 jt/bln','waktu':'2–3 minggu',
        'steps':[
            {'title':'Pilih kardus tebal','desc':'Kardus gelombang 5 lapis atau kardus hardcover buku lama — paling kuat untuk mainan.'},
            {'title':'Desain dan cetak gambar','desc':'Desain gambar di Canva, cetak di kertas photo, tempel di kardus, laminasi.'},
            {'title':'Potong puzzle','desc':'Gunakan cutter tajam dan cetakan puzzle atau potong bebas bentuk animal, buah, dll.'},
            {'title':'Amplas tepi','desc':'Amplas semua tepi agar tidak tajam dan aman untuk anak-anak.'},
            {'title':'Kemasan menarik','desc':'Masukkan dalam kotak atau kantong kain. Tambahkan label nama brand.'},
        ],
        'checklist':['Kardus tebal 5 lapis / hardcover','Printer atau kertas foto untuk gambar','Laminator / plastik laminasi','Cutter tajam dan cutting mat','Amplas halus','Kotak kemasan atau kantong kain'],
        'platforms':['Shopee','Tokopedia','Instagram','TikTok Shop','Facebook Marketplace'],
        'tips':['Puzzle bertema edukasi (huruf, angka, hewan) paling diminati orang tua','Custom nama anak di puzzle jadi produk gift premium','Sertifikat mainan aman dapat meningkatkan kepercayaan pembeli'],
    },
    'Bingkai Foto Kardus': {
        'icon':'🖼️','waste':'Kardus','desc':'Bingkai foto artistik dan dekorasi dinding dari lapisan kardus.',
        'modal':'Rp 100rb–400rb','profit':'Rp 2–4 jt/bln','waktu':'1–2 minggu',
        'steps':[
            {'title':'Buat template bingkai','desc':'Gambar desain bingkai di kertas, potong, jadikan template untuk memotong kardus.'},
            {'title':'Laminasi berlapis','desc':'Tempel 4–6 lapisan kardus dengan lem, tekan rata dan tunggu kering 24 jam.'},
            {'title':'Potong dan bentuk','desc':'Potong sesuai template. Buat lubang tengah sesuai ukuran foto (4R, A4, dll).'},
            {'title':'Finishing dan dekorasi','desc':'Cat, tempel potongan majalah/washi tape, atau buat tekstur emboss dengan lem tembak.'},
            {'title':'Pasang gantungan','desc':'Tempel gantungan bingkai di belakang, atau tambah kaki penyangga untuk berdiri.'},
        ],
        'checklist':['Kardus bekas berlapis','Lem putih dan lem tembak','Cat akrilik berbagai warna','Washi tape atau kertas dekorasi','Gantungan bingkai (beli di toko craft)','Amplas halus'],
        'platforms':['Shopee','Tokopedia','Instagram','Pinterest','Etsy'],
        'tips':['Bingkai dengan tulisan motivasi atau quote laku keras di marketplace','Bingkai set 3 ukuran (S, M, L) lebih menarik dari satuan','Target pasar: dekorasi kamar anak, kado wisuda, kado ulang tahun'],
    },
    'Lampu Hias Botol Kaca': {
        'icon':'💡','waste':'Kaca','desc':'Botol kaca bekas diubah jadi lampu gantung artistik untuk kafe dan rumah.',
        'modal':'Rp 300rb–1,5 jt','profit':'Rp 4–10 jt/bln','waktu':'2–3 minggu',
        'steps':[
            {'title':'Kumpulkan botol kaca','desc':'Botol bir, wine, sirup, kecap berbagai ukuran. Minta ke kafe/restoran.'},
            {'title':'Cuci dan sterilkan','desc':'Rendam air sabun panas, bilas bersih. Lepas label dengan minyak sayur.'},
            {'title':'Beli komponen listrik','desc':'Kabel gantung (Rp 15rb/m), fitting E27 (Rp 8rb), bohlam LED Edison (Rp 20–35rb).'},
            {'title':'Rakit dan uji keamanan','desc':'Kabel masuk dari mulut botol, botol dibalik sebagai shade. Uji nyala dulu.'},
            {'title':'Foto dramatis','desc':'Foto di ruang gelap agar efek cahaya terlihat menawan.'},
            {'title':'Pasarkan ke kafe','desc':'Tawarkan langsung ke kafe dengan membawa sample. Paket 3 lampu lebih laku.'},
        ],
        'checklist':['Botol kaca bersih min 10 buah','Kabel gantung + fitting E27','Bohlam LED Edison warm white','Tang dan obeng','Cat semprot (opsional)','HP untuk foto produk'],
        'platforms':['Instagram','Shopee','Tokopedia','WhatsApp Business','Pinterest'],
        'tips':['Kafe dan warung kopi adalah pembeli utama — tawarkan langsung dengan sample','Tali goni atau cat chalkboard di botol menaikkan harga jual 50%','Dokumentasikan proses — konten ini sangat viral di Reels/TikTok'],
    },
    'Vas Bunga & Dekorasi': {
        'icon':'🏺','waste':'Kaca','desc':'Botol kaca dihias cat dan tali jadi vas bunga premium artisan.',
        'modal':'Rp 100rb–500rb','profit':'Rp 2–5 jt/bln','waktu':'1–2 minggu',
        'steps':[
            {'title':'Pilih botol estetik','desc':'Botol wine, botol sirup berbentuk unik, toples selai — semakin unik semakin mahal.'},
            {'title':'Bersihkan total','desc':'Rendam dalam air sabun, gosok bagian dalam dengan sikat botol, keringkan sempurna.'},
            {'title':'Teknik dekorasi','desc':'Pilih metode: cat chalk paint, lilit tali rami, decoupage kertas, atau cat semprot.'},
            {'title':'Buat set koleksi','desc':'Buat set 3 botol tinggi berbeda untuk satu tema warna (monokrom, earthy tone, dll).'},
            {'title':'Foto dengan bunga','desc':'Isi dengan bunga kering atau artificial untuk foto produk yang menarik.'},
        ],
        'checklist':['Botol kaca bersih berbagai ukuran','Cat chalk/akrilik','Tali rami/jute','Lem tembak','Bunga kering untuk foto','Latar foto (kayu, kain linen)'],
        'platforms':['Instagram','Shopee','Tokopedia','Pinterest','Etsy'],
        'tips':['Set vas "trio" (3 botol berbeda tinggi) adalah produk terlaris','Tema warna musiman (natal, lebaran) meningkatkan penjualan 3x','Bunga kering (dried flower) dalam vas bisa dijual sebagai paket'],
    },
    'Jual Toples Artisan': {
        'icon':'🍯','waste':'Kaca','desc':'Toples kaca bersih dijual ke produsen selai, madu, dan pickle artisan.',
        'modal':'Rp 0–200rb','profit':'Rp 1–3 jt/bln','waktu':'1 minggu',
        'steps':[
            {'title':'Kumpulkan toples kaca','desc':'Toples selai, toples bumbu, botol minum kaca — kumpulkan dari tetangga/komunitas.'},
            {'title':'Sterilisasi profesional','desc':'Rebus toples dalam air mendidih 10 menit. Keringkan di oven 120°C 15 menit.'},
            {'title':'Sortasi dan grade','desc':'Grade A: tanpa goresan, tutup sempurna. Grade B: goresan kecil, harga lebih rendah.'},
            {'title':'Foto dan listing','desc':'Foto per set ukuran yang sama. Sertakan info ukuran volume (ml) dan kondisi tutup.'},
            {'title':'Target pembeli UMKM','desc':'Hubungi produsen selai, madu, pickle, atau kue kering rumahan di Instagram.'},
        ],
        'checklist':['Toples kaca bersih berbagai ukuran','Panci besar untuk sterilisasi','Oven atau pengering','Label ukuran dan kondisi','Kantong bubble wrap untuk pengiriman'],
        'platforms':['Instagram','Shopee','Tokopedia','Facebook Marketplace','WhatsApp Business'],
        'tips':['Target UMKM makanan adalah pembeli tetap yang butuh stok rutin','Harga toples bekas steril jauh lebih murah dari toples baru — daya tarik utama','Bergabung komunitas UMKM makanan di Facebook untuk memperluas jaringan pembeli'],
    },
    'Mosaik Seni Kaca': {
        'icon':'🎨','waste':'Kaca','desc':'Pecahan kaca berwarna dibuat mosaik seni dinding dan aksesori.',
        'modal':'Rp 200rb–1 jt','profit':'Rp 2–6 jt/bln','waktu':'3–4 minggu belajar',
        'steps':[
            {'title':'Kumpulkan kaca berwarna','desc':'Botol kaca berwarna (hijau, coklat, biru), piring kaca tua, cermin bekas.'},
            {'title':'Pecahkan dengan aman','desc':'Bungkus kaca dengan kain, pukul dengan palu. Gunakan kacamata dan sarung tangan.'},
            {'title':'Sortasi ukuran dan warna','desc':'Pisahkan pecahan per warna dan ukuran. Simpan dalam toples berlabel.'},
            {'title':'Buat desain di papan','desc':'Gambar desain di papan kayu/triplek, tempel pecahan kaca dengan tile adhesive.'},
            {'title':'Grouting','desc':'Isi celah dengan grout, lap bersih, tunggu kering 24 jam hingga mengkilap.'},
        ],
        'checklist':['Kacamata pelindung dan sarung tangan tebal','Palu dan kain pembungkus','Tile adhesive (lem keramik)','Grout dan spons','Papan kayu/triplek sebagai media','Pernis untuk finishing'],
        'platforms':['Instagram','Shopee','Etsy','Pinterest','Pasar kreatif'],
        'tips':['Panel mosaik dekoratif untuk dinding diminati restoran dan hotel boutique','Workshop mosaik kaca bisa jadi pemasukan tambahan Rp 100rb–200rb/peserta','Foto close-up detail kaca berkilap sangat menarik di Instagram dan Pinterest'],
    },
    'Kerajinan Kaleng Artistik': {
        'icon':'🪔','waste':'Logam','desc':'Kaleng bekas jadi tempat pensil, pot bunga, lampu taman berlubang.',
        'modal':'Rp 100rb–400rb','profit':'Rp 2–5 jt/bln','waktu':'1–2 minggu',
        'steps':[
            {'title':'Kumpulkan kaleng beragam','desc':'Kaleng susu, cat, biskuit berbagai ukuran. Cuci bersih dan keringkan.'},
            {'title':'Isi dengan es sebelum dilubangi','desc':'Bekukan air di dalam kaleng — tidak penyok saat dipaku. Tips penting ini!'},
            {'title':'Desain pola lubang','desc':'Gambar pola di kertas, tempel ke kaleng. Pola geometris atau bunga paling populer.'},
            {'title':'Lubangi dengan paku/bor','desc':'Paku berbagai ukuran untuk variasi lubang. Bor listrik lebih presisi dan rapi.'},
            {'title':'Finishing dan cat','desc':'Amplas tepi tajam, cat semprot metalik/matte. Warna copper dan matte hitam premium.'},
        ],
        'checklist':['Kaleng bekas berbagai ukuran','Paku berbagai ukuran dan palu','Bor listrik kecil + mata bor metal','Cat semprot warna pilihan','Amplas kasar dan halus','Fitting lampu E27 kecil (untuk lampu taman)'],
        'platforms':['Instagram','Pinterest','Shopee','Pasar kreasi','Facebook Marketplace'],
        'tips':['Lampu taman dari kaleng sangat laku di musim pernikahan dan dekorasi outdoor','Kombinasi kaleng + sukulen jadi paket "dekorasi meja" yang populer','Pesan massal dari kafe dan restoran bisa jadi kontrak tetap bulanan'],
    },
    'Aksesori & Suvenir Logam': {
        'icon':'💍','waste':'Logam','desc':'Pengecoran aluminium bekas jadi gelang, cincin, dan suvenir kustom.',
        'modal':'Rp 500rb–3 jt','profit':'Rp 3–8 jt/bln','waktu':'1–2 bln belajar',
        'steps':[
            {'title':'Kumpulkan aluminium bersih','desc':'Kaleng aluminium, tutup panci, sendok aluminium bekas — bersih dari cat.'},
            {'title':'Siapkan tungku sederhana','desc':'Tungku arang + krusibel (wadah tahan panas) bisa dibuat sendiri dengan batu bata.'},
            {'title':'Lebur aluminium','desc':'Panaskan di ±660°C (bisa cek dengan termometer infrared). Gunakan APD lengkap.'},
            {'title':'Tuang ke cetakan','desc':'Cetakan pasir (sand casting) paling mudah untuk pemula. Bisa cetak berbagai bentuk.'},
            {'title':'Finishing','desc':'Amplas, poles dengan metal polish, ukir desain custom, atau plating warna.'},
        ],
        'checklist':['APD lengkap (kacamata, sarung tangan kulit, sepatu safety)','Krusibel tahan panas','Termometer infrared','Pasir khusus cetak (greensand)','Metal polish dan amplas berbagai grit','Alat ukir/graver'],
        'platforms':['Instagram','Etsy','Tokopedia','Shopee','Pasar kreatif'],
        'tips':['Suvenir wisata kustom dengan logo daerah diminati oleh toko souvenir','Gelang dan cincin custom nama/inisial laku di acara pernikahan','Dokumentasikan proses pengecoran — sangat viral di YouTube dan TikTok'],
    },
    'Miniatur & Patung Logam': {
        'icon':'🗿','waste':'Logam','desc':'Patung dan miniatur dari kaleng dirangkai dan dicat jadi karya seni.',
        'modal':'Rp 200rb–1 jt','profit':'Rp 2–7 jt/bln','waktu':'3–4 minggu belajar',
        'steps':[
            {'title':'Kumpulkan kaleng beragam','desc':'Kaleng berbagai ukuran sebagai "tubuh" patung. Kawat untuk sambungan fleksibel.'},
            {'title':'Desain karakter','desc':'Sketsa karakter/miniatur dulu di kertas. Robot, hewan, karakter game populer.'},
            {'title':'Potong dan bentuk','desc':'Gunting kaleng jadi bagian-bagian. Gunting khusus logam (tin snips) untuk hasil rapi.'},
            {'title':'Rangkai dengan kawat/solder','desc':'Sambung bagian dengan kawat tebal atau solder timah untuk sambungan permanen.'},
            {'title':'Cat dan finishing','desc':'Cat dengan cat logam, tambahkan detail dengan kuas kecil. Vernish untuk perlindungan.'},
        ],
        'checklist':['Kaleng berbagai ukuran','Gunting logam (tin snips)','Kawat tebal dan solder (opsional)','Cat logam + kuas detail','Tang kombinasi','Vernish tahan karat'],
        'platforms':['Instagram','Etsy','Tokopedia','Pasar kreatif','Facebook Marketplace'],
        'tips':['Karakter game/anime populer selalu laku — pantau tren','Limited edition seri karakter meningkatkan nilai koleksi','Workshop pembuatan patung kaleng jadi sumber penghasilan tambahan'],
    },
    'Panel Dekorasi Bangunan': {
        'icon':'🏠','waste':'Logam','desc':'Kaleng dipress jadi panel dekoratif untuk atap gazebo dan dinding taman.',
        'modal':'Rp 500rb–2 jt','profit':'Rp 3–6 jt/bln','waktu':'1 bln',
        'steps':[
            {'title':'Kumpulkan kaleng besar','desc':'Kaleng cat 5kg, kaleng minyak, drum bekas ukuran kecil — kaleng besar lebih efisien.'},
            {'title':'Potong dan press datar','desc':'Potong sepanjang kaleng, gunting tepi, press datar dengan palu di atas landasan datar.'},
            {'title':'Emboss pola','desc':'Tempelkan stensil pola, tekan dengan stylus atau ballpoint untuk membuat tekstur timbul.'},
            {'title':'Cat tahan cuaca','desc':'Gunakan cat besi anti karat. Warna natural (antik, tembaga, hijau patina) paling populer.'},
            {'title':'Pasang dan presentasikan','desc':'Foto panel terpasang di gazebo/taman untuk portofolio penawaran ke klien.'},
        ],
        'checklist':['Kaleng besar atau drum kecil bekas','Gunting logam dan palu','Cat besi anti karat','Stensil pola (bisa cetak sendiri)','Paku/sekrup untuk pemasangan','Cat primer sebelum finishing'],
        'platforms':['Instagram','WhatsApp Business','Facebook Marketplace','Pasar kreatif'],
        'tips':['Target pasar: kontraktor, pemilik kafe outdoor, taman perumahan','Foto before-after taman yang dipasangi panel sangat meyakinkan klien','Satu proyek gazebo bisa menghasilkan Rp 500rb–2 jt dari material bekas'],
    },
    'Paper Mache & Kerajinan': {
        'icon':'🎭','waste':'Kertas','desc':'Bubur kertas dibentuk jadi topeng, patung, dekorasi bernilai seni tinggi.',
        'modal':'Rp 150rb–500rb','profit':'Rp 2–5 jt/bln','waktu':'1–2 minggu belajar',
        'steps':[
            {'title':'Buat pasta paper mache','desc':'1 bagian tepung terigu + 2 bagian air, masak hingga mengental. Atau lem kanji encer.'},
            {'title':'Robek kertas jadi strip','desc':'Robek (jangan gunting) koran/majalah jadi strip ±2x5cm. Permukaan sobek lebih menempel.'},
            {'title':'Bentuk dengan cetakan','desc':'Lapisi balon/mangkuk dengan petroleum jelly. Tempel strip berlapis-lapis (3–5 lapis).'},
            {'title':'Keringkan antar lapisan','desc':'Tunggu 4–6 jam per lapisan. Lapisan terakhir dengan tisu halus untuk permukaan mulus.'},
            {'title':'Cat dan vernish','desc':'Primer putih dulu, lalu cat akrilik sesuai desain. Vernish glossy untuk kilap tahan lama.'},
        ],
        'checklist':['Koran atau majalah bekas banyak','Tepung terigu atau lem kanji','Cetakan (balon, mangkuk)','Petroleum jelly anti lengket','Cat akrilik berbagai warna','Vernish transparan'],
        'platforms':['Instagram','Shopee','Tokopedia','Pasar seni','Sekolah & workshop'],
        'tips':['Topeng tradisional bermotif batik/wayang diminati turis dan kolektor','Workshop paper mache untuk anak-anak: Rp 50rb–100rb/peserta','Produk bertema hari raya (lebaran, natal) laku keras di momen tertentu'],
    },
    'Buku Catatan Handmade': {
        'icon':'📓','waste':'Kertas','desc':'Kertas daur ulang dijilid jadi buku catatan premium dengan cover artistik.',
        'modal':'Rp 200rb–600rb','profit':'Rp 3–7 jt/bln','waktu':'1–2 minggu',
        'steps':[
            {'title':'Proses bubur kertas','desc':'Rendam kertas 24 jam, blender halus, saring dan cetak di frame kawat jadi lembaran.'},
            {'title':'Keringkan lembaran','desc':'Tekan di antara kain menyerap, jemur matahari atau hair dryer.'},
            {'title':'Potong ukuran standar','desc':'A5 atau A6 paling populer. Kumpulkan 48–64 lembar per buku.'},
            {'title':'Buat cover artistik','desc':'Cover dari kain perca, kulit sintetis, papan kayu tipis, atau kardus tebal dicat.'},
            {'title':'Jilid dengan benang','desc':'Coptic binding atau Japanese stab binding — tanpa lem, sangat estetik dan kuat.'},
        ],
        'checklist':['Kertas bekas berbagai jenis','Blender atau mesin pulp','Frame kawat mold cetak','Kain menyerap (handuk tipis)','Bahan cover','Jarum & benang waxed bookbinding'],
        'platforms':['Instagram','Tokopedia','Shopee','Pasar kreatif','Etsy'],
        'tips':['Set buku 3 ukuran berbeda lebih laku dari satuan','Custom nama/inisial di cover: +Rp 15–30rb per unit','Target: mahasiswa desain, penulis jurnal, kolektor alat tulis premium'],
    },
    'Wrapping Paper Custom': {
        'icon':'🎁','waste':'Kertas','desc':'Kertas koran dicetak ulang motif unik untuk pembungkus kado artisan.',
        'modal':'Rp 100rb–400rb','profit':'Rp 1–3 jt/bln','waktu':'1 minggu',
        'steps':[
            {'title':'Kumpulkan koran lama','desc':'Koran lama berbahasa asing lebih estetik (terlihat vintage). Majalah juga bisa.'},
            {'title':'Buat stempel motif','desc':'Stempel dari foam/karet atau print screen printing motif geometris.'},
            {'title':'Cetak motif di atas koran','desc':'Gunakan cat berbahan dasar air (tidak tembus/luntur). Keringkan flat.'},
            {'title':'Potong ukuran dan gulung','desc':'Potong sesuai ukuran, gulung di tube kardus, ikat pita, tambahkan label brand.'},
            {'title':'Buat paket penjualan','desc':'Bundel 3 lembar + 1 pita + 1 kartu = satu paket gift wrapping premium.'},
        ],
        'checklist':['Koran/majalah lama','Cat akrilik encer untuk stempel','Stempel karet atau busa','Pita dan kartu ucapan','Tube kardus untuk menggulung','Label brand'],
        'platforms':['Shopee','Tokopedia','Instagram','TikTok Shop'],
        'tips':['Motif bertema hari raya (batik, christmas, lunar new year) laku keras','Paket gift wrapping lengkap lebih menarik dari lembaran satuan','Kolaborasi dengan toko kue/bakery untuk jadi supplier wrapping mereka'],
    },
    'Pot Semai dari Koran': {
        'icon':'🌱','waste':'Kertas','desc':'Pot semai biodegradable dari koran gulung diminati petani dan hobi tanam.',
        'modal':'Rp 50rb–200rb','profit':'Rp 1–4 jt/bln','waktu':'3–5 hari',
        'steps':[
            {'title':'Siapkan alat gulung','desc':'Botol kaca kecil sebagai cetakan. Gulung koran basah rapat di sekitar botol.'},
            {'title':'Bentuk dasar pot','desc':'Lipat bagian bawah ke dalam seperti melipat kertas, tekan kuat, keluarkan dari cetakan.'},
            {'title':'Keringkan pot','desc':'Susun di atas nampan, jemur 1–2 hari hingga kaku dan kokoh.'},
            {'title':'Isi media tanam','desc':'Pot siap diisi tanah + kompos. Tanam benih langsung, pot bisa langsung ditanam ke tanah.'},
            {'title':'Jual per set/pack','desc':'Jual per 10 atau 20 pot dalam plastik transparan. Target petani, toko tanaman, hobi tanam.'},
        ],
        'checklist':['Koran bekas banyak','Botol kaca kecil sebagai cetakan','Nampan pengering','Plastik kemasan transparan','Label harga dan instruksi pemakaian'],
        'platforms':['Shopee','Tokopedia','Instagram','Facebook Marketplace','Grab/Gojek'],
        'tips':['Pot koran biodegradable sangat diminati komunitas urban farming yang terus berkembang','Paket "pot semai + benih sayur" jadi produk lengkap yang nilai jualnya lebih tinggi','Konten video cara menanam dengan pot koran viral di TikTok dan YouTube'],
    },
    'Ecobrick': {
        'icon':'🧱','waste':'Plastik','desc':'Botol PET diisi plastik lunak rapat jadi bata konstruksi furniture dan taman.',
        'modal':'Rp 0–200rb','profit':'Rp 500rb–2 jt/bln','waktu':'1 minggu',
        'steps':[
            {'title':'Kumpulkan bahan','desc':'Botol PET 600ml–1.5L dan plastik lunak bersih: sachet, kresek, stiker, bungkus makanan.'},
            {'title':'Cuci dan keringkan','desc':'Semua plastik harus bersih dan kering — plastik kotor akan berjamur di dalam botol.'},
            {'title':'Isi botol rapat','desc':'Masukkan plastik sedikit-sedikit, padatkan dengan tongkat kayu/bambu terus menerus.'},
            {'title':'Ukur dan catat berat','desc':'Standar min 0.33 g/ml. Botol 600ml harus ≥200gr. Catat di label yang ditempel.'},
            {'title':'Kumpulkan dan jual','desc':'Jual ke sekolah, komunitas, atau pembuat furniture. Harga Rp 3.000–8.000/buah.'},
        ],
        'checklist':['Botol PET berbagai ukuran','Tongkat pendorong (kayu/bambu)','Timbangan digital kecil','Plastik lunak bersih dan kering','Label stiker keterangan berat'],
        'platforms':['Facebook Group','Instagram','Shopee','Pasar kreatif'],
        'tips':['Program ecobrick di sekolah bisa jadi pemasukan reguler','Furniture ecobrick dijual jauh lebih mahal dari ecobrick mentah','Foto progres pengisian ecobrick sangat menarik di media sosial'],
    },
    'Filamen 3D Printer': {
        'icon':'🖨️','waste':'Plastik','desc':'Botol PET diolah jadi filamen printer 3D dengan nilai tinggi di komunitas maker.',
        'modal':'Rp 3–8 jt','profit':'Rp 5–15 jt/bln','waktu':'1–3 bln setup',
        'steps':[
            {'title':'Riset mesin','desc':'Cari "PET bottle to filament machine" di marketplace. Harga Rp 3–8 juta atau rakit sendiri.'},
            {'title':'Kumpulkan botol PET','desc':'Botol PET kode 1 — botol air mineral, teh, jus. Harus bersih dan kering tanpa tutup.'},
            {'title':'Potong jadi strip','desc':'Potong botol jadi strip panjang dengan alat pemotong (banyak tutorial DIY di YouTube).'},
            {'title':'Cetak filamen','desc':'Mesin panaskan strip PET dan tarik jadi filamen 1.75mm. Gulung di spool standar.'},
            {'title':'QC dan kemas','desc':'Ukur diameter dengan calipers (toleransi ±0.05mm). Kemas per 500gr/1kg, kedap udara.'},
        ],
        'checklist':['Mesin PET to filament atau kit DIY','Alat pemotong strip botol','Spool filamen kosong','Calipers digital','Kantong ziplock 1kg kedap udara','Timbangan digital 0.1gr'],
        'platforms':['Facebook Group','Discord','Tokopedia','Shopee','Kaskus FJB'],
        'tips':['rPET diminati komunitas eco-conscious 3D printing yang terus berkembang','Dokumentasikan proses produksi untuk konten yang sangat menarik','Filamen warna natural (transparan) langsung dijual tanpa pewarna tambahan'],
    },
    'Tas & Aksesori Upcycle': {
        'icon':'👜','waste':'Plastik','desc':'Kemasan plastik berwarna dianyam jadi tas, dompet, dan aksesori eco fashion.',
        'modal':'Rp 200rb–800rb','profit':'Rp 3–8 jt/bln','waktu':'2–4 minggu belajar',
        'steps':[
            {'title':'Pilih kemasan berwarna','desc':'Bungkus kopi, minuman sachet, plastik hologram. Cuci dan keringkan semua bahan.'},
            {'title':'Potong jadi strip','desc':'Gunting kemasan jadi strip lebar 1–2 cm. Panjang strip menentukan lebar anyaman.'},
            {'title':'Pelajari teknik anyam','desc':'Mulai teknik anyam silang sederhana (over-under). Banyak tutorial gratis di YouTube.'},
            {'title':'Buat rangka tas','desc':'Kawat atau karton tebal sebagai rangka agar tas punya bentuk yang kokoh.'},
            {'title':'Finishing dan branding','desc':'Pasang resleting (Rp 5–15rb), tali pundak, label brand. Foto dengan model.'},
        ],
        'checklist':['Kemasan plastik berwarna-warni bersih banyak','Gunting tajam','Jarum tapestry besar','Resleting berbagai ukuran','Kawat atau karton untuk rangka','Label brand dan tag produk'],
        'platforms':['Instagram','TikTok Shop','Shopee','Etsy','Pasar kreatif'],
        'tips':['Branding "eco fashion" sangat powerful untuk milenial dan Gen-Z','Kolaborasi dengan influencer sustainability untuk promosi efektif','Ekspor ke Etsy bisa menghasilkan 3–5x lipat harga jual lokal'],
    },
    'Paving Block Plastik': {
        'icon':'🏗️','waste':'Plastik','desc':'Plastik dicampur pasir dicetak jadi paving block ringan dan tahan air.',
        'modal':'Rp 1–3 jt','profit':'Rp 5–12 jt/bln','waktu':'1–2 bln setup',
        'steps':[
            {'title':'Kumpulkan plastik campuran','desc':'Plastik PP, HDPE, LDPE dari berbagai sumber. Tidak perlu sortasi ketat untuk paving.'},
            {'title':'Cacah plastik','desc':'Cacah dengan shredder atau gunting manual jadi potongan ±2cm untuk efisiensi leleh.'},
            {'title':'Campur dengan pasir','desc':'Rasio 1:2 (plastik:pasir). Panaskan plastik hingga leleh, campur pasir, aduk rata.'},
            {'title':'Cetak dalam cetakan','desc':'Tuang campuran panas ke cetakan besi paving. Tekan padat, tunggu dingin.'},
            {'title':'Uji kualitas dan jual','desc':'Uji kekuatan tekan. Target pasar: kontraktor, pengembang perumahan, RT/RW.'},
        ],
        'checklist':['Plastik campuran bersih dalam jumlah banyak','Shredder atau alat cacah','Kompor/tungku pelebur','Cetakan besi paving (beli atau buat custom)','APD: kacamata, sarung tangan, masker','Timbangan industri'],
        'platforms':['WhatsApp Business','Instagram','Facebook Marketplace','Pasar kreatif'],
        'tips':['Paving block dari plastik 40% lebih ringan dari beton biasa — daya tarik utama','Satu batch produksi bisa menghasilkan ratusan paving dari plastik yang tidak terpakai','Sertifikasi SNI meningkatkan kepercayaan dan harga jual secara signifikan'],
    },
    'Kompos & Pupuk Organik': {
        'icon':'🌿','waste':'Sampah Umum','desc':'Sisa makanan dan organik diolah jadi kompos. Jual ke petani dan urban farming.',
        'modal':'Rp 100rb–500rb','profit':'Rp 500rb–2 jt/bln','waktu':'2–4 minggu proses',
        'steps':[
            {'title':'Pisahkan sampah organik','desc':'Pisahkan sisa makanan, sayuran, buah, daun kering dari sampah anorganik sejak awal.'},
            {'title':'Siapkan komposter','desc':'Ember/tong berlubang atau lubang di tanah. Tambahkan EM4 (activator) untuk mempercepat.'},
            {'title':'Layering bahan','desc':'Lapisi: bahan hijau (basah) + bahan coklat (kering) bergantian. Kelembaban 50–60%.'},
            {'title':'Balik rutin','desc':'Balik kompos setiap 3–5 hari untuk aerasi. Kompos matang dalam 3–6 minggu.'},
            {'title':'Kemas dan jual','desc':'Ayak kompos matang, kemas per kg dalam kantong. Jual ke petani, toko tanaman, komunitas.'},
        ],
        'checklist':['Komposter (ember/tong berlubang)','EM4 activator kompos','Sekop kecil dan sarung tangan','Ayakan kompos','Kantong kemasan 1kg dan 5kg','Timbangan dan label'],
        'platforms':['Instagram','Shopee','Tokopedia','Facebook Group','Grab/Gojek'],
        'tips':['Komunitas urban farming adalah pasar tetap yang terus berkembang pesat','Pupuk organik cair (POC) dari kompos bisa dijual lebih mahal dari kompos padat','Konten edukatif cara membuat kompos sangat viral dan membangun kepercayaan calon pembeli'],
    },
    'Biogas dari Sampah Organik': {
        'icon':'⚡','waste':'Sampah Umum','desc':'Sampah organik difermentasi menghasilkan gas bakar untuk kebutuhan rumah tangga.',
        'modal':'Rp 500rb–2 jt','profit':'Rp 1–3 jt/bln','waktu':'1–2 bln setup',
        'steps':[
            {'title':'Buat digester sederhana','desc':'Dua drum plastik 200L yang disambung — satu tempat fermentasi, satu penampung gas.'},
            {'title':'Kumpulkan bahan organik','desc':'Kotoran ternak, sisa makanan, limbah dapur. Jangan campur dengan plastik/logam.'},
            {'title':'Isi dan tutup rapat','desc':'Isi digester 70%, tambahkan air 1:1, tutup kedap udara. Proses 2–4 minggu.'},
            {'title':'Instalasi kompor gas','desc':'Sambungkan selang ke kompor gas biasa. Gas langsung bisa dipakai untuk memasak.'},
            {'title':'Jual sisa digestat','desc':'Cairan sisa fermentasi adalah pupuk cair berkualitas tinggi. Dikemas dan dijual.'},
        ],
        'checklist':['2 drum plastik 200L','Fitting dan selang gas','Kompor biogas','APD dan sealant untuk sambungan kedap','Meter gas sederhana','Kantong kemasan pupuk cair'],
        'platforms':['YouTube','Instagram','WhatsApp Business','Facebook Group'],
        'tips':['Satu unit biogas bisa mengurangi pengeluaran gas elpiji keluarga Rp 200rb/bln','Digestat (limbah cair) dari biogas adalah pupuk organik premium yang bisa dijual','Program biogas komunal di RT/RW bisa mendapat bantuan dari dinas lingkungan'],
    },
    'Seni Instalasi Sampah': {
        'icon':'🎪','waste':'Sampah Umum','desc':'Sampah non-daur ulang jadi instalasi seni edukatif. Diminati sekolah dan event.',
        'modal':'Rp 200rb–1 jt','profit':'Rp 2–8 jt/bln','waktu':'per proyek',
        'steps':[
            {'title':'Kumpulkan sampah unik','desc':'Styrofoam, mainan rusak, kemasan unik, barang elektronik rusak — semakin beragam semakin kaya.'},
            {'title':'Buat konsep instalasi','desc':'Tema lingkungan yang kuat: gunung sampah, laut plastik, hutan mati — visual yang menghentak.'},
            {'title':'Bangun struktur','desc':'Rangka dari kayu/pipa PVC, tempelkan dan ikat sampah ke struktur sesuai konsep.'},
            {'title':'Tambahkan elemen edukatif','desc':'QR code, teks fakta lingkungan, angka statistik sampah — buat instalasi bermakna.'},
            {'title':'Presentasikan ke sekolah','desc':'Tawarkan paket: instalasi + workshop edukasi sampah. Harga Rp 500rb–5jt per event.'},
        ],
        'checklist':['Sampah non-daur ulang beragam jenis','Rangka kayu/pipa PVC','Kawat dan tali pengikat','Cat semprot untuk finishing','Printer untuk teks/QR code edukatif','Alat dokumentasi (kamera)'],
        'platforms':['Instagram','YouTube','WhatsApp Business','Facebook Group'],
        'tips':['Instalasi yang "instagrammable" menarik perhatian media dan event organizer','Program sekolah biasanya punya anggaran untuk kegiatan lingkungan hidup','Dokumentasi yang bagus membuka peluang undangan ke festival dan pameran seni'],
    },
}

# ── Routes: Public ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    stats  = db_query("SELECT COUNT(*) as total, AVG(confidence)*100 as avg_conf FROM classifications", fetchone=True) or {'total':0,'avg_conf':0}
    recent = db_query("SELECT * FROM v_recent_classifications LIMIT 6", fetchall=True) or []
    dist   = db_query("SELECT predicted_class, COUNT(*) as cnt FROM classifications GROUP BY predicted_class", fetchall=True) or []
    gallery= db_query("SELECT * FROM umkm_gallery WHERE is_active=1 ORDER BY created_at DESC LIMIT 8", fetchall=True) or []
    model_ready = os.path.exists(os.path.join(MODEL_DIR,'waste_classifier.keras'))
    return render_template('index.html', stats=stats, recent=recent, distribution=dist,
                           class_info=CLASS_INFO, model_ready=model_ready, gallery=gallery)

@app.route('/classify', methods=['GET','POST'])
def classify():
    if request.method == 'GET':
        return render_template('classify.html', class_info=CLASS_INFO)
    if 'file' not in request.files:
        return jsonify({'error':'Tidak ada file yang dikirim'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error':'Format tidak didukung. Gunakan JPG, PNG, atau WEBP'}), 400
    saved_name, orig_name, img_path = save_upload(file)
    try:
        result = predict_image(img_path)
    except Exception as e:
        if os.path.exists(img_path): os.remove(img_path)
        return jsonify({'error':f'Prediksi gagal: {e}'}), 500
    user_id = session.get('user_id')
    record_id = db_query(
        "INSERT INTO classifications (user_id,image_filename,original_filename,predicted_class,confidence,probabilities,processing_time,ip_address) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (user_id, saved_name, orig_name, result['predicted_class'], result['confidence'],
         json.dumps(result['probabilities']), result['processing_time'], request.remote_addr))
    cls  = result['predicted_class']
    info = CLASS_INFO.get(cls, {})
    tips = db_query("SELECT recycling_tips, description FROM waste_categories WHERE name=%s", (cls,), fetchone=True) or {}
    return jsonify({'success':True,'id':record_id,'predicted_class':cls,
        'label':info.get('label',cls),'icon':info.get('icon','🗑️'),'color':info.get('color','#888'),
        'confidence':round(result['confidence']*100,2),'is_recyclable':info.get('recyclable',False),
        'probabilities':{k:round(v*100,2) for k,v in result['probabilities'].items()},
        'processing_time':result['processing_time'],'demo_mode':result.get('demo_mode',False),
        'recycling_tips':tips.get('recycling_tips',''),'description':tips.get('description',''),
        'image_url':url_for('static',filename=f'uploads/{saved_name}')})

@app.route('/history')
def history():
    page, per_page = int(request.args.get('page',1)), 20
    offset = (page-1)*per_page
    rows  = db_query("SELECT c.*, ROUND(c.confidence*100,2) as confidence_pct, wc.color_code,wc.icon,wc.label,wc.is_recyclable FROM classifications c LEFT JOIN waste_categories wc ON c.predicted_class=wc.name ORDER BY c.created_at DESC LIMIT %s OFFSET %s",(per_page,offset),fetchall=True) or []
    total = (db_query("SELECT COUNT(*) as cnt FROM classifications",fetchone=True) or {'cnt':0})['cnt']
    pages = (total+per_page-1)//per_page
    return render_template('history.html', rows=rows, page=page, pages=pages, total=total, class_info=CLASS_INFO)

@app.route('/stats')
def stats():
    dist_raw = db_query("SELECT * FROM v_classification_summary",fetchall=True) or []
    daily_raw = db_query("SELECT DATE(created_at) as date, COUNT(*) as cnt FROM classifications WHERE created_at >= DATE_SUB(NOW(),INTERVAL 30 DAY) GROUP BY DATE(created_at) ORDER BY date",fetchall=True) or []
    summary_raw = db_query("SELECT COUNT(*) as total, AVG(confidence)*100 as avg_conf, MAX(created_at) as last_pred FROM classifications",fetchone=True) or {}

    distribution = [{
        'predicted_class': row.get('predicted_class'),
        'total': int(row.get('total') or 0),
        'avg_confidence_pct': float(row.get('avg_confidence_pct') or 0),
        'percentage': float(row.get('percentage') or 0),
    } for row in dist_raw]

    daily = [{
        'date': row.get('date').strftime('%Y-%m-%d') if row.get('date') else None,
        'cnt': int(row.get('cnt') or 0),
    } for row in daily_raw]

    summary = {
        'total': int(summary_raw.get('total') or 0),
        'avg_conf': float(summary_raw.get('avg_conf') or 0),
        'last_pred': summary_raw.get('last_pred').strftime('%d/%m/%Y') if summary_raw.get('last_pred') else None,
    }

    return render_template('stats.html', distribution=distribution, daily=daily, summary=summary, class_info=CLASS_INFO)

# ── Routes: Auth ───────────────────────────────────────────────────────────────

def db_connected():
    return db_query("SELECT 1 as ok", fetchone=True) is not None

@app.route('/login', methods=['GET','POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard') if session.get('role')=='admin' else url_for('index'))
    if request.method == 'POST':
        login_id = request.form.get('username','').strip()
        password = request.form.get('password','')
        if login_id.isdigit():
            user = db_query("SELECT * FROM users WHERE id=%s AND is_active=1", (int(login_id),), fetchone=True)
        else:
            user = db_query("SELECT * FROM users WHERE username=%s AND is_active=1",(login_id,),fetchone=True)
        if user and verify_password_hash(password, user.get('password_hash')):
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            session['full_name']= user.get('full_name','')
            db_query("UPDATE users SET last_login=NOW() WHERE id=%s",(user['id'],))
            flash('Login berhasil! Selamat datang, '+user['username'],'success')
            return redirect(url_for('dashboard') if user['role']=='admin' else url_for('index'))
        if not user:
            flash('Akun tidak ditemukan. Pastikan username sudah terdaftar dan akun aktif.','error')
            return render_template('login.html')
        if not db_connected():
            flash('Tidak dapat terhubung ke database. Pastikan MySQL berjalan dan konfigurasi DB benar.','error')
        else:
            flash('Username atau password salah.','error')
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username  = request.form.get('username','').strip()
        email     = request.form.get('email','').strip()
        full_name = request.form.get('full_name','').strip()
        password  = request.form.get('password','')
        confirm   = request.form.get('confirm_password','')
        # Validasi
        errors = []
        if len(username) < 3: errors.append('Username minimal 3 karakter.')
        if len(password) < 6: errors.append('Password minimal 6 karakter.')
        if password != confirm: errors.append('Konfirmasi password tidak cocok.')
        if not email or '@' not in email: errors.append('Format email tidak valid.')
        if errors:
            for e in errors: flash(e,'error')
            return render_template('register.html', form=request.form)
        existing = db_query("SELECT id FROM users WHERE username=%s OR email=%s",(username,email),fetchone=True)
        if existing:
            flash('Username atau email sudah terdaftar.','error')
            return render_template('register.html', form=request.form)
        pw_hash = generate_password_hash(password)
        user_id = db_query("INSERT INTO users (username,password_hash,email,full_name,role) VALUES (%s,%s,%s,%s,'user')",
                           (username, pw_hash, email, full_name))
        if not user_id:
            flash('Pendaftaran gagal tersimpan ke database. Pastikan MySQL aktif dan tabel users tersedia.','error')
            return render_template('register.html', form=request.form)
        flash('Akun berhasil dibuat! Silakan login.','success')
        return redirect(url_for('login'))
    return render_template('register.html', form={})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ── Routes: Admin Dashboard ───────────────────────────────────────────────────
@app.route('/dashboard')
@admin_required
def dashboard():
    stats_data = db_query("SELECT COUNT(*) as total, AVG(confidence)*100 as avg_conf FROM classifications",fetchone=True) or {}
    dist       = db_query("SELECT * FROM v_classification_summary",fetchall=True) or []
    recent     = db_query("SELECT * FROM v_recent_classifications LIMIT 10",fetchall=True) or []
    user_count = (db_query("SELECT COUNT(*) as cnt FROM users",fetchone=True) or {'cnt':0})['cnt']
    model_info = {}
    mp = os.path.join(MODEL_DIR,'model_info.json')
    if os.path.exists(mp):
        with open(mp) as f: model_info = json.load(f)
    return render_template('dashboard.html', stats=stats_data, distribution=dist, recent=recent,
                           model_info=model_info, class_info=CLASS_INFO, user_count=user_count)

@app.route('/admin/delete/<int:record_id>', methods=['POST'])
@admin_required
def delete_record(record_id):
    row = db_query("SELECT image_filename FROM classifications WHERE id=%s",(record_id,),fetchone=True)
    if row:
        p = os.path.join(UPLOAD_DIR, row['image_filename'])
        if os.path.exists(p): os.remove(p)
        db_query("DELETE FROM classifications WHERE id=%s",(record_id,))
    return redirect(request.referrer or url_for('dashboard'))

# ── Routes: Admin — Manajemen User ────────────────────────────────────────────
@app.route('/admin/users')
@admin_required
def admin_users():
    users = db_query("SELECT * FROM v_user_stats ORDER BY created_at DESC",fetchall=True) or []
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/<int:uid>/detail')
@admin_required
def admin_user_detail(uid):
    user = db_query("SELECT * FROM v_user_stats WHERE id=%s",(uid,),fetchone=True)
    if not user: flash('User tidak ditemukan.','error'); return redirect(url_for('admin_users'))
    classifications = db_query(
        "SELECT c.*,wc.color_code,wc.icon,wc.label FROM classifications c LEFT JOIN waste_categories wc ON c.predicted_class=wc.name WHERE c.user_id=%s ORDER BY c.created_at DESC",
        (uid,),fetchall=True) or []
    # Per-class summary
    class_summary = db_query(
        "SELECT predicted_class, COUNT(*) as cnt, ROUND(AVG(confidence)*100,1) as avg_conf FROM classifications WHERE user_id=%s GROUP BY predicted_class ORDER BY cnt DESC",
        (uid,),fetchall=True) or []
    return render_template('admin_user_detail.html', user=user,
                           classifications=classifications, class_summary=class_summary, class_info=CLASS_INFO)

@app.route('/admin/users/<int:uid>/edit', methods=['POST'])
@admin_required
def admin_user_edit(uid):
    if uid == session['user_id']:
        flash('Tidak dapat mengedit akun sendiri melalui panel ini.','error')
        return redirect(url_for('admin_users'))
    role      = request.form.get('role','user')
    is_active = 1 if request.form.get('is_active') else 0
    full_name = request.form.get('full_name','').strip()
    email     = request.form.get('email','').strip()
    db_query("UPDATE users SET role=%s, is_active=%s, full_name=%s, email=%s WHERE id=%s",
             (role, is_active, full_name, email, uid))
    new_pw = request.form.get('new_password','').strip()
    if new_pw:
        if len(new_pw) < 6:
            flash('Password baru minimal 6 karakter.','error')
        else:
            db_query("UPDATE users SET password_hash=%s WHERE id=%s",(generate_password_hash(new_pw), uid))
            flash('Password berhasil diperbarui.','success')
    flash('Data user berhasil diperbarui.','success')
    return redirect(url_for('admin_user_detail', uid=uid))

@app.route('/admin/users/<int:uid>/delete', methods=['POST'])
@admin_required
def admin_user_delete(uid):
    if uid == session['user_id']:
        flash('Tidak dapat menghapus akun sendiri.','error')
        return redirect(url_for('admin_users'))
    db_query("UPDATE classifications SET user_id=NULL WHERE user_id=%s",(uid,))
    db_query("DELETE FROM users WHERE id=%s",(uid,))
    flash('User berhasil dihapus.','success')
    return redirect(url_for('admin_users'))

# ── Routes: Admin — UMKM Gallery ──────────────────────────────────────────────
@app.route('/admin/umkm-gallery', methods=['GET','POST'])
@admin_required
def admin_umkm_gallery():
    if request.method == 'POST':
        title      = request.form.get('title','').strip()
        waste_type = request.form.get('waste_type','')
        desc       = request.form.get('description','').strip()
        file       = request.files.get('image')
        if not title or not file or not allowed_file(file.filename):
            flash('Judul dan gambar wajib diisi.','error')
        else:
            saved_name, _, _ = save_upload(file, UMKM_DIR)
            db_query("INSERT INTO umkm_gallery (title,waste_type,image_filename,description,created_by) VALUES (%s,%s,%s,%s,%s)",
                     (title, waste_type, saved_name, desc, session['user_id']))
            flash('Foto UMKM berhasil diunggah.','success')
        return redirect(url_for('admin_umkm_gallery'))
    gallery = db_query("SELECT g.*, u.username FROM umkm_gallery g LEFT JOIN users u ON g.created_by=u.id ORDER BY g.created_at DESC",fetchall=True) or []
    return render_template('admin_umkm_gallery.html', gallery=gallery, class_info=CLASS_INFO)

@app.route('/admin/umkm-gallery/<int:gid>/toggle', methods=['POST'])
@admin_required
def admin_umkm_toggle(gid):
    db_query("UPDATE umkm_gallery SET is_active = NOT is_active WHERE id=%s",(gid,))
    return redirect(url_for('admin_umkm_gallery'))

@app.route('/admin/umkm-gallery/<int:gid>/delete', methods=['POST'])
@admin_required
def admin_umkm_delete(gid):
    row = db_query("SELECT image_filename FROM umkm_gallery WHERE id=%s",(gid,),fetchone=True)
    if row:
        p = os.path.join(UMKM_DIR, row['image_filename'])
        if os.path.exists(p): os.remove(p)
        db_query("DELETE FROM umkm_gallery WHERE id=%s",(gid,))
    flash('Foto berhasil dihapus.','success')
    return redirect(url_for('admin_umkm_gallery'))

# ── Routes: Edukasi & UMKM ────────────────────────────────────────────────────
WASTE_DATA = {
    'cardboard':{'label':'Kardus','icon':'📦','description':'Kotak packaging, karton gelombang, kotak sereal',
        'recycling_steps':[
            {'title':'Kumpulkan & Pisahkan','desc':'Kumpulkan kardus bekas, pisahkan dari plastik, lakban, dan staples.'},
            {'title':'Ratakan & Bersihkan','desc':'Ratakan kardus jadi lembar datar. Pastikan bebas minyak dan cairan.'},
            {'title':'Cacah / Shredding','desc':'Cacah kardus menjadi potongan kecil untuk mempercepat proses pulping.'},
            {'title':'Pulping','desc':'Rendam dalam air dan proses menjadi bubur kertas (pulp) menggunakan mesin.'},
            {'title':'Cetak & Keringkan','desc':'Cetak pulp jadi lembaran karton baru kemudian keringkan.'},
        ],
        'processing_steps':[
            {'title':'Sortasi','desc':'Pisahkan kardus coklat, karton warna, dan kertas tebal — nilai berbeda.'},
            {'title':'Pembersihan','desc':'Lepas lakban, staples, dan bahan non-karton.'},
            {'title':'Pemadatan','desc':'Ratakan dan ikat per 10–20 kg untuk kemudahan transportasi.'},
            {'title':'Penyimpanan','desc':'Simpan di tempat kering — kardus basah turun nilai hingga 50%.'},
        ],
        'market_prices':[{'item':'Kardus coklat bersih','price':'Rp 1.500–2.500/kg'},{'item':'Karton gelombang','price':'Rp 1.200–2.000/kg'}]},
    'glass':{'label':'Kaca','icon':'🫙','description':'Botol kaca, toples, wadah kaca berbagai warna',
        'recycling_steps':[
            {'title':'Sortasi Warna','desc':'Pisahkan kaca bening, hijau, dan coklat untuk menjaga kualitas.'},
            {'title':'Pembersihan','desc':'Bilas botol dan toples dari sisa isi. Lepas tutup logam atau plastik.'},
            {'title':'Penghancuran','desc':'Kaca dihancurkan menjadi cullet (pecahan kecil) menggunakan crusher.'},
            {'title':'Pemurnian','desc':'Cullet disortasi dari kontaminan logam, keramik, dan bahan asing.'},
            {'title':'Peleburan & Pembentukan','desc':'Cullet dilebur di tungku 1.500°C dan dibentuk jadi produk baru.'},
        ],
        'processing_steps':[
            {'title':'Cuci & Sterilisasi','desc':'Cuci dengan sabun dan air panas. Botol bersih bernilai lebih tinggi.'},
            {'title':'Pisahkan Kontaminan','desc':'Lepas tutup, label, dan bagian non-kaca sebelum disetor.'},
            {'title':'Kemas Aman','desc':'Kemas botol utuh dalam kardus. Jangan pecahkan kecuali ada crusher.'},
            {'title':'Jual Utuh vs Cullet','desc':'Botol utuh bisa dijual lebih mahal untuk reuse oleh produsen.'},
        ],
        'market_prices':[{'item':'Botol kaca bening utuh','price':'Rp 500–1.000/btl'},{'item':'Cullet (pecahan kaca)','price':'Rp 300–500/kg'}]},
    'metal':{'label':'Logam','icon':'🥫','description':'Kaleng aluminium, kaleng baja, tutup botol, logam campuran',
        'recycling_steps':[
            {'title':'Sortasi Jenis','desc':'Pisahkan aluminium dan baja menggunakan magnet. Baja menempel, aluminium tidak.'},
            {'title':'Pembersihan','desc':'Bilas kaleng dari sisa makanan. Hancurkan untuk efisiensi transportasi.'},
            {'title':'Shredding','desc':'Logam dicacah jadi potongan kecil untuk efisiensi peleburan.'},
            {'title':'Peleburan','desc':'Aluminium dilebur di 660°C, baja di 1.500°C — jauh hemat energi vs bahan baru.'},
            {'title':'Pengecoran','desc':'Logam cair dimurnikan dan dicetak menjadi ingot atau produk baru.'},
        ],
        'processing_steps':[
            {'title':'Identifikasi Material','desc':'Gunakan magnet — baja menempel, aluminium tidak. Pisahkan untuk nilai berbeda.'},
            {'title':'Bersihkan & Keringkan','desc':'Kaleng bersih dan kering bernilai lebih. Hindari kaleng berkarat berat.'},
            {'title':'Padatkan Volume','desc':'Injak kaleng aluminium untuk hemat ruang simpan dan transportasi.'},
            {'title':'Timbang & Jual','desc':'Kumpulkan minimal 10 kg untuk nilai jual yang signifikan.'},
        ],
        'market_prices':[{'item':'Kaleng aluminium bersih','price':'Rp 8.000–12.000/kg'},{'item':'Kaleng baja / tin can','price':'Rp 1.000–2.000/kg'}]},
    'paper':{'label':'Kertas','icon':'📄','description':'Kertas kantor, koran, majalah, buku bekas',
        'recycling_steps':[
            {'title':'Sortasi Kualitas','desc':'Pisahkan kertas HVS, koran, majalah glossy, dan kardus — nilai berbeda.'},
            {'title':'Pembersihan','desc':'Buang klip, staples, plastik. Kertas berminyak atau basah dipisahkan.'},
            {'title':'Pulping','desc':'Kertas direndam dan dihancurkan jadi bubur kertas dengan air.'},
            {'title':'De-inking','desc':'Tinta dihilangkan dari pulp menggunakan bahan kimia dan flotasi udara.'},
            {'title':'Pembentukan Baru','desc':'Pulp bersih dicetak, dipress, dan dikeringkan jadi kertas baru.'},
        ],
        'processing_steps':[
            {'title':'Jaga Kekeringan','desc':'Simpan kertas di tempat kering — kertas basah tidak laku dan berjamur.'},
            {'title':'Ikat per Kategori','desc':'Ikat kertas per jenis. HVS terpisah dari koran, majalah dipisah.'},
            {'title':'Buat Bundel 5–10 kg','desc':'Bundel rapi memudahkan penimbangan dan transportasi ke pengepul.'},
            {'title':'Hindari Kontaminasi','desc':'Jangan campur dengan tissue, popok, atau kertas berlaminasi plastik.'},
        ],
        'market_prices':[{'item':'Kertas HVS / duplex','price':'Rp 1.500–2.500/kg'},{'item':'Koran / majalah','price':'Rp 800–1.500/kg'}]},
    'plastic':{'label':'Plastik','icon':'♻️','description':'Botol PET, wadah makanan, plastik keras',
        'recycling_steps':[
            {'title':'Identifikasi Kode Resin','desc':'Cek kode segitiga (1–7). Kode 1 (PET) dan 2 (HDPE) paling mudah didaur ulang.'},
            {'title':'Pisahkan & Bersihkan','desc':'Bilas dari sisa makanan. Lepas tutup (material berbeda) dan label.'},
            {'title':'Shredding','desc':'Plastik dicacah jadi flakes kecil menggunakan mesin shredder.'},
            {'title':'Pencucian & Pemisahan','desc':'Flakes dicuci dan dipisahkan berdasarkan densitas dalam air.'},
            {'title':'Pelet & Produksi','desc':'Flakes dilelehkan dan dicetak jadi pelet plastik sebagai bahan baku baru.'},
        ],
        'processing_steps':[
            {'title':'Baca Kode Plastik','desc':'PET (1): botol air. HDPE (2): botol shampoo. PP (5): wadah makanan.'},
            {'title':'Pisahkan dari Plastik Lunak','desc':'Plastik keras (botol, wadah) pisah dari plastik lunak (kantong kresek).'},
            {'title':'Cuci Bersih','desc':'Botol bersih dari kontaminasi organik memiliki nilai jual lebih tinggi.'},
            {'title':'Padatkan Volume','desc':'Tekan botol plastik agar lebih efisien disimpan. Ikat per jenis.'},
        ],
        'market_prices':[{'item':'Botol PET bersih (kode 1)','price':'Rp 1.500–3.000/kg'},{'item':'HDPE botol shampoo (kode 2)','price':'Rp 2.000–4.000/kg'}]},
}

@app.route('/panduan')
def panduan():
    return render_template('panduan.html', waste_data=WASTE_DATA)

@app.route('/pengolahan')
def pengolahan():
    return render_template('pengolahan.html', waste_data=WASTE_DATA)

@app.route('/manfaat')
def manfaat():
    return render_template('manfaat.html')

@app.route('/umkm')
def umkm():
    return render_template('umkm.html', umkm_list=UMKM_LIST,
                           guides=UMKM_GUIDES, platform_urls=PLATFORM_URLS)

# ── API ────────────────────────────────────────────────────────────────────────
@app.route('/api/classify', methods=['POST'])
def api_classify():
    if 'file' not in request.files: return jsonify({'error':'Field "file" tidak ditemukan'}),400
    file = request.files['file']
    if not allowed_file(file.filename): return jsonify({'error':'Format tidak didukung'}),400
    saved_name,_,img_path = save_upload(file)
    result = predict_image(img_path)
    return jsonify({'predicted_class':result['predicted_class'],'confidence':round(result['confidence'],4),
                    'probabilities':result['probabilities'],'processing_time':result['processing_time']})

@app.route('/api/stats')
def api_stats():
    dist_raw = db_query("SELECT * FROM v_classification_summary",fetchall=True) or []
    distribution = [{
        'predicted_class': row.get('predicted_class'),
        'total': int(row.get('total') or 0),
        'avg_confidence_pct': float(row.get('avg_confidence_pct') or 0),
        'percentage': float(row.get('percentage') or 0),
    } for row in dist_raw]
    total = int((db_query("SELECT COUNT(*) as cnt FROM classifications",fetchone=True) or {'cnt':0})['cnt'])
    return jsonify({'total':total,'distribution':distribution})

@app.route('/api/categories')
def api_categories():
    return jsonify(db_query("SELECT * FROM waste_categories",fetchall=True) or [])

# ── Errors ─────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e): return render_template('error.html',code=404,msg='Halaman tidak ditemukan'),404
@app.errorhandler(413)
def too_large(e): return jsonify({'error':'File terlalu besar. Maksimal 16MB'}),413
@app.errorhandler(500)
def server_error(e): return render_template('error.html',code=500,msg='Terjadi kesalahan server'),500

# ── Startup ────────────────────────────────────────────────────────────────────
with app.app_context():
    init_database()
    ensure_users_schema()
    activate_all_users()
    load_model()

if __name__ == '__main__':
    print("\n"+"="*60)
    print("  ♻  ECOSCAN — Waste Classifier")
    print("  URL   : http://localhost:5000")
    print("  Admin : http://localhost:5000/login")
    print("="*60+"\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
