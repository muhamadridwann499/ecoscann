# ♻ EcoScan — Sistem Klasifikasi Sampah CNN
**Tugas Akhir 2025** | Flask + TensorFlow + MySQL

## Fitur Lengkap

### Pengguna
- ✅ Registrasi akun baru (`/register`) — validasi + strength meter password
- ✅ Login aman (`/login`) — halaman standalone tanpa navbar
- ✅ Klasifikasi sampah via upload gambar (CNN / mode demo)
- ✅ Riwayat klasifikasi
- ✅ Statistik & distribusi
- ✅ Panduan Daur Ulang per jenis sampah
- ✅ Cara Pengolahan lengkap
- ✅ Manfaat Daur Ulang (statistik nasional)
- ✅ Peluang UMKM 23 ide bisnis + filter + panduan lengkap + link platform aktif

### Admin Panel
- ✅ Dashboard admin (statistik, distribusi, riwayat terbaru)
- ✅ Manajemen User — lihat semua user, edit role/status/password, hapus user
- ✅ Detail User — profil + statistik + seluruh riwayat klasifikasi
- ✅ Galeri UMKM — upload foto hasil UMKM, toggle tampil/sembunyikan di beranda

## Instalasi

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup database MySQL
mysql -u root -p < database/waste_db.sql

# 3. (Opsional) Train model CNN
python train_model.py

# 4. Jalankan aplikasi
python app.py
```

Buka: http://localhost:5000

## Akun Default
- **Admin**: username `admin` / password `admin123`

## Struktur
```
waste_classifier/
├── app.py                   # Main Flask app
├── train_model.py           # Training CNN
├── database/waste_db.sql    # Schema + seed data
├── model/                   # Model CNN (setelah training)
├── static/
│   ├── css/main.css         # Tema utama
│   ├── css/sidebar.css      # Layout sidebar
│   ├── js/main.js
│   ├── uploads/             # Hasil upload klasifikasi
│   └── umkm_uploads/        # Foto galeri UMKM
└── templates/               # 16 halaman HTML
```

## Kategori Sampah
| Kategori | Dapat Daur Ulang |
|---|---|
| 📦 Kardus | ✅ Ya |
| 🫙 Kaca | ✅ Ya |
| 🥫 Logam | ✅ Ya |
| 📄 Kertas | ✅ Ya |
| ♻️ Plastik | ✅ Ya |
| 🗑️ Sampah Umum | ❌ Tidak |
