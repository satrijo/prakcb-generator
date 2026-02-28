#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import struct
import os
import sys
import time
import json
import re
from datetime import datetime, timedelta, timezone
from tqdm import tqdm


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    
    current_dir = os.getcwd()

OUTPUT_DIR    = os.path.join(current_dir, "CB")
LOGO_PATH     = os.path.join(current_dir, 'assets', 'BMKG.png')

SHP_PROVINCES = os.path.join(current_dir, "datadasar", "Provinsi.shp")
SHP_SEA       = os.path.join(current_dir, "datadasar", "Laut.shp")

DOMAIN = {'lon_min': 90, 'lon_max': 145, 'lat_min': -15, 'lat_max': 10}

CB_LEVELS = [0, 10, 20.1, 50, 300]
CB_COLORS = ['#FFFFFF', '#87CEFA', '#000080', '#00FF00']

FILTER_URL    = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
NOMADS_CHECK  = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"

# ── Font ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Times New Roman'
try:
    font_path = os.path.join(current_dir, 'assets', 'times.ttf')
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        print(f"INFO: Custom font loaded: {font_path}")
    else:
        print("INFO: Using system font (times.ttf not found)")
except Exception as e:
    print(f"INFO: Using system font ({e})")

# =============================================================================
# 1. DIREKTORI & WAKTU
# =============================================================================

def create_date_directory():
    today    = datetime.now().strftime('%d%m%Y')
    date_dir = os.path.join(OUTPUT_DIR, today)
    os.makedirs(date_dir, exist_ok=True)
    print(f"INFO: Output directory: {date_dir}")
    return date_dir


def get_tmp_dir(out_dir):
    tmp = os.path.join(out_dir, 'tmp_grib')
    os.makedirs(tmp, exist_ok=True)
    return tmp


def get_initial_time():
    """
    Tentukan GFS run berdasarkan jam WIB saat ini.
    Return: (date YYYYMMDD, hour '00'/'06'/'18', timesteps)
    1 index = 3 jam.

    MODIFIKASI: menggunakan timezone-aware UTC agar tidak bergantung
    pada timezone sistem operasi.
    """
    now_utc  = datetime.now(timezone.utc).replace(tzinfo=None)
    now_wib  = now_utc + timedelta(hours=7)
    hour_wib = now_wib.hour

    if 7 <= hour_wib < 13:
        init_date = now_utc - timedelta(days=1)
        init_hour = '18'
        ts = {
            'H+1': {'start': 11, 'end': 18},
            'H+2': {'start': 19, 'end': 26},
            'H+3': {'start': 27, 'end': 34},
            'H+4': {'start': 35, 'end': 42},
            'H+5': {'start': 43, 'end': 50},
            'H+6': {'start': 51, 'end': 58},
            'H+7': {'start': 59, 'end': 66},
        }
    elif 13 <= hour_wib <= 20:
        init_date = now_utc
        init_hour = '00'
        ts = {
            'H+1': {'start': 9,  'end': 16},
            'H+2': {'start': 17, 'end': 24},
            'H+3': {'start': 25, 'end': 32},
            'H+4': {'start': 33, 'end': 40},
            'H+5': {'start': 41, 'end': 48},
            'H+6': {'start': 49, 'end': 56},
            'H+7': {'start': 57, 'end': 64},
        }
    else:
        init_date = now_utc
        init_hour = '06'
        ts = {
            'H+1': {'start': 7,  'end': 14},
            'H+2': {'start': 15, 'end': 22},
            'H+3': {'start': 23, 'end': 30},
            'H+4': {'start': 31, 'end': 38},
            'H+5': {'start': 39, 'end': 46},
            'H+6': {'start': 47, 'end': 54},
            'H+7': {'start': 55, 'end': 62},
        }
    return init_date.strftime('%Y%m%d'), init_hour, ts


# =============================================================================
# 1B. VALIDASI KETERSEDIAAN GFS RUN DI NOMADS
# =============================================================================

def check_gfs_availability(date, hour, timeout=20):
    """
    Validasi apakah GFS run (date, hour) sudah tersedia di NOMADS
    dengan mengecek URL direktori atmos/ terlebih dahulu.

    Strategi 3 tingkat:
      1. Cek direktori /gfs.{date}/{hour}/atmos/ → HTTP 200 = tersedia
      2. Jika gagal (timeout/error), coba HEAD request file acuan (f024)
      3. Jika masih gagal, tampilkan warning dan tanya pengguna

    Return: True jika tersedia, False jika tidak.
    """
    dir_url = f"{NOMADS_CHECK}gfs.{date}/{hour}/atmos/"
    print(f"\n  [CHECK] Validasi ketersediaan GFS {date} {hour}Z ...")
    print(f"          URL: {dir_url}")

    # ── Tingkat 1: cek direktori ──────────────────────────────────────
    try:
        r = requests.head(dir_url, timeout=timeout, allow_redirects=True)
        if r.status_code == 200:
            print(f"  [OK]    GFS {date} {hour}Z tersedia (HTTP 200)")
            return True
        elif r.status_code == 403:
            # NOMADS kadang return 403 tapi data tetap bisa diakses via filter
            print(f"  [WARN]  HTTP 403 pada direktori — mencoba file acuan...")
        else:
            print(f"  [WARN]  HTTP {r.status_code} pada direktori")
    except requests.exceptions.RequestException as e:
        print(f"  [WARN]  Gagal cek direktori: {str(e)[:60]}")

    # ── Tingkat 2: HEAD request file acuan f024 ───────────────────────
    params_check = {
        'dir':         f"/gfs.{date}/{hour}/atmos",
        'file':        f"gfs.t{hour}z.pgrb2.0p25.f024",
        'var_CPRAT':   'on',
        'lev_surface': 'on',
        'subregion':   '',
        'leftlon': 100, 'rightlon': 110,
        'toplat': 5,    'bottomlat': -5,
    }
    try:
        r2 = requests.get(FILTER_URL, params=params_check, timeout=timeout)
        if r2.status_code == 200 and r2.content[:4] == b'GRIB':
            print(f"  [OK]    GFS {date} {hour}Z tersedia (file acuan valid)")
            return True
        else:
            print(f"  [WARN]  File acuan tidak valid "
                  f"(HTTP {r2.status_code}, content: {r2.content[:20]})")
    except requests.exceptions.RequestException as e:
        print(f"  [WARN]  Gagal cek file acuan: {str(e)[:60]}")

    # ── Tingkat 3: warning interaktif ────────────────────────────────
    print(f"\n  [!] GFS {date} {hour}Z BELUM TERSEDIA atau tidak dapat diakses.")
    print(f"      Ini normal jika dijalankan terlalu awal.")
    print(f"      Kemungkinan jadwal ketersediaan:")
    avail = {
        '00': 'sekitar jam 13:00 WIB',
        '06': 'sekitar jam 19:00 WIB',
        '12': 'sekitar jam 01:00 WIB (keesokan hari)',
        '18': 'sekitar jam 07:00 WIB',
    }
    print(f"      GFS {hour}Z biasanya tersedia {avail.get(hour, 'tidak diketahui')}")

    # Non-interaktif (cron/server): lanjutkan dengan peringatan
    if not sys.stdin.isatty():
        print("  [AUTO] Mode non-interaktif: melanjutkan proses...")
        return True

    # Interaktif: tanya pengguna
    try:
        ans = input("\n  Lanjutkan quand meme? [y/N]: ").strip().lower()
        return ans == 'y'
    except (EOFError, KeyboardInterrupt):
        return False


# =============================================================================
# 2. DOWNLOAD — NOMADS GRIB FILTER
# =============================================================================

def download_one_fhour(date, hour, fhour_int, tmp_dir, retries=3, wait=5):
    """
    Download CPRAT untuk satu forecast hour via NOMADS GRIB filter.
    Return: path file .grb2 lokal, atau None jika gagal.
    """
    out_path = os.path.join(tmp_dir, f"cprat_f{fhour_int:03d}.grb2")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"  [CACHE] f{fhour_int:03d}")
        return out_path

    params = {
        'dir':         f"/gfs.{date}/{hour}/atmos",
        'file':        f"gfs.t{hour}z.pgrb2.0p25.f{fhour_int:03d}",
        'var_CPRAT':   'on',
        'lev_surface': 'on',
        'subregion':   '',
        'leftlon':     DOMAIN['lon_min'],
        'rightlon':    DOMAIN['lon_max'],
        'toplat':      DOMAIN['lat_max'],
        'bottomlat':   DOMAIN['lat_min'],
    }

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(FILTER_URL, params=params, timeout=120)
            if r.status_code == 200 and r.content[:4] == b'GRIB':
                with open(out_path, 'wb') as f:
                    f.write(r.content)
                print(f"  [OK]   f{fhour_int:03d}  {len(r.content)//1024} KB")
                return out_path
            else:
                print(f"  [WARN] f{fhour_int:03d} Attempt {attempt}: "
                      f"HTTP {r.status_code}")
        except requests.exceptions.RequestException as e:
            err_short = str(e).split('\n')[0][:80]
            print(f"  [WARN] f{fhour_int:03d} Attempt {attempt}: {err_short}")

        if attempt < retries:
            print(f"         Menunggu {wait}s sebelum retry...")
            time.sleep(wait)

    print(f"  [ERROR] Gagal download f{fhour_int:03d}")
    return None


def download_period(date, hour, start_step, end_step, tmp_dir):
    """
    Download semua file GRIB2 untuk satu periode 24 jam (inclusive).
    Return: list of (fhour_int, filepath)
    """
    n = end_step - start_step + 1
    print(f"\n  Download {n} file "
          f"(f{start_step*3:03d}–f{end_step*3:03d})...")

    files = []
    for step in range(start_step, end_step + 1):
        fhour = step * 3
        path  = download_one_fhour(date, hour, fhour, tmp_dir)
        if path:
            files.append((fhour, path))

    print(f"  Berhasil: {len(files)}/{n} file")
    return files


# =============================================================================
# 3. PARSER GRIB2 — PURE PYTHON
#    MODIFIKASI: deteksi template packing (Simple / JPEG2000 / PNG / lainnya)
# =============================================================================

def _scan_sections(raw):
    """Scan binary GRIB2, return dict {sec_num: byte_offset}."""
    if raw[:4] != b'GRIB':
        raise ValueError("Bukan file GRIB2 yang valid")
    sections = {}
    pos = 16  # lewati Section 0 (16 bytes fixed)
    while pos < len(raw) - 4:
        if raw[pos:pos+4] == b'7777':
            break
        sec_len = struct.unpack('>I', raw[pos:pos+4])[0]
        if sec_len == 0:
            break  # guard infinite loop
        sec_num = struct.unpack('>B', raw[pos+4:pos+5])[0]
        sections[sec_num] = pos
        pos += sec_len
    return sections


def _grib_latlon(raw_bytes):
    """
    Decode GRIB2 lat/lon integer (sign-magnitude, bukan two's complement).
    MSB = tanda, sisa 31 bit = nilai absolut × 1e6.
    """
    val  = struct.unpack('>I', raw_bytes)[0]
    sign = (val >> 31) & 1
    mag  = val & 0x7FFFFFFF
    return (-1)**sign * mag / 1e6


def _detect_packing_template(raw, s5):
    """
    Deteksi template Data Representation (Section 5).
    Return: integer template number (0=Simple, 3=JPEG2000, 41=PNG, dll)

    Referensi GRIB2:
      Template 0  → Grid Point Data – Simple Packing
      Template 2  → Grid Point Data – Complex Packing
      Template 3  → Grid Point Data – Complex Packing + Spatial Diff
      Template 40 → Grid Point Data – JPEG2000 Packing
      Template 41 → Grid Point Data – PNG Packing
    """
    template_num = struct.unpack('>H', raw[s5+9:s5+11])[0]
    template_names = {
        0:  'Simple Packing',
        2:  'Complex Packing',
        3:  'Complex Packing + Spatial Differencing',
        40: 'JPEG2000 Packing',
        41: 'PNG Packing',
        50: 'Spectral Simple Packing',
        51: 'Spectral Simple Packing (deprecated)',
    }
    name = template_names.get(template_num, f'Unknown (Template {template_num})')
    print(f"    [PACK] Detected packing template: {name}")
    return template_num


def _decode_simple_packing(raw, s5, s7, n_vals, bits):
    """
    Decode GRIB2 Simple Packing (Template 0).
    Formula: Y = (R + X * 2^E) / 10^D
    """
    R = struct.unpack('>f', raw[s5+11:s5+15])[0]
    E = struct.unpack('>h', raw[s5+15:s5+17])[0]
    D = struct.unpack('>h', raw[s5+17:s5+19])[0]

    buf      = np.frombuffer(raw[s7+5:], dtype=np.uint8)
    bits_arr = np.unpackbits(buf)
    total    = n_vals * bits

    if len(bits_arr) < total:
        bits_arr = np.concatenate(
            [bits_arr, np.zeros(total - len(bits_arr), dtype=np.uint8)])

    reshaped = bits_arr[:total].reshape(n_vals, bits)
    powers   = (2 ** np.arange(bits - 1, -1, -1)).astype(np.int64)
    X        = reshaped.astype(np.int64).dot(powers)

    return (R + X.astype(np.float64) * (2.0**E)) / (10.0**D)


def _decode_jpeg2000_packing(raw, s5, s7, n_vals):
    """
    Decode GRIB2 JPEG2000 Packing (Template 40).
    Membutuhkan library 'glymur' (pip install glymur) atau 'imageio'.
    Fallback: coba via numpy jika data ternyata raw float.
    """
    try:
        import glymur
        import io
        # Data JPEG2000 dimulai setelah 5 byte header Section 7
        jp2_bytes = raw[s7+5:]
        jp2_stream = io.BytesIO(jp2_bytes)
        jp2 = glymur.Jp2k(jp2_stream)
        packed = jp2[:]
        X = packed.flatten().astype(np.float64)
    except ImportError:
        print("    [WARN] glymur tidak tersedia untuk decode JPEG2000. "
              "Install: pip install glymur")
        return None
    except Exception as e:
        print(f"    [WARN] Gagal decode JPEG2000: {e}")
        return None

    # Scale factors sama dengan Simple Packing
    R = struct.unpack('>f', raw[s5+11:s5+15])[0]
    E = struct.unpack('>h', raw[s5+15:s5+17])[0]
    D = struct.unpack('>h', raw[s5+17:s5+19])[0]
    return (R + X * (2.0**E)) / (10.0**D)


def _decode_png_packing(raw, s5, s7, n_vals):
    """
    Decode GRIB2 PNG Packing (Template 41).
    Membutuhkan Pillow (pip install pillow) atau imageio.
    """
    try:
        from PIL import Image
        import io
        png_bytes  = raw[s7+5:]
        img        = Image.open(io.BytesIO(png_bytes))
        packed     = np.array(img).flatten().astype(np.float64)
    except ImportError:
        print("    [WARN] Pillow tidak tersedia untuk decode PNG packing. "
              "Install: pip install pillow")
        return None
    except Exception as e:
        print(f"    [WARN] Gagal decode PNG packing: {e}")
        return None

    R = struct.unpack('>f', raw[s5+11:s5+15])[0]
    E = struct.unpack('>h', raw[s5+15:s5+17])[0]
    D = struct.unpack('>h', raw[s5+17:s5+19])[0]
    return (R + packed[:n_vals] * (2.0**E)) / (10.0**D)


def _decode_data(raw, s5, s7, n_vals, template_num):
    """
    Dispatcher: pilih fungsi decode sesuai template packing.
    Return: numpy array nilai float, atau None jika tidak didukung.
    """
    if template_num == 0:
        bits = struct.unpack('>B', raw[s5+19:s5+20])[0]
        return _decode_simple_packing(raw, s5, s7, n_vals, bits)

    elif template_num in (2, 3):
        # Complex packing — fallback ke simple packing header
        # (hanya works jika kompleksitasnya minimal)
        print("    [WARN] Complex Packing (Template 2/3) — mencoba Simple Packing decode...")
        bits = struct.unpack('>B', raw[s5+19:s5+20])[0]
        return _decode_simple_packing(raw, s5, s7, n_vals, bits)

    elif template_num == 40:
        return _decode_jpeg2000_packing(raw, s5, s7, n_vals)

    elif template_num == 41:
        return _decode_png_packing(raw, s5, s7, n_vals)

    else:
        print(f"    [ERROR] Template packing {template_num} tidak didukung. "
              f"Pertimbangkan install cfgrib untuk template ini.")
        return None


def read_grib2_cprat(filepath):
    """
    Baca satu file GRIB2 → data CPRAT mm/3jam.
    MODIFIKASI: deteksi dan handle berbagai template packing.
    Return: (data_2d, lats_1d, lons_1d) atau (None, None, None)
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        sec = _scan_sections(raw)
        if not {3, 5, 7}.issubset(sec):
            print(f"    [WARN] Section tidak lengkap: {os.path.basename(filepath)}")
            return None, None, None

        s3, s5, s7 = sec[3], sec[5], sec[7]

        # Grid definition (Section 3)
        ni  = struct.unpack('>I', raw[s3+30:s3+34])[0]
        nj  = struct.unpack('>I', raw[s3+34:s3+38])[0]
        la1 = _grib_latlon(raw[s3+46:s3+50])
        lo1 = struct.unpack('>I', raw[s3+50:s3+54])[0] / 1e6
        la2 = _grib_latlon(raw[s3+55:s3+59])
        lo2 = struct.unpack('>I', raw[s3+59:s3+63])[0] / 1e6

        # Deteksi template packing
        template_num = _detect_packing_template(raw, s5)

        # Decode nilai data
        values = _decode_data(raw, s5, s7, ni * nj, template_num)
        if values is None:
            print(f"    [ERROR] Decode gagal: {os.path.basename(filepath)}")
            return None, None, None

        # kg/m²/s → mm/3jam
        data_mm = values * 3600.0 * 3.0
        data_mm = np.where(np.isnan(data_mm) | (data_mm < 0), 0.0, data_mm)
        data_2d = data_mm.reshape(nj, ni)

        lats_1d = np.linspace(la1, la2, nj)
        lons_1d = np.linspace(lo1, lo2, ni)
        lons_1d = np.where(lons_1d > 180, lons_1d - 360, lons_1d)

        # Pastikan lat ascending
        if lats_1d[0] > lats_1d[-1]:
            lats_1d = lats_1d[::-1]
            data_2d = data_2d[::-1, :]

        return data_2d, lats_1d, lons_1d

    except Exception as e:
        print(f"    [ERROR] Parse GRIB2 {os.path.basename(filepath)}: {e}")
        return None, None, None


def accumulate_precipitation(file_list):
    """Akumulasi presipitasi semua file dalam satu periode."""
    total = lats = lons = None
    for fhour, path in file_list:
        d, la, lo = read_grib2_cprat(path)
        if d is None:
            continue
        if total is None:
            total, lats, lons = d.copy(), la, lo
        elif d.shape == total.shape:
            total += d
        else:
            print(f"    [WARN] Shape mismatch f{fhour:03d}, skip")
    if total is not None:
        print(f"  Akumulasi: min={total.min():.2f}  max={total.max():.2f} mm/24jam")
    return total, lats, lons


# =============================================================================
# 4. KALKULASI HUJAN PER WILAYAH
#  
# =============================================================================

def calculate_rain_by_region(precip, lats_1d, lons_1d, provinces, sea_areas):
    """
    Hitung max hujan per provinsi dan area laut menggunakan gpd.sjoin().

    Return: (results dict, sorting_info dict)
    """
    # ── Bangun GeoDataFrame grid points ──────────────────────────────
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    points_gdf = gpd.GeoDataFrame(
        {'rain': precip.flatten()},
        geometry=gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()),
        crs='EPSG:4326'
    )

    results      = {'Provinsi': {}, 'Laut': {}}
    sorting_info = {'Provinsi': {}, 'Laut': {}}

    # ── Provinsi ──────────────────────────────────────────────────────
    print("  Menghitung wilayah Provinsi (sjoin)...")
    try:
        joined_prov = gpd.sjoin(
            points_gdf,
            provinces[['PROVINSI', 'KODE_PROV', 'geometry']],
            how='inner',
            predicate='within'
        )
        # Groupby → max rain per provinsi
        max_prov = (
            joined_prov[joined_prov['rain'] >= 0]
            .groupby('PROVINSI')['rain']
            .max()
        )
        for prov_name, rain_val in max_prov.items():
            results['Provinsi'][prov_name] = rain_val
        # Ambil kode provinsi untuk sorting
        kode_map = provinces.set_index('PROVINSI')['KODE_PROV'].to_dict()
        sorting_info['Provinsi'] = kode_map

    except Exception as e:
        print(f"  [WARN] sjoin Provinsi gagal: {e} — fallback ke .within()")
        for _, prov in tqdm(provinces.iterrows(), total=len(provinces)):
            if prov.geometry is None or prov.geometry.is_empty:
                continue
            try:
                mask = points_gdf.geometry.within(prov.geometry)
                pts  = points_gdf.loc[mask, 'rain']
                pts  = pts[(~np.isnan(pts)) & (pts >= 0)]
                if len(pts) > 0:
                    name = prov['PROVINSI']
                    results['Provinsi'][name]      = pts.max()
                    sorting_info['Provinsi'][name] = prov['KODE_PROV']
            except Exception as ex:
                print(f"  [WARN] {prov.get('PROVINSI','?')}: {ex}")

    # ── Laut ──────────────────────────────────────────────────────────
    print("  Menghitung area Laut (sjoin)...")
    try:
        joined_sea = gpd.sjoin(
            points_gdf,
            sea_areas[['Met_Area', 'ID', 'geometry']],
            how='inner',
            predicate='within'
        )
        max_sea = (
            joined_sea[joined_sea['rain'] >= 0]
            .groupby('Met_Area')['rain']
            .max()
        )
        for sea_name, rain_val in max_sea.items():
            results['Laut'][sea_name] = rain_val
        id_map = sea_areas.set_index('Met_Area')['ID'].to_dict()
        sorting_info['Laut'] = id_map

    except Exception as e:
        print(f"  [WARN] sjoin Laut gagal: {e} — fallback ke .within()")
        for _, sea in tqdm(sea_areas.iterrows(), total=len(sea_areas)):
            if sea.geometry is None or sea.geometry.is_empty:
                continue
            try:
                mask = points_gdf.geometry.within(sea.geometry)
                pts  = points_gdf.loc[mask, 'rain']
                pts  = pts[(~np.isnan(pts)) & (pts >= 0)]
                if len(pts) > 0:
                    name = sea['Met_Area']
                    results['Laut'][name]      = pts.max()
                    sorting_info['Laut'][name] = sea['ID']
            except Exception as ex:
                print(f"  [WARN] {sea.get('Met_Area','?')}: {ex}")

    print(f"  Selesai: {len(results['Provinsi'])} provinsi, "
          f"{len(results['Laut'])} area laut")
    return results, sorting_info


def classify_regions(results, sorting_info):
    """Klasifikasi ISOL / OCNL / FRQ berdasarkan threshold hujan."""
    cls = {
        'ISOL': {'Provinsi': [], 'Laut': []},
        'OCNL': {'Provinsi': [], 'Laut': []},
        'FRQ':  {'Provinsi': [], 'Laut': []},
    }

    for rtype in ('Provinsi', 'Laut'):
        for name, rain in results[rtype].items():
            if 10 <= rain <= 20:
                cls['ISOL'][rtype].append(name)
            elif rain > 20 and rain <= 50:
                cls['OCNL'][rtype].append(name)
            elif rain > 50:
                cls['FRQ'][rtype].append(name)

        for cat in cls:
            cls[cat][rtype] = sorted(
                cls[cat][rtype],
                key=lambda x: sorting_info[rtype].get(x, 999)
            )

    print(f"  ISOL: {len(cls['ISOL']['Provinsi'])}P/{len(cls['ISOL']['Laut'])}L | "
          f"OCNL: {len(cls['OCNL']['Provinsi'])}P/{len(cls['OCNL']['Laut'])}L | "
          f"FRQ: {len(cls['FRQ']['Provinsi'])}P/{len(cls['FRQ']['Laut'])}L")
    return cls


# =============================================================================
# 5. SIMPAN TXT PER HARI
# =============================================================================

def save_classification_txt(classification, forecast_day, start_time, output_dir):
    """Simpan hasil klasifikasi ke file .txt per hari."""
    stamp    = start_time.strftime('%d%m%Y')
    txt_path = os.path.join(output_dir, f"CB_CLASS_{stamp}_H{forecast_day}.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Klasifikasi Hujan CB 24 Jam — Prediksi H+{forecast_day}\n")
        f.write(f"Valid: {start_time.strftime('%d-%m-%Y')}\n")
        f.write("=" * 50 + "\n")
        for cat, areas in classification.items():
            f.write(f"\n{cat}:\n")
            if areas['Provinsi']:
                f.write("Provinsi: " + ", ".join(areas['Provinsi']) + "\n")
            if areas['Laut']:
                f.write("Laut: " + ", ".join(areas['Laut']) + "\n")

    print(f"  ✓ TXT: {txt_path}")
    return txt_path


def create_summary_report(forecasts, output_dir):
    """Laporan ringkasan gabungan H+1 s/d H+7 (tanpa duplikasi)."""
    combined  = {c: {'Provinsi': set(), 'Laut': set()} for c in ('ISOL','OCNL','FRQ')}
    sort_all  = {'Provinsi': {}, 'Laut': {}}

    for fc in forecasts:
        for cat, areas in fc['classification'].items():
            combined[cat]['Provinsi'].update(areas['Provinsi'])
            combined[cat]['Laut'].update(areas['Laut'])
        sort_all['Provinsi'].update(fc['sorting_info']['Provinsi'])
        sort_all['Laut'].update(fc['sorting_info']['Laut'])

    out = os.path.join(output_dir,
                       f"SUMMARY_CB_FORECAST_{datetime.now().strftime('%d%m%Y')}.txt")
    with open(out, 'w', encoding='utf-8') as f:
        f.write("RINGKASAN PREDIKSI CUMULONIMBUS (CB)\n")
        f.write(f"Dibuat: {datetime.now().strftime('%d-%m-%Y %H:%M')}\n")
        f.write(f"Periode: {forecasts[0]['valid_time'].strftime('%d-%m-%Y')} "
                f"s/d {forecasts[-1]['valid_time'].strftime('%d-%m-%Y')}\n")
        f.write("=" * 50 + "\n")
        for cat in ('FRQ', 'OCNL', 'ISOL'):
            f.write(f"\n{cat}:\n")
            if combined[cat]['Provinsi']:
                provs = sorted(combined[cat]['Provinsi'],
                               key=lambda x: sort_all['Provinsi'].get(x, 999))
                f.write("Provinsi: " + ", ".join(provs) + "\n")
            if combined[cat]['Laut']:
                seas = sorted(combined[cat]['Laut'],
                              key=lambda x: sort_all['Laut'].get(x, 999))
                f.write("Laut: " + ", ".join(seas) + "\n")

    print(f"  ✓ Ringkasan: {out}")



def generate_output_json(forecasts, output_dir):
    """
    Membuat file JSON dengan struktur untuk web aviation BMKG.
    - Field utama: title, slug, from, to, content (JSON string),
                   created_at, updated_at
    - content berisi per-hari (H+1..H+7) dan cover (gabungan)
    """
    if not forecasts:
        raise ValueError("Daftar forecasts kosong; tidak bisa membuat JSON.")

    # Pemetaan bulan dalam Bahasa Indonesia
    bulan_id = {
        1:  ('Januari',   'JANUARI'),
        2:  ('Februari',  'FEBRUARI'),
        3:  ('Maret',     'MARET'),
        4:  ('April',     'APRIL'),
        5:  ('Mei',       'MEI'),
        6:  ('Juni',      'JUNI'),
        7:  ('Juli',      'JULI'),
        8:  ('Agustus',   'AGUSTUS'),
        9:  ('September', 'SEPTEMBER'),
        10: ('Oktober',   'OKTOBER'),
        11: ('November',  'NOVEMBER'),
        12: ('Desember',  'DESEMBER'),
    }

    def format_tanggal_id(dt):
        nama_bulan, _ = bulan_id[dt.month]
        return f"{dt.day} {nama_bulan} {dt.year}"

    def format_key_harian(dt):
        _, nama_bulan_caps = bulan_id[dt.month]
        return f"{dt.day}_{nama_bulan_caps}_{dt.year}"

    def slugify(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        text = re.sub(r"\s+", "-", text).strip('-')
        return text

    # Susun rentang tanggal
    first_dt = forecasts[0]['valid_time']
    last_dt  = forecasts[-1]['valid_time']
    from_iso = first_dt.strftime('%Y-%m-%d')
    to_iso   = last_dt.strftime('%Y-%m-%d')

    from_id = format_tanggal_id(first_dt)
    to_id   = format_tanggal_id(last_dt)

    title = (f"POTENSI PERTUMBUHAN AWAN CB DI WILAYAH UDARA INDONESIA "
             f"BERLAKU {from_id.upper()} - {to_id.upper()}")
    slug  = f"{slugify(title)}-{int(datetime.now().timestamp())}"

    # Kumpulan gabungan untuk cover
    cover_ocnl_set = set()
    cover_frq_set  = set()

    # Bangun entri per hari
    per_hari = {}
    for forecast in forecasts:
        valid_dt  = forecast['valid_time']
        key       = format_key_harian(valid_dt)
        date_text = format_tanggal_id(valid_dt)

        classification = forecast['classification']
        ocnl_list = (classification.get('OCNL', {}).get('Provinsi', []) +
                     classification.get('OCNL', {}).get('Laut', []))
        frq_list  = (classification.get('FRQ', {}).get('Provinsi', []) +
                     classification.get('FRQ', {}).get('Laut', []))

        # Tambahkan ke cover set
        cover_ocnl_set.update(ocnl_list)
        cover_frq_set.update(frq_list)


        stamp          = valid_dt.strftime('%d%m%Y')
        image_filename = f"CB_PRED_{stamp}_H{forecast['day']}.jpg"
        image_path     = (f"https://web-aviation.bmkg.go.id/prakcb/"
                          f"{first_dt.strftime('%d%m%Y')}/{image_filename}")

        per_hari[key] = {
            "title": (f"POTENSI PERTUMBUHAN AWAN CB DI WILAYAH UDARA "
                      f"INDONESIA BERLAKU {date_text}"),
            "date":  date_text,
            "image": image_path,
            "ocnl":  ", ".join(ocnl_list) if ocnl_list else "-",
            "frq":   ", ".join(frq_list)  if frq_list  else "-",
        }


    gif_filename = f"CB_FORECAST_ANIMATION_{datetime.now().strftime('%d%m%Y')}.gif"
    gif_path     = (f"https://web-aviation.bmkg.go.id/prakcb/"
                    f"{first_dt.strftime('%d%m%Y')}/{gif_filename}")

    content_obj          = per_hari.copy()
    content_obj["cover"] = {
        "title": title,
        "date":  f"{from_id} - {to_id}",
        "image": gif_path,
        "ocnl":  ", ".join(sorted(cover_ocnl_set)) if cover_ocnl_set else "-",
        "frq":   ", ".join(sorted(cover_frq_set))  if cover_frq_set  else "-",
    }

  
    payload = {
        "title":      title,
        "slug":       slug,
        "from":       from_iso,
        "to":         to_iso,
        "content":    json.dumps(content_obj, ensure_ascii=False),
        "created_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
        "updated_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
    }

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(
        output_dir,
        f"prakiraancb_{datetime.now().strftime('%d%m%Y')}.json"
    )
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"  ✓ JSON: {outfile}")
    return outfile

def ping_callback():
    """
    Mengirim ping ke callback URL setelah semua task selesai
    """
    callback_url = "https://web-aviation.bmkg.go.id/web/predcb/callback"
    try:
        response = requests.get(callback_url, timeout=10)
        if response.status_code == 200:
            print(f"✓ Callback ping berhasil: {response.status_code}")
        else:
            print(f"⚠ Callback ping gagal: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Error mengirim callback ping: {str(e)}")


# =============================================================================
# 6. VISUALISASI
# =============================================================================

def _load_logo(logo_path):
    """
    [HELPER] Load logo BMKG. Return numpy array image atau None jika tidak ada.
    """
    if os.path.exists(logo_path):
        try:
            img = mpimg.imread(logo_path)
            print(f"  [OK]  Logo dimuat: {logo_path}")
            return img
        except Exception as e:
            print(f"  [WARN] Gagal load logo {logo_path}: {e}")
    else:
        print(f"  [WARN] Logo tidak ditemukan: {logo_path}")
    return None


def create_plot(precip, lats_1d, lons_1d, classification,
                forecast_day, start_time, init_hour,
                provinces, sea_areas):
    """
    Buat peta CB forecast lengkap dengan:
    - Kontur ISOL/OCNL/FRQ
    - Batas provinsi dan area laut
    - Gridlines
    - Legend
    - Info box kiri bawah:
        [LOGO BMKG] | Provided by BMKG
                       Cumulonimbus Cloud Forecast ...
        ─────────────────────────────────────────────
        Definisi ISOL/OCNL/FRQ
    """
    fig     = plt.figure(figsize=(15, 10))
    main_ax = plt.axes(projection=ccrs.PlateCarree())

    main_ax.set_extent(
        [DOMAIN['lon_min'], DOMAIN['lon_max'],
         DOMAIN['lat_min'], DOMAIN['lat_max']],
        crs=ccrs.PlateCarree()
    )
    main_ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
                        linewidth=1.2, color='black')

    # ── Kontur hujan ──────────────────────────────────────────────────
    main_ax.contourf(
        lons_1d, lats_1d, precip,
        levels=CB_LEVELS, colors=CB_COLORS,
        extend='max', transform=ccrs.PlateCarree()
    )

    # ── Batas wilayah ─────────────────────────────────────────────────
    
    if len(provinces) > 0:
        provinces.plot(ax=main_ax, facecolor='none',
                       edgecolor='black', linewidth=0.8,
                       transform=ccrs.PlateCarree())
    
    if len(sea_areas) > 0:
        sea_areas.plot(ax=main_ax, facecolor='none',
                         edgecolor='blue', linewidth=0.8, alpha=0.3,
                         transform=ccrs.PlateCarree())
    
    # ── Gridlines ─────────────────────────────────────────────────────
    gl = main_ax.gridlines(
        draw_labels=True, linewidth=0.5, color='gray',
        alpha=0.5, linestyle='--',
        xlocs=np.arange(90, 146, 4),
        ylocs=np.arange(-15, 11, 2)
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    # ── Legend CB ─────────────────────────────────────────────────────
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#87CEFA', label='ISOL (Isolated)'),
        plt.Rectangle((0,0),1,1, facecolor='#000080', label='OCNL (Occasional)'),
        plt.Rectangle((0,0),1,1, facecolor='#00FF00', label='FRQ (Frequent)'),
    ]
    main_ax.legend(handles=legend_elements, loc='upper right',
                   title='Cumulonimbus Cloud',
                   fontsize=9, title_fontsize=10)

    # ── Info box (kiri bawah) ─────────────────────────────────────────
    box_ax = main_ax.inset_axes([0.03, 0.02, 0.30, 0.21])
    box_ax.set_facecolor('white')
    box_ax.set_alpha(0.8)
    for spine in box_ax.spines.values():
        spine.set_visible(True)
    box_ax.set_xticks([])
    box_ax.set_yticks([])

    # ── LOGO BMKG ─────────────────────────────────────────────────────
    logo_img = _load_logo(LOGO_PATH)
    if logo_img is not None:
        logo_ax = box_ax.inset_axes([-0.15, 0.4, 0.5, 0.5])
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')

    # ── Teks info utama ───────────────────────────────────────────────
    main_text = (
        "Provided by Badan Meteorologi Klimatologi dan Geofisika\n"
        "Direktorat Meteorologi Penerbangan\n"
        "Cumulonimbus Cloud Forecast\n"
        "Based on GFS 0.25\n"
        f"Valid {start_time.strftime('%d-%m-%Y')}"
    )
    box_ax.text(0.5, 0.70, main_text,
                fontsize=8, va='center', ha='center', linespacing=1.3)

    # ── Garis pemisah ─────────────────────────────────────────────────
    box_ax.axhline(y=0.40, xmin=0.05, xmax=0.95,
                   color='black', linewidth=0.8)

    # ── Teks definisi CB ──────────────────────────────────────────────
    cb_text = (
        "ISOL CB means CB activity in less than 50% of the area\n"
        "OCNL CB means CB activity in 50-75% of the area\n"
        "FRQ CB means CB activity in more than 75% of the area"
    )
    box_ax.text(0.5, 0.18, cb_text,
                fontsize=7, va='center', ha='center', linespacing=1.3)

    plt.tight_layout()
    return fig



def save_plot(fig, forecast_day, start_time, output_dir):
    """Simpan figure ke file JPG."""
    stamp    = start_time.strftime('%d%m%Y')
    jpg_path = os.path.join(output_dir, f"CB_PRED_{stamp}_H{forecast_day}.jpg")
    fig.savefig(jpg_path, dpi=300, bbox_inches='tight',
                facecolor='white', format='jpeg')
    print(f"  ✓ JPG: {jpg_path}")
    return jpg_path


# =============================================================================
# 7. ANIMASI GIF — H+1 s/d H+7
# =============================================================================

def create_gif_animation(jpg_paths, output_dir, duration_ms=3000):
    """
    Buat animasi GIF dari list file JPG (H+1 s/d H+7).

    Parameter:
      jpg_paths   : list path JPG yang sudah terurut H+1 → H+7
      output_dir  : direktori output
      duration_ms : jeda antar frame dalam milidetik (default 3000 = 3 detik)

    Output:
      CB_FORECAST_ANIMATION_DDMMYYYY.gif  di output_dir

    Strategi library (berurutan, pakai yang tersedia):
      1. Pillow  → paling umum, ringan
      2. imageio → alternatif jika Pillow tidak ada
    """
    if not jpg_paths:
        print("  [WARN] Tidak ada file JPG untuk GIF — dilewati")
        return None


    stamp    = datetime.now().strftime('%d%m%Y')
    gif_path = os.path.join(output_dir, f"CB_FORECAST_ANIMATION_{stamp}.gif")

    print(f"\n  Membuat animasi GIF dari {len(jpg_paths)} frame "
          f"(interval {duration_ms} ms)...")

    try:
        from PIL import Image as PILImage

        frames = []
        for p in jpg_paths:
            if not os.path.exists(p):
                print(f"  [WARN] JPG tidak ditemukan, skip frame: {p}")
                continue
            img = PILImage.open(p).convert('RGB')
            frames.append(img)

        if not frames:
            print("  [ERROR] Tidak ada frame valid untuk GIF")
            return None

        frames[0].save(
            gif_path,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,   # ms per frame
            loop=0,                 # 0 = loop selamanya
            optimize=False,
        )
        print(f"  ✓ GIF (Pillow): {gif_path}")
        return gif_path

    except ImportError:
        print("  [WARN] Pillow tidak tersedia, mencoba imageio...")

    try:
        import imageio

        frames = []
        for p in jpg_paths:
            if not os.path.exists(p):
                print(f"  [WARN] JPG tidak ditemukan, skip frame: {p}")
                continue
            frames.append(imageio.imread(p))

        if not frames:
            print("  [ERROR] Tidak ada frame valid untuk GIF")
            return None

        imageio.mimsave(
            gif_path,
            frames,
            duration=duration_ms / 1000.0,
            loop=0
        )
        print(f"  ✓ GIF (imageio): {gif_path}")
        return gif_path

    except ImportError:
        print("  [ERROR] Pillow maupun imageio tidak tersedia.")
        print("          Install salah satu: pip install pillow  atau  pip install imageio")
        return None

    except Exception as e:
        print(f"  [ERROR] Gagal buat GIF: {e}")
        return None


# =============================================================================
# 8. MAIN 
# =============================================================================

def main():
    print(f"\n{'='*62}")
    print("  CB FORECAST — Download, Klasifikasi & Visualisasi")
    print("  Parser : Pure Python  |  Sumber : GFS pgrb2.0p25 NOMADS")
    print(f"{'='*62}\n")

    # ── Validasi shapefile ────────────────────────────────────────────
    missing = False
    for label, path in [("Provinsi", SHP_PROVINCES), ("Laut", SHP_SEA)]:
        if os.path.exists(path):
            print(f"  ✓ {label}: {path}")
        else:
            print(f"  ✗ {label} TIDAK DITEMUKAN: {path}")
            missing = True

    if missing:
        print("\n  [ERROR] Shapefile tidak lengkap. Proses dihentikan.")
        sys.exit(1)

    # ── Validasi logo ─────────────────────────────────────────────────
    if os.path.exists(LOGO_PATH):
        print(f"  ✓ Logo BMKG: {LOGO_PATH}")
    else:
        print(f"  [WARN] Logo BMKG tidak ditemukan: {LOGO_PATH} "
              f"(peta tetap dibuat tanpa logo)")

    # ── Muat shapefile ────────────────────────────────────────────────
    print("\n  Memuat shapefile...")
    provinces = gpd.read_file(SHP_PROVINCES)
    sea_areas = gpd.read_file(SHP_SEA)

    for gdf in (provinces, sea_areas):
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf.to_crs(epsg=4326, inplace=True)
        gdf.drop(index=gdf[gdf.geometry.isna()].index, inplace=True)

    print(f"  Provinsi: {len(provinces)}  |  Laut: {len(sea_areas)}")

    # ── Direktori ─────────────────────────────────────────────────────
    out_dir = create_date_directory()
    tmp_dir = get_tmp_dir(out_dir)
    print(f"  Tmp    : {tmp_dir}")

    # ── Tentukan GFS run ──────────────────────────────────────────────
    date, hour, timesteps = get_initial_time()
    init_dt = datetime.strptime(f"{date}{hour}", "%Y%m%d%H")
    print(f"\n  GFS Init : {date}  {hour}Z")
    print(f"  Forecast : H+1 s/d H+7")

    # ── Validasi ketersediaan GFS run di NOMADS ───────────────────────
    available = check_gfs_availability(date, hour)
    if not available:
        print("\n  [ABORT] GFS run tidak tersedia. Proses dihentikan.")
        sys.exit(0)

    print(f"\n  Memulai proses 7 hari...\n")

    forecasts = []
    jpg_paths = []   # kumpulkan path JPG untuk GIF

    for day in range(1, 8):
        key        = f'H+{day}'
        start_step = timesteps[key]['start']
        end_step   = timesteps[key]['end']
        valid_time = init_dt + timedelta(hours=start_step * 3)

        print(f"\n{'─'*62}")
        print(f"  H+{day}  |  Valid: {valid_time.strftime('%d-%m-%Y')}  "
              f"|  Steps {start_step}–{end_step}  "
              f"(f{start_step*3:03d}–f{end_step*3:03d})")
        print(f"{'─'*62}")

        # 1. Download
        print("\n  [1/5] Download GRIB2...")
        file_list = download_period(date, hour, start_step, end_step, tmp_dir)
        if not file_list:
            print(f"  [SKIP] Tidak ada data H+{day}")
            continue

        # 2. Baca & Akumulasi
        print("\n  [2/5] Parsing dan akumulasi presipitasi...")
        precip, lats, lons = accumulate_precipitation(file_list)
        if precip is None:
            print(f"  [SKIP] Gagal parse H+{day}")
            continue

        # 3. Kalkulasi per wilayah
        print("\n  [3/5] Kalkulasi hujan per wilayah (sjoin)...")
        results, sorting_info = calculate_rain_by_region(
            precip, lats, lons, provinces, sea_areas)
        classification = classify_regions(results, sorting_info)

        # 4. Simpan TXT
        print("\n  [4/5] Simpan klasifikasi TXT...")
        save_classification_txt(classification, day, valid_time, out_dir)

        # 5. Visualisasi & Simpan JPG
        print("\n  [5/5] Membuat dan menyimpan JPG...")
        fig = create_plot(precip, lats, lons, classification,
                          day, valid_time, hour,
                          provinces, sea_areas)
        jpg_file = save_plot(fig, day, valid_time, out_dir)
        jpg_paths.append(jpg_file)
        plt.close(fig)

        forecasts.append({
            'day':            day,
            'classification': classification,
            'sorting_info':   sorting_info,
            'valid_time':     valid_time,
        })
        print(f"\n  ✓ H+{day} selesai")

    # ── Summary, GIF, JSON & Cleanup ──────────────────────────────────
    print(f"\n{'─'*62}")
    if forecasts:
        print("  Membuat laporan ringkasan...")
        create_summary_report(forecasts, out_dir)

    # ── Animasi GIF ───────────────────────────────────────────────────
    if jpg_paths:
        print("  Membuat animasi GIF...")
        create_gif_animation(jpg_paths, out_dir, duration_ms=3000)

    #  Generate output JSON
    if forecasts:
        print("  Membuat output JSON...")
        try:
            generate_output_json(forecasts, out_dir)
        except Exception as e:
            print(f"  [ERROR] Gagal generate JSON: {e}")

    # ── Cleanup tmp ───────────────────────────────────────────────────
    try:
        import shutil
        shutil.rmtree(tmp_dir)
        print("  ✓ File GRIB2 temporary dibersihkan")
    except Exception as e:
        print(f"  [WARN] Cleanup: {e}")

    # ── Callback ping ─────────────────────────────────────────────────
    ping_callback()

    print(f"\n{'='*62}")
    print(f"  SELESAI — {len(forecasts)}/7 forecast berhasil")
    print(f"  Output  : {out_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()

