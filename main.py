#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib
matplotlib.use('Agg')

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.font_manager as fm
import os
import imageio
import json
import re

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

current_dir = os.getcwd()
try:
   # Prefer bundled font in assets if available
   font_path = os.path.join(current_dir, 'assets', 'times.ttf')
   if os.path.exists(font_path):
       fm.fontManager.addfont(font_path)
       custom_font_name = fm.FontProperties(fname=font_path).get_name()
       plt.rcParams['font.family'] = custom_font_name
except Exception:
   # If loading custom font fails, keep default rcParams
   pass
def create_date_directory():
   """
   Membuat direktori berdasarkan tanggal hari ini
   Format: /home/wrf/CB/YYYYMMDD
   """
   #pwd di current directory
   base_dir = current_dir + "/CB"
   today = datetime.now().strftime('%d%m%Y')
   date_dir = os.path.join(base_dir, today)
   
   # Buat direktori base jika belum ada
   if not os.path.exists(base_dir):
       os.makedirs(base_dir)
   
   # Buat direktori tanggal jika belum ada
   if not os.path.exists(date_dir):
       os.makedirs(date_dir)
   
   return date_dir

def get_initial_time():
   current_time = datetime.now()
   current_time_utc = current_time - timedelta(hours=7)
   current_hour = current_time.hour

   timesteps = {
       'H+1': {'start': 0, 'end': 0},
       'H+2': {'start': 0, 'end': 0},
       'H+3': {'start': 0, 'end': 0},
       'H+4': {'start': 0, 'end': 0},
       'H+5': {'start': 0, 'end': 0},
       'H+6': {'start': 0, 'end': 0},
       'H+7': {'start': 0, 'end': 0}
   }

   if 7 <= current_hour < 13:
       init_date = current_time_utc - timedelta(days=1)
       init_hour = '18'
       timesteps = {
           'H+1': {'start': 11, 'end': 18},
           'H+2': {'start': 19, 'end': 26},
           'H+3': {'start': 27, 'end': 34},
           'H+4': {'start': 35, 'end': 42},
           'H+5': {'start': 43, 'end': 49},
           'H+6': {'start': 50, 'end': 57},
           'H+7': {'start': 58, 'end': 65}
       }
   elif 13 <= current_hour <= 20:
       init_date = current_time_utc
       init_hour = '00'
       timesteps = {
           'H+1': {'start': 9, 'end': 16},
           'H+2': {'start': 17, 'end': 24},
           'H+3': {'start': 25, 'end': 32},
           'H+4': {'start': 33, 'end': 40},
           'H+5': {'start': 41, 'end': 48},
           'H+6': {'start': 49, 'end': 56},
           'H+7': {'start': 57, 'end': 64}
       }
   else:
       init_date = current_time_utc
       init_hour = '06'
       timesteps = {
           'H+1': {'start': 7, 'end': 14},
           'H+2': {'start': 15, 'end': 22},
           'H+3': {'start': 23, 'end': 30},
           'H+4': {'start': 31, 'end': 38},
           'H+5': {'start': 39, 'end': 46},
           'H+6': {'start': 47, 'end': 54},
           'H+7': {'start': 55, 'end': 62}
       }
   return init_date.strftime('%Y%m%d'), init_hour, timesteps

def get_gfs_url(date, hour):
   base_url = "http://nomads.ncep.noaa.gov:80/dods/gfs_0p25"
   return f"{base_url}/gfs{date}/gfs_0p25_{hour}z"

def calculate_rain_by_region(ds, provinces, sea_areas, precip):
   """
   Menghitung nilai hujan untuk setiap provinsi dan area laut menggunakan spatial indexing
   dan menyimpan informasi kode provinsi/ID laut untuk pengurutan
   """
   # Membuat grid points
   lon_grid, lat_grid = np.meshgrid(ds.lon, ds.lat)
   points_coords = list(zip(lon_grid.flatten(), lat_grid.flatten()))
   
   # Buat DataFrame
   points_df = pd.DataFrame({
       'longitude': [p[0] for p in points_coords],
       'latitude': [p[1] for p in points_coords],
       'rain': precip.flatten()
   })
   
   # Buat GeoDataFrame dengan CRS WGS84
   points_gdf = gpd.GeoDataFrame(
       points_df,
       geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude)
   )
   try:
       points_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
   except Exception:
       # Fallback untuk versi GeoPandas lama
       points_gdf.crs = "EPSG:4326"
   
   results = {
       'Provinsi': {},
       'Laut': {}
   }
   
   # Menyimpan informasi pengurutan
   sorting_info = {
       'Provinsi': {},  # Format: {nama_provinsi: kode_prov}
       'Laut': {}       # Format: {nama_laut: id}
   }
   
   # Process provinces
   print("Memproses data provinsi...")
   for _, province in tqdm(provinces.iterrows(), total=len(provinces)):
       # Lewati jika geometri tidak valid/None
       if province.geometry is None or getattr(province.geometry, "is_empty", False):
           continue
       # Filter points dalam provinsi
       mask = points_gdf.geometry.within(province.geometry)
       valid_points = points_gdf[mask]
       
       if not valid_points.empty:
           rain_values = valid_points.rain[~np.isnan(valid_points.rain) & (valid_points.rain >= 0)]
           if len(rain_values) > 0:
               results['Provinsi'][province['PROVINSI']] = rain_values.max()
               # Simpan kode provinsi untuk pengurutan
               sorting_info['Provinsi'][province['PROVINSI']] = province['KODE_PROV']
   
   # Process sea areas
   print("Memproses data laut...")
   for _, sea in tqdm(sea_areas.iterrows(), total=len(sea_areas)):
       # Lewati jika geometri tidak valid/None
       if sea.geometry is None or getattr(sea.geometry, "is_empty", False):
           continue
       # Filter points dalam area laut
       mask = points_gdf.geometry.within(sea.geometry)
       valid_points = points_gdf[mask]
       
       if not valid_points.empty:
           rain_values = valid_points.rain[~np.isnan(valid_points.rain) & (valid_points.rain >= 0)]
           if len(rain_values) > 0:
               results['Laut'][sea['Met_Area']] = rain_values.max()
               # Simpan ID laut untuk pengurutan
               sorting_info['Laut'][sea['Met_Area']] = sea['ID']
   
   return results, sorting_info

def classify_regions(results, sorting_info):
   """
   Mengklasifikasikan wilayah berdasarkan nilai hujan
   """
   classification = {
       'ISOL': {'Provinsi': [], 'Laut': []},
       'OCNL': {'Provinsi': [], 'Laut': []},
       'FRQ': {'Provinsi': [], 'Laut': []}
   }
   
   # Klasifikasi provinsi
   for province, rain_value in results['Provinsi'].items():
       if 10 <= rain_value <= 20:
           classification['ISOL']['Provinsi'].append(province)
       elif 20.1 <= rain_value <= 50:
           classification['OCNL']['Provinsi'].append(province)
       elif rain_value > 50:
           classification['FRQ']['Provinsi'].append(province)
   
   # Klasifikasi laut
   for sea, rain_value in results['Laut'].items():
       if 10 <= rain_value <= 20:
           classification['ISOL']['Laut'].append(sea)
       elif 20.1 <= rain_value <= 50:
           classification['OCNL']['Laut'].append(sea)
       elif rain_value > 50:
           classification['FRQ']['Laut'].append(sea)
   
   # Urutkan hasil klasifikasi berdasarkan KODE_PROV dan ID
   for class_type in classification:
       # Urutkan provinsi berdasarkan KODE_PROV
       classification[class_type]['Provinsi'] = sorted(
           classification[class_type]['Provinsi'],
           key=lambda x: sorting_info['Provinsi'].get(x, 999)  # 999 sebagai nilai default jika tidak ditemukan
       )
       
       # Urutkan laut berdasarkan ID
       classification[class_type]['Laut'] = sorted(
           classification[class_type]['Laut'],
           key=lambda x: sorting_info['Laut'].get(x, 999)  # 999 sebagai nilai default jika tidak ditemukan
       )
   
   return classification

def save_results(fig, classification, date, hour, forecast_day, start_time):
   """
   Menyimpan plot sebagai JPG dan klasifikasi sebagai TXT dengan start_time
   """
   # Buat direktori output berdasarkan tanggal
   output_dir = create_date_directory()
   
   # Format timestamp menggunakan start_time
   timestamp = start_time.strftime('%d%m%Y')
   
   # Simpan plot
   plot_filename = f"{output_dir}/CB_PRED_{timestamp}_H{forecast_day}.jpg"
   fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
   print(f"Plot saved as: {plot_filename}")
   
   # Simpan klasifikasi
   txt_filename = f"{output_dir}/CB_CLASS_{timestamp}_H{forecast_day}.txt"
   with open(txt_filename, 'w') as f:
       f.write(f"Klasifikasi Hujan 24 Jam - Prediksi H+{forecast_day}\n")
       f.write(f"Valid: {start_time.strftime('%d-%m-%Y')} \n")
       f.write("="*50 + "\n\n")
       
       for class_type, areas in classification.items():
           f.write(f"\n{class_type}:\n")
           if areas['Provinsi']:
               f.write("Provinsi: " + ", ".join(areas['Provinsi']) + "\n")
           if areas['Laut']:
               f.write("Laut: " + ", ".join(areas['Laut']) + "\n")
   
   print(f"Classification saved as: {txt_filename}")

def create_accumulated_rain_plot_with_provinces_and_sea(ds, shp_provinces, shp_sea, forecast_day, timesteps, start_time=None):
   """
   Membuat plot hujan akumulasi untuk periode tertentu dengan kriteria dan legenda yang diperbarui
   """
   fig = plt.figure(figsize=(15, 10))
   
   main_ax = plt.axes(projection=ccrs.PlateCarree())
   
   provinces = gpd.read_file(shp_provinces)
   sea_areas = gpd.read_file(shp_sea)

   # Pastikan CRS konsisten (EPSG:4326) dan geometri valid
   try:
       if provinces.crs and provinces.crs.to_string().upper() not in ["EPSG:4326", "WGS84", "OGC:CRS84"]:
           provinces = provinces.to_crs(epsg=4326)
   except Exception:
       pass
   try:
       if sea_areas.crs and sea_areas.crs.to_string().upper() not in ["EPSG:4326", "WGS84", "OGC:CRS84"]:
           sea_areas = sea_areas.to_crs(epsg=4326)
   except Exception:
       pass

   # Buang geometri None/empty/invalid
   provinces = provinces[provinces.geometry.notnull()]
   sea_areas = sea_areas[sea_areas.geometry.notnull()]
   provinces = provinces[~provinces.geometry.is_empty]
   sea_areas = sea_areas[~sea_areas.geometry.is_empty]
   
   main_ax.set_extent([90, 145, -15, 10], crs=ccrs.PlateCarree())
   main_ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=1.2)
   
   # Gunakan timesteps yang sesuai
   forecast_key = f'H+{forecast_day}'
   start_step = timesteps[forecast_key]['start']
   end_step = timesteps[forecast_key]['end']
   
   # Hitung presipitasi
   precip = ds['cpratsfc'].isel(time=slice(start_step, end_step)).sum(dim='time').values*3600*3
   
   # Hitung waktu valid jika tidak disediakan
   if start_time is None:
       model_time = pd.to_datetime(ds.time.values[0])
       start_time = model_time + timedelta(hours=start_step * 3)
   end_time = start_time + timedelta(hours=24)
   
   results, sorting_info = calculate_rain_by_region(ds, provinces, sea_areas, precip)
   classification = classify_regions(results, sorting_info)
   
   # Setup plot
   levels = [0, 10, 20.1, 50, 300]
   colors = ['#FFFFFF', '#87CEFA', '#000080', '#00FF00']
   
   cs = main_ax.contourf(ds.lon, ds.lat, precip, levels=levels, 
                        colors=colors, extend='max',
                        transform=ccrs.PlateCarree())
   
   if len(provinces) > 0:
       provinces.plot(ax=main_ax, facecolor='none', edgecolor='black', linewidth=0.8)
   if len(sea_areas) > 0:
       sea_areas.plot(ax=main_ax, facecolor='none', edgecolor='blue', linewidth=0.8, alpha=0)
   
   # Add gridlines
   gl = main_ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5,
                         linestyle='--', xlocs=np.arange(90, 146, 4), 
                         ylocs=np.arange(-15, 11, 2))
   gl.top_labels = False
   gl.right_labels = False
   gl.xlabel_style = {'size': 8}  # Ukuran font label bujur
   gl.ylabel_style = {'size': 8}  # Ukuran font label lintang
   
   # Add Legend
   legend_elements = [
       plt.Rectangle((0, 0), 1, 1, facecolor='#87CEFA', label='ISOL (Isolated)'),
       plt.Rectangle((0, 0), 1, 1, facecolor='#000080', label='OCNL (Occasional)'),
       plt.Rectangle((0, 0), 1, 1, facecolor='#00FF00', label='FRQ (Frequent)')
   ]
   main_ax.legend(handles=legend_elements, loc='upper right', 
                 title='Cumulonimbus Cloud',
                 fontsize=9, title_fontsize=10)
   
   # Add information box
   box_pos = [0.03, 0.02, 0.30, 0.21]
   box_ax = main_ax.inset_axes(box_pos)
   box_ax.set_facecolor('white')
   box_ax.set_alpha(0.8)
   box_ax.spines['top'].set_visible(True)
   box_ax.spines['right'].set_visible(True)
   box_ax.spines['bottom'].set_visible(True)
   box_ax.spines['left'].set_visible(True)
   box_ax.set_xticks([])
   box_ax.set_yticks([])
   
   # Add logo
   logo_ax = box_ax.inset_axes([-0.15, 0.4, 0.5, 0.5])
   logo = plt.imread( current_dir + '/assets/BMKG.png')
   logo_ax.imshow(logo)
   logo_ax.axis('off')
   
   # Add text information
   main_text = (
       "Provided by Badan Meteorologi Klimatologi dan Geofisika\n"
       "Cumulonimbus Cloud Forecast\n"
       "Based on GFS 0.25\n"
       f"Valid {start_time.strftime('%d-%m-%Y')} "
   )
   
   box_ax.text(0.5, 0.7, main_text,
               fontsize=8,
               verticalalignment='center',
               horizontalalignment='center',
               linespacing=1.3)
   
   box_ax.axhline(y=0.40, xmin=0.05, xmax=0.95, color='black', linewidth=0.8)
   
   cb_text = (
       "ISOL CB means CB activity in less than 50% of the area\n"
       "OCNL CB means CB activity in 50-75% of the area\n"
       "FRQ CB means CB activity in more than 75% of the area"
   )
   
   box_ax.text(0.5, 0.2, cb_text,
               fontsize=7,
               verticalalignment='center',
               horizontalalignment='center',
               linespacing=1.3)
   
   return fig, classification, sorting_info

def create_summary_report(forecasts, output_dir=None):
   """
   Membuat laporan ringkasan prediksi yang merupakan gabungan dari H+1 sampai H+7
   dengan menghindari duplikasi lokasi
   """
   if output_dir is None:
       output_dir = create_date_directory()
       
   summary_filename = os.path.join(output_dir, f"SUMMARY_CB_FORECAST_{datetime.now().strftime('%d%m%Y')}.txt")
   
   # Dictionary untuk menyimpan klasifikasi gabungan
   combined_classification = {
       'ISOL': {'Provinsi': set(), 'Laut': set()},
       'OCNL': {'Provinsi': set(), 'Laut': set()},
       'FRQ': {'Provinsi': set(), 'Laut': set()}
   }
   
   # Gabungkan semua informasi pengurutan
   combined_sorting_info = {
       'Provinsi': {},
       'Laut': {}
   }
   
   # Gabungkan semua klasifikasi dan informasi pengurutan
   for forecast in forecasts:
       classification = forecast['classification']
       sorting_info = forecast['sorting_info']
       
       # Update klasifikasi
       for class_type, areas in classification.items():
           combined_classification[class_type]['Provinsi'].update(areas['Provinsi'])
           combined_classification[class_type]['Laut'].update(areas['Laut'])
       
       # Update informasi pengurutan
       combined_sorting_info['Provinsi'].update(sorting_info['Provinsi'])
       combined_sorting_info['Laut'].update(sorting_info['Laut'])
   
   # Urutkan kategori klasifikasi berdasarkan intensitas (FRQ, OCNL, ISOL)
   ordered_classes = ['FRQ', 'OCNL', 'ISOL']
   
   # Tulis laporan
   with open(summary_filename, 'w') as f:
       f.write("RINGKASAN PREDIKSI CUMULONIMBUS (CB)\n")
       f.write(f"Dibuat pada: {datetime.now().strftime('%d-%m-%Y %H:%M')}\n")
       f.write(f"Periode: {forecasts[0]['valid_time'].strftime('%d-%m-%Y')} s/d {forecasts[-1]['valid_time'].strftime('%d-%m-%Y')}\n")
       f.write("="*50 + "\n\n")
       
       # Tulis hasil klasifikasi dalam urutan yang ditentukan
       for class_type in ordered_classes:
           f.write(f"\n{class_type}:\n")
           
           if combined_classification[class_type]['Provinsi']:
               # Urutkan provinsi berdasarkan KODE_PROV
               provinces = sorted(
                   list(combined_classification[class_type]['Provinsi']), 
                   key=lambda x: combined_sorting_info['Provinsi'].get(x, 999)
               )
               f.write("Provinsi: " + ", ".join(provinces) + "\n")
           
           if combined_classification[class_type]['Laut']:
               # Urutkan laut berdasarkan ID
               seas = sorted(
                   list(combined_classification[class_type]['Laut']), 
                   key=lambda x: combined_sorting_info['Laut'].get(x, 999)
               )
               f.write("Laut: " + ", ".join(seas) + "\n")
   
   print(f"Laporan ringkasan disimpan di: {summary_filename}")
   return summary_filename

def create_forecast_gif(output_dir=None):
   """
   Membuat animasi GIF dari plot-plot forecast
   """
   if output_dir is None:
       output_dir = create_date_directory()
       
   # Cari semua file plot yang sesuai
   plot_files = sorted([
       f for f in os.listdir(output_dir) 
       if f.startswith('CB_PRED_') and f.endswith('.jpg')
   ])
   
   # Baca gambar-gambar tersebut
   images = []
   for filename in plot_files:
       filepath = os.path.join(output_dir, filename)
       images.append(imageio.imread(filepath))
   
   # Buat GIF
   gif_filename = os.path.join(output_dir, f"CB_FORECAST_ANIMATION_{datetime.now().strftime('%d%m%Y')}.gif")
   imageio.mimsave(gif_filename, images, duration=7, loop=0)  # 7 detik per frame, loop infinitely
   
   print(f"Animasi forecast disimpan di: {gif_filename}")
   return gif_filename

def generate_output_json(forecasts, output_dir=None):
   """
   Membuat file JSON dengan struktur seperti assets/dummy.json
   - Field utama: title, slug, from, to, content (JSON string), created_at, updated_at
   - content berisi per-hari (H+1..H+7) dan cover (gabungan)
   """
   if output_dir is None:
       output_dir = create_date_directory()

   # Pemetaan bulan dalam Bahasa Indonesia
   bulan_id = {
       1: ('Januari', 'JANUARI'), 2: ('Februari', 'FEBRUARI'), 3: ('Maret', 'MARET'),
       4: ('April', 'APRIL'), 5: ('Mei', 'MEI'), 6: ('Juni', 'JUNI'),
       7: ('Juli', 'JULI'), 8: ('Agustus', 'AGUSTUS'), 9: ('September', 'SEPTEMBER'),
       10: ('Oktober', 'OKTOBER'), 11: ('November', 'NOVEMBER'), 12: ('Desember', 'DESEMBER')
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

   if not forecasts:
       raise ValueError("Daftar forecasts kosong; tidak bisa membuat JSON.")

   # Susun rentang tanggal
   first_dt = forecasts[0]['valid_time']
   last_dt = forecasts[-1]['valid_time']
   from_iso = first_dt.strftime('%Y-%m-%d')
   to_iso = last_dt.strftime('%Y-%m-%d')

   from_id = format_tanggal_id(first_dt)
   to_id = format_tanggal_id(last_dt)

   title = f"POTENSI PERTUMBUHAN AWAN CB DI WILAYAH UDARA INDONESIA BERLAKU {from_id.upper()} - {to_id.upper()}"
   slug = f"{slugify(title)}-{int(datetime.now().timestamp())}"

   # Kumpulan gabungan untuk cover
   cover_ocnl_set = set()
   cover_frq_set = set()

   # Bangun entri per hari
   per_hari = {}
   for forecast in forecasts:
       valid_dt = forecast['valid_time']
       key = format_key_harian(valid_dt)
       date_text = format_tanggal_id(valid_dt)

       # Klasifikasi per hari
       classification = forecast['classification']
       ocnl_list = classification.get('OCNL', {}).get('Provinsi', []) + classification.get('OCNL', {}).get('Laut', [])
       frq_list = classification.get('FRQ', {}).get('Provinsi', []) + classification.get('FRQ', {}).get('Laut', [])

       # Tambahkan ke cover set
       cover_ocnl_set.update(ocnl_list)
       cover_frq_set.update(frq_list)

       # Tentukan nama file gambar sesuai yang dihasilkan save_results
       stamp = valid_dt.strftime('%d%m%Y')
       image_filename = f"CB_PRED_{stamp}_H{forecast['day']}.jpg"
    #    image_path = os.path.join(output_dir, image_filename)
       #date format DDMMYYYY today
       image_path = f"https://web-aviation.bmkg.go.id/prakcb/{first_dt.strftime('%d%m%Y')}/{image_filename}"
       per_hari[key] = {
           "title": f"POTENSI PERTUMBUHAN AWAN CB DI WILAYAH UDARA INDONESIA BERLAKU {date_text}",
           "date": date_text,
           "image": image_path,
           "ocnl": ", ".join(ocnl_list) if ocnl_list else "-",
           "frq": ", ".join(frq_list) if frq_list else "-"
       }

   # GIF cover
   gif_filename = f"CB_FORECAST_ANIMATION_{datetime.now().strftime('%d%m%Y')}.gif"
#    gif_path = os.path.join(output_dir, gif_filename)
   gif_path = f"https://web-aviation.bmkg.go.id/prakcb/{first_dt.strftime('%d%m%Y')}/{gif_filename}"
   content_obj = per_hari.copy()
   content_obj["cover"] = {
       "title": title,
       "date": f"{from_id} - {to_id}",
       "image": gif_path,
       "ocnl": ", ".join(sorted(cover_ocnl_set)) if cover_ocnl_set else "-",
       "frq": ", ".join(sorted(cover_frq_set)) if cover_frq_set else "-"
   }

   # Bungkus sesuai struktur dummy.json
   payload = {
       "title": title,
       "slug": slug,
       "from": from_iso,
       "to": to_iso,
       "content": json.dumps(content_obj, ensure_ascii=False),
       "created_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z'),
       "updated_at": datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z')
   }

   # Tulis file ke direktori tanggal (sama dengan lokasi JPG/TXT)
   if not os.path.exists(output_dir):
       os.makedirs(output_dir)
   outfile = os.path.join(output_dir, f"prakiraancb_{datetime.now().strftime('%d%m%Y')}.json")
   with open(outfile, 'w', encoding='utf-8') as f:
       json.dump(payload, f, ensure_ascii=False, indent=1)

   return outfile

def create_flexible_forecast(date, hour, forecast_days, timesteps, shp_provinces, shp_sea):
   """
   Modifikasi fungsi existing untuk menggunakan direktori tanggal
   """
   url = get_gfs_url(date, hour)
   forecasts = []
   output_dir = create_date_directory()
   
   try:
       ds = xr.open_dataset(url)
       ds = ds.sel(lon=slice(90, 145), lat=slice(-15, 10))
       
       # Hitung start_time valid berbasis init date/hour dan timesteps (bukan dari ds.time)
       init_datetime_utc = datetime.strptime(f"{date}{hour}", "%Y%m%d%H")
       start_times = []
       for day in range(1, forecast_days + 1):
           forecast_key = f'H+{day}'
           start_step = timesteps[forecast_key]['start']
           start_time = init_datetime_utc + timedelta(hours=start_step * 3)
           start_times.append(start_time)
       
       print(f"\nMembuat forecast dari data GFS tanggal {date} jam {hour}Z")
       print(f"Periode prediksi: {forecast_days} hari")
       for day, start_time in enumerate(start_times, 1):
           print(f"H+{day} valid: {start_time.strftime('%d-%m-%Y')} UTC")
       
       print(f"\nMengunduh data GFS untuk tanggal {date} jam {hour}Z...")
       
   except Exception as e:
       print(f"Error mengunduh data: {str(e)}")
       return None
   
   for day in range(1, forecast_days + 1):
       print(f"\nMembuat prediksi untuk H+{day} valid {start_times[day-1].strftime('%d-%m-%Y')}...")
       try:
           fig, classification, sorting_info = create_accumulated_rain_plot_with_provinces_and_sea(
               ds, shp_provinces, shp_sea, 
               forecast_day=day, 
               timesteps=timesteps,
               start_time=start_times[day-1]
           )
           
           # Simpan hasil
           save_results(fig, classification, date, hour, day, start_times[day-1])
           
           forecasts.append({
               'day': day,
               'figure': fig,
               'classification': classification,
               'sorting_info': sorting_info,
               'valid_time': start_times[day-1]
           })
           
           plt.close(fig)  # Tutup figure untuk menghemat memori
           
       except Exception as e:
           print(f"Error membuat prediksi H+{day}: {str(e)}")
           continue
   
   # Tambahkan laporan ringkasan
   create_summary_report(forecasts, output_dir)
   
   # Tambahkan animasi GIF
   create_forecast_gif(output_dir)
   
   # Buat file JSON hasil yang mengikuti struktur assets/dummy.json
   json_path = generate_output_json(forecasts, output_dir)
   print(f"File JSON disimpan di: {json_path}")
   
   return forecasts

if __name__ == "__main__":
   shp_provinces = current_dir + "/datadasar/Provinsi.shp"
   shp_sea = current_dir + "/datadasar/Laut.shp"
   
   # Dapatkan waktu inisial dan timesteps
   selected_date, selected_hour, timesteps = get_initial_time()
   selected_days = 7
   
   print(f"\nMenggunakan data GFS tanggal {selected_date} jam {selected_hour}Z di folder {current_dir} dan shp di folder {current_dir}/datadasar ")
   print(f"Periode prediksi: {selected_days} hari")
   
   forecasts = create_flexible_forecast(selected_date, selected_hour, selected_days,
                                     timesteps, shp_provinces, shp_sea)
   
   if not forecasts:
       print("Gagal membuat forecast. Silakan coba lagi nanti.")

