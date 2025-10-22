# 1. Gunakan base image resmi Miniconda
FROM continuumio/miniconda3

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Salin HANYA file environment.yml terlebih dahulu
# Ini memanfaatkan cache Docker. Environment tidak akan di-build ulang
# jika Anda hanya mengubah kode .py Anda.
COPY environment.yml .

# 4. Buat environment Conda dari file .yml
# Ini adalah langkah yang paling lama
RUN conda env create -f environment.yml

# 5. Salin semua file project Anda (cb.py, shapefiles, dll.) ke /app
COPY . .

# 6. Perintah default untuk menjalankan skrip Anda
# Ini menggunakan "conda run" untuk memastikan skrip berjalan
# DI DALAM environment 'geo_env' yang sudah kita buat
CMD ["conda", "run", "-n", "geo_env", "python", "main.py"]