# Gunakan micromamba supaya proses dependency solving/install lebih cepat
FROM mambaorg/micromamba:latest

WORKDIR /app

# Pastikan output print Python langsung masuk ke log saat stdout di-redirect
ENV PYTHONUNBUFFERED=1

# Salin environment lebih dulu agar layer dependency bisa di-cache
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Buat environment Conda/Mamba dari file .yml
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Tambahkan decoder GRIB resmi di layer terpisah supaya perubahan ini tidak
# memaksa solve ulang seluruh environment utama saat cache masih tersedia.
RUN micromamba install -y -n geo_env -c conda-forge cfgrib eccodes && \
    micromamba clean --all --yes

# Salin semua file project ke /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Jalankan skrip di dalam environment geo_env
CMD ["micromamba", "run", "-n", "geo_env", "python", "-u", "main.py"]
