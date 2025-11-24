#!/bin/bash
# Script untuk menghapus file besar dari git history

echo "Menghapus file shapefile besar dari git history..."

# Hapus file dari staging area
git rm --cached datadasar/Provinsi.shp datadasar/Provinsi.shx datadasar/Provinsi.dbf datadasar/Provinsi.prj datadasar/Provinsi.sbn datadasar/Provinsi.sbx datadasar/Provinsi.cpg datadasar/Provinsi.shp.xml 2>/dev/null

# Hapus dari semua commit history menggunakan filter-branch
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch datadasar/Provinsi.shp datadasar/Provinsi.shx datadasar/Provinsi.dbf datadasar/Provinsi.prj datadasar/Provinsi.sbn datadasar/Provinsi.sbx datadasar/Provinsi.cpg datadasar/Provinsi.shp.xml' \
  --prune-empty --tag-name-filter cat -- --all

echo "Selesai! File sudah dihapus dari git history."
echo "Sekarang commit perubahan .gitignore dan push ulang:"
echo "  git add .gitignore"
echo "  git commit -m 'Remove large shapefile from git, add to .gitignore'"
echo "  git push origin main --force"

