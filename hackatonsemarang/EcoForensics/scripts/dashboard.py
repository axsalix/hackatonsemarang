import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

# --- CONFIG ---
st.set_page_config(page_title="Eco-Forensics", layout="wide", page_icon="🛡️")

# --- SESSION STATE (Memori Agar Data Tidak Hilang) ---
if 'forensic_data' not in st.session_state:
    st.session_state['forensic_data'] = None

# --- HEADER ---
st.title("🛡️ Eco-Forensics: Audit & Early Warning System")
st.markdown("""
**Platform Deep Tech berbasis AI & SAR Satellite untuk Mitigasi Krisis Hidrologi.**
_Sesuai Proposal Tim Trio Borma - ITB_
""")

# --- SIDEBAR (INPUT) ---
with st.sidebar:
    st.header("📍 Lokasi Audit")
    
    # Koordinat Default (Semarang/Demak)
    lat_input = st.number_input("Latitude", value=-6.8946, format="%.4f")
    lon_input = st.number_input("Longitude", value=110.6401, format="%.4f")
    
    st.info("💡 Tips: Gunakan Google Maps untuk cari koordinat daerah rawan banjir.")
    
    # Tombol Pemicu
    analyze_btn = st.button("🚀 JALANKAN ANALISIS FORENSIK", type="primary")

# --- LOGIKA ANALISIS (Hanya jalan saat tombol ditekan) ---
if analyze_btn:
    with st.spinner('🛰️ Menghubungi Satelit Sentinel-1, Data Hujan & Memproses AI...'):
        try:
            # 1. Panggil API Backend
            API_URL = f"http://127.0.0.1:8000/analyze?lat={lat_input}&lon={lon_input}"
            response = requests.get(API_URL)
            
            if response.status_code == 200:
                # 2. SIMPAN HASIL KE SESSION STATE (KUNCI AGAR TIDAK HILANG)
                st.session_state['forensic_data'] = response.json()
                st.success("✅ Analisis Selesai! Data tersimpan.")
            else:
                st.error(f"❌ Terjadi Kesalahan API: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Gagal terkoneksi ke Backend: {e}")
            st.warning("Pastikan Anda sudah menjalankan 'python scripts/api.py' di terminal lain!")

# --- TAMPILAN DASHBOARD (Mengambil data dari Session State) ---
data = st.session_state['forensic_data']

if data is not None:
    # Jika data sudah ada di memori, tampilkan!
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Peta Lokasi & Visualisasi")
        
        # Tampilkan Peta Interaktif (Folium)
        m = folium.Map(location=[data['lat'], data['lon']], zoom_start=12)
        
        # Marker Lokasi
        folium.Marker(
            [data['lat'], data['lon']], 
            popup="Titik Audit", 
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        # Render Peta
        st_folium(m, width=None, height=400)
        
        # Tampilkan Gambar Hasil AI (Heatmap)
        st.markdown("### 👁️ Pandangan Mata AI (SAR-Hydra)")
        st.image(data['image_url'], caption=f"Heatmap Kejenuhan Tanah - Tanggal Citra: {data['date']}", use_column_width=True)

    with col2:
        st.header("📊 Laporan Audit")
        
        # -- BAGIAN FITUR BARU --
        final_risk = data['avg_risk'] # Ini sudah ditambah hujan
        base_sat = data['base_saturation']
        avg_rain = data['avg_rain']
        rain_inc = data['rain_increase']

        # Metric Utama
        st.metric("Tanggal Data Satelit", data['date'])
        
        # Metric Rata-rata Hujan
        st.metric("Rata-rata Hujan (Sejak Satelit)", f"{avg_rain:.2f} mm/hari")

        # Metric Kejenuhan (Dengan Delta Hujan)
        delta_color = "inverse" if final_risk > 50 else "normal"
        st.metric(
            "Estimasi Kejenuhan Tanah (Final)", 
            f"{final_risk:.2f}%", 
            delta=f"+{rain_inc:.2f}% dari Hujan", 
            delta_color=delta_color
        )
        st.caption(f"Base Saturation (AI only): {base_sat:.2f}%")
        
        st.divider()

        # Kotak Status
        status = data['status']
        if "BAHAYA" in status:
            st.error(f"### {status}")
            st.write(f"**Action:** {data['message']}")
        elif "WASPADA" in status:
            st.warning(f"### {status}")
            st.write(f"**Action:** {data['message']}")
        else:
            st.success(f"### {status}")
            st.write(f"**Action:** {data['message']}")

        st.divider()

        # Simulasi Data Forensik
        st.subheader("📁 Data Legalitas Lahan")
        
        # Logika dummy untuk simulasi forensik
        zona = "HUTAN LINDUNG" if final_risk > 60 else "PEMUKIMAN"
        temuan = "ANOMALI TERDETEKSI ❗" if final_risk > 75 else "SESUAI PERUNTUKAN"
        
        st.code(f"""
        [AUDIT SYSTEM LOG]
        > Checking Permit Database...
        > Coord: {data['lat']}, {data['lon']}
        > Designated Zone: {zona}
        > Rain Load Analysis: {avg_rain:.2f} mm/day avg
        > Detected Condition: Wet/Saturated ({final_risk:.1f}%)
        > FORENSIC RESULT: {temuan}
        """)

else:
    # Tampilan jika belum ada data
    st.info("👈 Silakan masukkan koordinat di sidebar kiri dan klik tombol untuk memulai analisis.")