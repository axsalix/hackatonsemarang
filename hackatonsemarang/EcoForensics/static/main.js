let map;
let marker;

document.addEventListener('DOMContentLoaded', () => {
    const mapEl = document.getElementById('map');
    if (!mapEl) return;

    // Init Map
    map = new maplibregl.Map({
        container: 'map',
        style: {
            version: 8,
            sources: {
                'osm': {
                    type: 'raster',
                    tiles: ['https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'],
                    tileSize: 256,
                    attribution: '&copy; OpenStreetMap'
                }
            },
            layers: [{
                id: 'osm-layer',
                type: 'raster',
                source: 'osm',
                paint: { 'raster-fade-duration': 0 }
            }]
        },
        center: [110.6401, -6.8946],
        zoom: 12,
        pitch: 0
    });

    map.addControl(new maplibregl.NavigationControl());

    marker = new maplibregl.Marker({ color: "#ef4444", draggable: true })
        .setLngLat([110.6401, -6.8946])
        .addTo(map);

    // Events
    map.on('click', (e) => updatePosition(e.lngLat.lat, e.lngLat.lng));
    marker.on('dragend', () => {
        const p = marker.getLngLat();
        updatePosition(p.lat, p.lng, false);
    });
});

function updatePosition(lat, lng, moveMap = true) {
    const safeLat = parseFloat(lat);
    const safeLng = parseFloat(lng);

    if (marker) marker.setLngLat([safeLng, safeLat]);
    
    const latIn = document.getElementById('lat');
    const lonIn = document.getElementById('lon');
    if (latIn) latIn.value = safeLat.toFixed(4);
    if (lonIn) lonIn.value = safeLng.toFixed(4);
    
    const disp = document.getElementById('coord-display');
    if (disp) disp.innerText = `${safeLat.toFixed(4)}, ${safeLng.toFixed(4)}`;

    if (moveMap && map) {
        map.flyTo({ center: [safeLng, safeLat], zoom: 14 });
    }
}

// SEARCH PROXY (Lewat Python)
async function searchLocation() {
    const query = document.getElementById('search-input').value;
    const errorMsg = document.getElementById('search-error');
    if (!query) return;

    try {
        document.body.style.cursor = 'wait';
        const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();

        if (data && data.length > 0) {
            const lat = parseFloat(data[0].lat);
            const lon = parseFloat(data[0].lon);
            updatePosition(lat, lon, true);
            if (errorMsg) errorMsg.style.display = 'none';
        } else {
            if (errorMsg) errorMsg.style.display = 'block';
            else alert("Lokasi tidak ditemukan!");
        }
    } catch (e) {
        console.error(e);
    } finally {
        document.body.style.cursor = 'default';
    }
}

function handleEnter(e) {
    if (e.key === 'Enter') searchLocation();
}

// --- FUNGSI UTAMA: ANALISIS ---
async function runAnalysis() {
    const loader = document.getElementById('loader');
    const dash = document.getElementById('dashboard');
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;

    loader.style.display = 'flex';
    
    try {
        const res = await fetch(`/analyze?lat=${lat}&lon=${lon}`);
        if (!res.ok) throw new Error("API Error");
        const data = await res.json();

        // 1. TAMPILKAN HASIL DI BAWAH PETA
        dash.style.display = 'block';
        
        // 2. ISI DATA
        const setVal = (id, val) => { 
            const el = document.getElementById(id); 
            if (el) el.innerText = val; 
        };

        setVal('ai-val', (data.base_saturation || 0).toFixed(1) + "%");
        setVal('rain-val', (data.avg_rain || 0).toFixed(1) + " mm");
        setVal('final-val', (data.avg_risk || 0).toFixed(1) + "%");

        const img = document.getElementById('heatmap');
        if (img) {
            // Tambah timestamp agar gambar selalu refresh (tidak cache)
            img.src = data.image_url + "?t=" + new Date().getTime();
        }

        setVal('date-tag', data.date || "-");
        setVal('status-text', data.status || "UNKNOWN");
        
        const badge = document.getElementById('risk-badge');
        const st = data.status || "UNKNOWN";
        if (badge) {
            badge.innerText = st;
            badge.className = "badge " + (st.includes("BAHAYA")?"bg-danger":st.includes("WASPADA")?"bg-warn":"bg-safe");
        }

        setVal('logs', `Target: ${lat}, ${lon}\nLogic: ${data.logic}\nRain Impact: ${(data.rain_increase||0).toFixed(2)}%\nMsg: ${data.message}`);

        // 3. SCROLL HALUS (TAPI TIDAK MENGHILANGKAN PETA)
        // Kita scroll ke dashboard, tapi 'block: nearest' memastikan peta tidak terlempar keluar jika dashboard ada di bawah
        setTimeout(() => {
            dash.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);

    } catch (e) {
        alert("Error: " + e.message);
    } finally {
        loader.style.display = 'none';
    }
}