"""Resolve the dashboard's featured water bodies to OpenStreetMap references.

    python scripts/build_featured_locations.py      # writes web/lib/featured.json

Each entry has a hand-picked analysis point (large lakes: the area worth looking at) and a
short reason it is interesting. Nominatim provides the OSM reference so the API can fetch
the outline quickly. Respects Nominatim's 1 request/second policy.
"""

import json
import math
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aquadoc_core.waterbody import find_water_body, lookup_water_body  # noqa: E402

USER_AGENT = "AquaDoc/0.2 (satellite water-quality demo)"
OUT = Path(__file__).resolve().parents[1] / "web" / "lib" / "featured.json"

# (group, query, display name, country, lat, lon, why)
FEATURED = [
    ("India", "Hussain Sagar, Hyderabad", "Hussain Sagar", "Telangana", 17.4239, 78.4738,
     "Urban lake receiving sewage inflows; regularly hypereutrophic"),
    ("India", "Dal Lake, Srinagar", "Dal Lake", "Jammu & Kashmir", 34.1120, 74.8650,
     "Weed growth and nutrient loading from houseboats and settlements"),
    ("India", "Chilika Lake", "Chilika Lake", "Odisha", 19.7000, 85.3500,
     "Asia's largest brackish lagoon; sediment-laden river inflows"),
    ("India", "Vembanad Lake", "Vembanad Lake", "Kerala", 9.6200, 76.3800,
     "Long backwater lake with water-hyacinth blooms and turbid inflows"),
    ("India", "Loktak Lake", "Loktak Lake", "Manipur", 24.5300, 93.8000,
     "Floating vegetation islands (phumdis) cover much of the surface"),
    ("India", "Bellandur Lake, Bengaluru", "Bellandur Lake", "Karnataka", 12.9360, 77.6640,
     "Heavily polluted urban lake known for foam and weed mats"),
    ("India", "Wular Lake", "Wular Lake", "Jammu & Kashmir", 34.3700, 74.6000,
     "Large shallow freshwater lake with heavy siltation"),
    ("India", "Upper Lake, Bhopal", "Upper Lake (Bhojtal)", "Madhya Pradesh", 23.2450, 77.3600,
     "City drinking-water reservoir under urban pressure"),
    ("India", "Pulicat Lake", "Pulicat Lake", "Andhra Pradesh / Tamil Nadu", 13.5500, 80.1700,
     "Shallow coastal lagoon with variable turbidity"),
    ("India", "Sela Lake", "Sela Lake", "Arunachal Pradesh", 27.5050, 92.1040,
     "High-altitude lake near Sela Pass; often frozen in winter"),
    ("India", "Deepor Beel", "Deepor Beel", "Assam", 26.1200, 91.6600,
     "Ramsar wetland beside Guwahati receiving city runoff"),
    ("India", "Carambolim Lake", "Carambolim Lake", "Goa", 15.4870, 73.9220,
     "Small bird-rich lake covered in floating vegetation"),
    ("India", "Kankaria Lake, Ahmedabad", "Kankaria Lake", "Gujarat", 23.0060, 72.6010,
     "Urban lake in Ahmedabad, recharged with river water"),
    ("India", "Nal Sarovar", "Nal Sarovar", "Gujarat", 22.8000, 72.0300,
     "Shallow, turbid bird sanctuary lake with many islets"),
    ("India", "Damdama Lake", "Damdama Lake", "Haryana", 28.3000, 77.1300,
     "Rain-fed reservoir in the Aravalli hills near Gurugram"),
    ("India", "Gobind Sagar", "Gobind Sagar", "Himachal Pradesh", 31.4000, 76.5500,
     "Bhakra Dam reservoir; sediment from the Sutlej"),
    ("India", "Lonar Lake", "Lonar Lake", "Maharashtra", 19.9770, 76.5080,
     "Soda crater lake so dense with algae that it reads as surface scum, not open water"),
    ("India", "Powai Lake, Mumbai", "Powai Lake", "Maharashtra", 19.1270, 72.9050,
     "Urban Mumbai lake with sewage inflow and water hyacinth"),
    ("India", "Umiam Lake", "Umiam Lake", "Meghalaya", 25.6600, 91.8900,
     "Hill reservoir near Shillong affected by catchment erosion"),
    ("India", "Palak Dil", "Palak Dil", "Mizoram", 22.1900, 92.9000,
     "Mizoram's largest natural lake, in a forested reserve"),
    ("India", "Lake Pichola, Udaipur", "Lake Pichola", "Rajasthan", 24.5720, 73.6800,
     "Historic urban lake in Udaipur under tourism pressure"),
    ("India", "Sambhar Salt Lake", "Sambhar Salt Lake", "Rajasthan", 26.9500, 75.0800,
     "India's largest inland salt lake; often too salty and shallow to read as open water"),
    ("India", "Tsomgo Lake", "Tsomgo Lake", "Sikkim", 27.3750, 88.7630,
     "Glacial lake at 3,750 m; often frozen in winter"),
    ("India", "Chembarambakkam Lake", "Chembarambakkam Lake", "Tamil Nadu", 13.0100, 80.0600,
     "Reservoir supplying Chennai's drinking water"),
    ("India", "Kodaikanal Lake", "Kodaikanal Lake", "Tamil Nadu", 10.2330, 77.4880,
     "Star-shaped hill-station lake with nutrient loading"),
    ("India", "Rudrasagar Lake", "Rudrasagar Lake", "Tripura", 23.5000, 91.3100,
     "Lake around the Neermahal water palace"),
    ("India", "Ramgarh Tal, Gorakhpur", "Ramgarh Tal", "Uttar Pradesh", 26.7300, 83.4100,
     "Large urban lake in Gorakhpur, restored after heavy pollution"),
    ("India", "Naini Lake, Nainital", "Naini Lake", "Uttarakhand", 29.3900, 79.4550,
     "Hill-town lake with seasonal algal growth"),
    ("India", "Tehri Lake", "Tehri Reservoir", "Uttarakhand", 30.4000, 78.4500,
     "Deep Himalayan reservoir on the Bhagirathi"),
    ("India", "Rabindra Sarobar, Kolkata", "Rabindra Sarobar", "West Bengal", 22.5120, 88.3620,
     "Artificial lake in south Kolkata used for rowing"),
    ("India", "Bhalswa Lake", "Bhalswa Lake", "Delhi", 28.7430, 77.1650,
     "Horseshoe lake beside a landfill in north Delhi"),
    ("India", "Pangong Tso", "Pangong Tso", "Ladakh", 33.7500, 78.6600,
     "Very clear high-altitude lake at 4,225 m; frozen in winter"),
    ("India", "Sukhna Lake", "Sukhna Lake", "Chandigarh", 30.7420, 76.8180,
     "Rain-fed reservoir at the foot of the Shivaliks; prone to silting"),
    ("India", "Usteri Lake", "Ousteri Lake", "Puducherry", 11.9550, 79.7450,
     "Large shallow wetland with dense aquatic vegetation"),
    ("World", "Lake Erie", "Lake Erie (western basin)", "USA / Canada", 41.7000, -83.2500,
     "Recurring summer cyanobacteria blooms off Toledo"),
    ("World", "太湖", "Lake Taihu", "China", 31.2500, 120.1500,  # OSM name is Chinese
     "Large shallow lake with persistent cyanobacteria blooms"),
    ("World", "Lake Victoria", "Lake Victoria (Winam Gulf)", "Kenya / Uganda / Tanzania", -0.2500, 34.6500,
     "Water hyacinth and algal blooms near Kisumu"),
    ("World", "Lake Okeechobee", "Lake Okeechobee", "USA", 26.9500, -80.8000,
     "Large Florida lake with frequent harmful algal blooms"),
    ("World", "Lake Titicaca", "Lake Titicaca (Puno Bay)", "Peru / Bolivia", -15.8200, -69.9800,
     "High-altitude lake; algal blooms near Puno"),
    ("World", "Lake Balaton", "Lake Balaton", "Hungary", 46.8500, 17.7500,
     "Shallow lake whose turbidity changes with wind"),
    ("World", "Clear Lake, California", "Clear Lake", "USA", 39.0800, -122.8200,
     "Ancient eutrophic lake with recurring toxic cyanobacteria"),
    ("World", "Lake Winnipeg", "Lake Winnipeg (south basin)", "Canada", 50.7500, -96.8000,
     "Extensive late-summer cyanobacteria blooms"),
    ("World", "Lake Tahoe", "Lake Tahoe", "USA", 39.0900, -120.0300,
     "Very clear oligotrophic reference lake"),
    ("World", "Lake Geneva", "Lake Geneva (Lac Léman)", "Switzerland / France", 46.4500, 6.5500,
     "Deep, clear lake for comparison with turbid ones"),
]

# Lakes OpenStreetMap maps without a name (so no name search finds them): fixed references.
# The API fetches their outline from Overpass by id.
FIXED = [
    {"group": "India", "name": "Potheri Lake (SRM University)", "country": "Tamil Nadu",
     "latitude": 12.816, "longitude": 80.0458, "osm_ref": "way/67551369",
     "why": "Campus-side lake at SRM Kattankulathur; unnamed on OpenStreetMap, outline fetched by id"},
]


def resolve(query: str, lat: float, lon: float) -> str | None:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "jsonv2", "limit": 10},
        headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    water = [p for p in response.json()
             if p.get("category") in {"natural", "water", "landuse"} and p.get("type") in
             {"water", "lake", "reservoir", "lagoon", "bay", "wetland", "basin"}]
    # Prefer the candidate whose bounding box contains the chosen point ...
    for place in water:
        south, north, west, east = (float(v) for v in place["boundingbox"])
        if south <= lat <= north and west <= lon <= east:
            return f"{place['osm_type']}/{place['osm_id']}"
    # ... otherwise the nearest one within ~5 km (hand-picked points can be slightly off).
    near = [(math.hypot(float(p["lat"]) - lat, (float(p["lon"]) - lon) * math.cos(math.radians(lat))), p) for p in water]
    near = [item for item in near if item[0] < 0.045]
    if near:
        place = min(near, key=lambda item: item[0])[1]
        return f"{place['osm_type']}/{place['osm_id']}"
    return None


def resolve_by_point(lat: float, lon: float) -> str | None:
    """Fallback when the name is not on OpenStreetMap: the mapped water body around the point."""
    try:
        body = find_water_body(lat, lon, search_radius_m=1500)
    except Exception as exc:  # Overpass is slow and sometimes busy
        print("   point lookup failed:", type(exc).__name__)
        return None
    return f"{body.osm_type}/{body.osm_id}" if body else None


def main() -> None:
    entries = []
    for group, query, name, country, lat, lon, why in FEATURED:
        ref = resolve(query, lat, lon)
        note = ""
        if not ref:
            ref, note = resolve_by_point(lat, lon), "found by point"
        if ref:
            # The analysis point must be on the water; otherwise move it onto the lake surface.
            time.sleep(1.1)
            body = lookup_water_body(ref, lat, lon)
            if body is None:
                ref, note = None, "no polygon"
            elif not body.contains_point:
                inside = body.geometry.representative_point()
                lat, lon, note = round(inside.y, 4), round(inside.x, 4), "point moved onto water"
        if not ref:
            print(f"{name:32s} SKIPPED (not found on OpenStreetMap) {note}")
            time.sleep(1.1)
            continue
        print(f"{name:32s} {ref} {note}")
        entries.append({"group": group, "name": name, "country": country, "latitude": lat, "longitude": lon,
                        "osm_ref": ref, "why": why})
        time.sleep(1.1)
    india_end = max((i for i, e in enumerate(entries) if e["group"] == "India"), default=-1) + 1
    entries[india_end:india_end] = FIXED
    OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
