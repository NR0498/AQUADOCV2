"""What each Sentinel-2 band reveals about water, and which pollutants AquaDoc estimates.

Static reference content served to the dashboard's band explainer and result cards.
"""

from __future__ import annotations

BANDS: list[dict] = [
    {"band": "B2", "name": "Blue", "center_nm": 490, "resolution_m": 10,
     "water": "Strongly absorbed by dissolved organic matter (CDOM) and by chlorophyll's blue absorption peak. "
              "High blue reflectance indicates clear water; low blue relative to green indicates CDOM or algae.",
     "indicates": ["cdom_a440", "secchi_depth", "chlorophyll_a"]},
    {"band": "B3", "name": "Green", "center_nm": 560, "resolution_m": 10,
     "water": "Least absorbed by pigments: phytoplankton and particles scatter green, so it brightens with algae "
              "and sediment. Green vs blue separates CDOM-rich from clear water.",
     "indicates": ["chlorophyll_a", "tss", "cdom_a440"]},
    {"band": "B4", "name": "Red", "center_nm": 665, "resolution_m": 10,
     "water": "Chlorophyll-a absorption maximum (a dip where algae are present) and the primary sediment band: "
              "suspended particles raise red reflectance almost linearly until it saturates.",
     "indicates": ["chlorophyll_a", "tss", "turbidity"]},
    {"band": "B5", "name": "Red edge 1", "center_nm": 705, "resolution_m": 20,
     "water": "Reflectance peak of algae-rich water, between chlorophyll absorption at 665 nm and water absorption "
              "beyond 700 nm. B5 relative to B4 (NDCI) is the classic chlorophyll signal in turbid inland waters.",
     "indicates": ["chlorophyll_a"]},
    {"band": "B6", "name": "Red edge 2", "center_nm": 740, "resolution_m": 20,
     "water": "Water absorbs strongly here, so only dense particles or near-surface algae return signal. Rising B6 "
              "signals very high biomass, surface scum or heavy sediment.",
     "indicates": ["tss", "chlorophyll_a", "floating_material"]},
    {"band": "B7", "name": "Red edge 3", "center_nm": 783, "resolution_m": 20,
     "water": "Near-infrared scattering from suspended sediment in very turbid water; near zero in clear water.",
     "indicates": ["tss", "turbidity"]},
    {"band": "B8", "name": "NIR", "center_nm": 842, "resolution_m": 10,
     "water": "Clear water is nearly black. Bright NIR on water means floating material (vegetation mats, algal scum, "
              "debris, foam) or extreme turbidity. Used in the Floating Debris Index and NDVI.",
     "indicates": ["floating_material"]},
    {"band": "B8A", "name": "Narrow NIR", "center_nm": 865, "resolution_m": 20,
     "water": "Turbidity in highly turbid water, where red saturates (Dogliotti 2015 switches to this band).",
     "indicates": ["turbidity"]},
    {"band": "B11", "name": "SWIR 1", "center_nm": 1610, "resolution_m": 20,
     "water": "Water absorbs SWIR completely, so any signal is glint, land mixing, or floating material. Used to "
              "detect water (MNDWI), flag glint and as the FDI baseline.",
     "indicates": ["water_mask", "floating_material"]},
    {"band": "B12", "name": "SWIR 2", "center_nm": 2190, "resolution_m": 20,
     "water": "Like B11; helps separate land and cloud from water.",
     "indicates": ["water_mask"]},
]

BAND_CENTERS: dict[str, int] = {band["band"]: band["center_nm"] for band in BANDS}

PARAMETERS: dict[str, dict] = {
    "chlorophyll_a": {
        "label": "Chlorophyll-a", "unit": "µg/L", "kind": "model", "colormap": "algae",
        "what": "Pigment in algae and cyanobacteria; the standard proxy for algal biomass and nutrient pollution (eutrophication).",
        "bands": ["B4", "B5", "B3", "B2"],
        "scale": [{"max": 2.6, "label": "oligotrophic (low)"}, {"max": 7.3, "label": "mesotrophic"},
                  {"max": 56, "label": "eutrophic"}, {"max": None, "label": "hypereutrophic / bloom"}],
        "scale_source": "Carlson & Simpson (1996) trophic state thresholds",
    },
    "tss": {
        "label": "Total suspended solids", "unit": "mg/L", "kind": "model", "colormap": "sediment",
        "what": "Mass of particles in suspension: sediment from erosion, runoff, dredging or resuspension, plus algae.",
        "bands": ["B4", "B7", "B6", "B3"],
    },
    "turbidity": {
        "label": "Turbidity", "unit": "NTU", "kind": "model", "colormap": "sediment",
        "what": "Cloudiness caused by particles scattering light. Closely tied to suspended solids.",
        "bands": ["B4", "B8A", "B7"],
    },
    "cdom_a440": {
        "label": "Coloured dissolved organic matter", "unit": "1/m (absorption at 440 nm)", "kind": "model",
        "colormap": "cdom",
        "what": "Dissolved organic compounds from decaying vegetation, wetlands, soils or wastewater that stain water brown.",
        "bands": ["B2", "B3"],
    },
    "secchi_depth": {
        "label": "Water clarity (Secchi depth)", "unit": "m", "kind": "model", "colormap": "clarity",
        "what": "Depth at which a white disk disappears from view. Lower means murkier water.",
        "bands": ["B2", "B3", "B4"],
    },
    "floating_material": {
        "label": "Floating material", "unit": "% of water surface", "kind": "index", "colormap": "debris",
        "what": "Surface coverage by vegetation-like material (algal scum, floating plants) or non-vegetation floating "
                "debris (Floating Debris Index). Cannot identify plastic specifically at 10–20 m resolution.",
        "bands": ["B8", "B6", "B11", "B4"],
    },
}

NOT_DETECTABLE: list[dict] = [
    {"name": "Heavy metals (lead, mercury, arsenic…)", "why": "Dissolved metals have no visible or near-infrared signature."},
    {"name": "Bacteria and pathogens (E. coli…)", "why": "Microorganisms are far too small and optically inactive at these wavelengths."},
    {"name": "Nitrates and phosphates", "why": "Dissolved nutrients are invisible; high chlorophyll is only an indirect hint of nutrient pollution."},
    {"name": "Pesticides, pharmaceuticals, PFAS", "why": "Trace chemicals at µg/L levels do not change reflectance."},
    {"name": "Algal toxins (microcystin)", "why": "Toxin production cannot be seen; a bloom does not mean toxins, nor does no bloom mean safe."},
    {"name": "Oil films", "why": "Thin oil is unreliable in optical imagery; radar (Sentinel-1) is the standard tool."},
]
