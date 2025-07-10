"""
Database of renowned water bodies with their coordinates and information
"""

WATER_BODIES = {
    "Lake Superior": {
        "coordinates": [47.7211, -87.4494],
        "category": "Great Lakes",
        "location": "North America (USA/Canada)",
        "area": "82,100 km²",
        "description": "Largest of the Great Lakes by surface area"
    },
    "Lake Victoria": {
        "coordinates": [-1.2921, 33.2435],
        "category": "African Great Lakes",
        "location": "East Africa (Uganda/Kenya/Tanzania)",
        "area": "59,947 km²",
        "description": "Africa's largest lake by area"
    },
    "Caspian Sea": {
        "coordinates": [42.5, 51.5],
        "category": "Inland Sea",
        "location": "Central Asia",
        "area": "371,000 km²",
        "description": "World's largest inland body of water"
    },
    "Dead Sea": {
        "coordinates": [31.5, 35.5],
        "category": "Salt Lake",
        "location": "Middle East (Israel/Jordan)",
        "area": "605 km²",
        "description": "Hypersaline lake, lowest point on Earth's surface"
    },
    "Lake Baikal": {
        "coordinates": [53.5587, 108.165],
        "category": "Freshwater Lake",
        "location": "Siberia, Russia",
        "area": "31,722 km²",
        "description": "World's deepest and oldest freshwater lake"
    },
    "Great Salt Lake": {
        "coordinates": [41.0, -112.5],
        "category": "Salt Lake",
        "location": "Utah, USA",
        "area": "4,400 km²",
        "description": "Largest saltwater lake in the Western Hemisphere"
    },
    "Lake Geneva": {
        "coordinates": [46.4, 6.5],
        "category": "Alpine Lake",
        "location": "Switzerland/France",
        "area": "580 km²",
        "description": "One of the largest lakes in Western Europe"
    },
    "Lake Tahoe": {
        "coordinates": [39.0968, -120.0324],
        "category": "Mountain Lake",
        "location": "California/Nevada, USA",
        "area": "495 km²",
        "description": "Large freshwater lake in the Sierra Nevada"
    },
    "Crater Lake": {
        "coordinates": [42.9446, -122.1090],
        "category": "Volcanic Lake",
        "location": "Oregon, USA",
        "area": "53 km²",
        "description": "Lake formed in a volcanic caldera"
    },
    "Lake Bled": {
        "coordinates": [46.3683, 14.0937],
        "category": "Alpine Lake",
        "location": "Slovenia",
        "area": "1.45 km²",
        "description": "Glacial lake in the Julian Alps"
    },
    "Loch Ness": {
        "coordinates": [57.3229, -4.4244],
        "category": "Highland Loch",
        "location": "Scotland, UK",
        "area": "56 km²",
        "description": "Large, deep freshwater loch in the Scottish Highlands"
    },
    "Lake Como": {
        "coordinates": [46.0, 9.25],
        "category": "Alpine Lake",
        "location": "Lombardy, Italy",
        "area": "146 km²",
        "description": "Glacial lake in the Italian Alps"
    },
    "Lake Titicaca": {
        "coordinates": [-15.8422, -69.6447],
        "category": "High Altitude Lake",
        "location": "Peru/Bolivia",
        "area": "8,372 km²",
        "description": "Large freshwater lake in the Andes"
    },
    "Lake Malawi": {
        "coordinates": [-12.0, 34.5],
        "category": "African Great Lakes",
        "location": "Malawi/Mozambique/Tanzania",
        "area": "29,600 km²",
        "description": "African Great Lake with high biodiversity"
    },
    "Lake Tanganyika": {
        "coordinates": [-6.0, 29.5],
        "category": "African Great Lakes",
        "location": "Central Africa",
        "area": "32,900 km²",
        "description": "Second deepest lake in the world"
    },
    "Yellowstone Lake": {
        "coordinates": [44.4605, -110.3964],
        "category": "High Altitude Lake",
        "location": "Wyoming, USA",
        "area": "341 km²",
        "description": "Largest high-altitude lake in North America"
    },
    "Lake Winnipeg": {
        "coordinates": [52.0, -97.0],
        "category": "Prairie Lake",
        "location": "Manitoba, Canada",
        "area": "24,514 km²",
        "description": "Large lake in central Canada"
    },
    "Lake Okeechobee": {
        "coordinates": [26.9342, -80.8010],
        "category": "Freshwater Lake",
        "location": "Florida, USA",
        "area": "1,900 km²",
        "description": "Largest freshwater lake in Florida"
    },
    "Salton Sea": {
        "coordinates": [33.3, -115.8],
        "category": "Salt Lake",
        "location": "California, USA",
        "area": "970 km²",
        "description": "Shallow saline lake in the Colorado Desert"
    },
    "Lake Constance": {
        "coordinates": [47.6, 9.5],
        "category": "Alpine Lake",
        "location": "Germany/Switzerland/Austria",
        "area": "536 km²",
        "description": "Lake on the Rhine at the northern foot of the Alps"
    },
    "Lake Garda": {
        "coordinates": [45.6389, 10.7578],
        "category": "Alpine Lake",
        "location": "Northern Italy",
        "area": "370 km²",
        "description": "Largest lake in Italy"
    },
    "Lake Champlain": {
        "coordinates": [44.5, -73.3],
        "category": "Valley Lake",
        "location": "Vermont/New York, USA",
        "area": "1,127 km²",
        "description": "Natural freshwater lake in the Champlain Valley"
    },
    "Lake George": {
        "coordinates": [43.5, -73.7],
        "category": "Mountain Lake",
        "location": "New York, USA",
        "area": "113 km²",
        "description": "Long, narrow oligotrophic lake"
    },
    "Mono Lake": {
        "coordinates": [38.0, -119.0],
        "category": "Salt Lake",
        "location": "California, USA",
        "area": "180 km²",
        "description": "Saline soda lake in the Eastern Sierra"
    },
    "Lake Powell": {
        "coordinates": [37.0, -111.0],
        "category": "Reservoir",
        "location": "Utah/Arizona, USA",
        "area": "658 km²",
        "description": "Artificial reservoir on the Colorado River"
    }
}

def get_water_bodies_by_category(category):
    """Get all water bodies in a specific category"""
    return {name: data for name, data in WATER_BODIES.items() 
            if data['category'] == category}

def get_water_bodies_by_region(region_keyword):
    """Get water bodies containing a region keyword in their location"""
    return {name: data for name, data in WATER_BODIES.items() 
            if region_keyword.lower() in data['location'].lower()}

def get_categories():
    """Get all unique categories"""
    return list(set(data['category'] for data in WATER_BODIES.values()))

def get_nearest_water_body(lat, lon, max_distance_km=100):
    """Find the nearest water body to given coordinates"""
    import math
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points on Earth"""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance = R * c
        
        return distance
    
    nearest = None
    min_distance = float('inf')
    
    for name, data in WATER_BODIES.items():
        wb_lat, wb_lon = data['coordinates']
        distance = haversine_distance(lat, lon, wb_lat, wb_lon)
        
        if distance < min_distance and distance <= max_distance_km:
            min_distance = distance
            nearest = {
                'name': name,
                'data': data,
                'distance_km': distance
            }
    
    return nearest
