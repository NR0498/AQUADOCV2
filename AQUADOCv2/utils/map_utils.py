import folium
import numpy as np
from folium import plugins

class MapUtils:
    """Utility functions for map generation and geospatial operations"""
    
    def __init__(self):
        self.default_zoom = 12
        self.satellite_tile_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        
    def create_location_map(self, location_data, show_satellite=True):
        """
        Create an interactive map for a given location
        
        Args:
            location_data: Dictionary with location information
            show_satellite: Whether to include satellite layer
            
        Returns:
            folium.Map: Interactive map object
        """
        # Create base map
        m = folium.Map(
            location=[location_data['lat'], location_data['lon']],
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Add satellite layer if requested
        if show_satellite:
            folium.TileLayer(
                tiles=self.satellite_tile_url,
                attr='Esri',
                name='Satellite',
                overlay=False,
                control=True
            ).add_to(m)
        
        # Add location marker
        folium.Marker(
            [location_data['lat'], location_data['lon']],
            popup=self._create_popup_content(location_data),
            tooltip=f"Click for details: {location_data['name']}",
            icon=folium.Icon(color='blue', icon='water')
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_detection_overlay_map(self, location_data, detection_results):
        """
        Create a map with waste detection overlays
        
        Args:
            location_data: Dictionary with location information
            detection_results: Results from waste detection
            
        Returns:
            folium.Map: Map with detection overlays
        """
        # Create base map
        m = self.create_location_map(location_data, show_satellite=True)
        
        # Add detection overlays
        if detection_results and 'waste_regions' in detection_results:
            self._add_detection_overlays(m, detection_results['waste_regions'], location_data)
        
        # Add heat map if there are enough detection points
        if detection_results and len(detection_results.get('waste_regions', [])) > 3:
            self._add_detection_heatmap(m, detection_results['waste_regions'], location_data)
        
        return m
    
    def _create_popup_content(self, location_data):
        """Create HTML content for location popup"""
        popup_html = f"""
        <div style="width: 200px;">
            <h4>{location_data['name']}</h4>
            <p><strong>Coordinates:</strong><br>
            {location_data['lat']:.4f}, {location_data['lon']:.4f}</p>
        """
        
        # Add additional info if available
        if 'info' in location_data:
            info = location_data['info']
            popup_html += f"""
            <p><strong>Category:</strong> {info.get('category', 'N/A')}</p>
            <p><strong>Location:</strong> {info.get('location', 'N/A')}</p>
            """
            if 'area' in info:
                popup_html += f"<p><strong>Area:</strong> {info['area']}</p>"
        
        popup_html += "</div>"
        
        return folium.Popup(popup_html, max_width=250)
    
    def _add_detection_overlays(self, map_obj, waste_regions, location_data):
        """Add waste detection overlays to the map"""
        
        # Calculate approximate pixel-to-meter conversion (simplified)
        # This is a rough approximation and would need proper geospatial calculation
        lat = location_data['lat']
        meters_per_pixel = 111320 * np.cos(np.radians(lat)) / (2 ** self.default_zoom)
        
        for i, region in enumerate(waste_regions):
            # Convert pixel coordinates to approximate lat/lon offsets
            x, y, w, h = region['bbox']
            
            # Simple conversion (in reality, need proper geospatial transformation)
            lat_offset = (y + h/2) * meters_per_pixel / 111320
            lon_offset = (x + w/2) * meters_per_pixel / (111320 * np.cos(np.radians(lat)))
            
            # Calculate detection location
            detection_lat = location_data['lat'] + (lat_offset * 0.001)  # Scale factor
            detection_lon = location_data['lon'] + (lon_offset * 0.001)
            
            # Determine marker color based on confidence
            if region['confidence'] >= 0.8:
                color = 'red'
                icon = 'exclamation-triangle'
            elif region['confidence'] >= 0.6:
                color = 'orange'
                icon = 'exclamation-circle'
            else:
                color = 'yellow'
                icon = 'question-circle'
            
            # Create popup content for detection
            popup_content = f"""
            <div style="width: 180px;">
                <h5>Waste Detection #{i+1}</h5>
                <p><strong>Type:</strong> {region['waste_type']}</p>
                <p><strong>Confidence:</strong> {region['confidence']:.1%}</p>
                <p><strong>Area:</strong> {region['area']} pixels</p>
            </div>
            """
            
            # Add detection marker
            folium.Marker(
                [detection_lat, detection_lon],
                popup=folium.Popup(popup_content, max_width=200),
                tooltip=f"Waste Detection: {region['waste_type']}",
                icon=folium.Icon(color=color, icon=icon, prefix='fa')
            ).add_to(map_obj)
    
    def _add_detection_heatmap(self, map_obj, waste_regions, location_data):
        """Add a heatmap overlay for waste detections"""
        
        # Prepare heat map data
        heat_data = []
        
        lat = location_data['lat']
        meters_per_pixel = 111320 * np.cos(np.radians(lat)) / (2 ** self.default_zoom)
        
        for region in waste_regions:
            x, y, w, h = region['bbox']
            
            # Convert to lat/lon (simplified)
            lat_offset = (y + h/2) * meters_per_pixel / 111320
            lon_offset = (x + w/2) * meters_per_pixel / (111320 * np.cos(np.radians(lat)))
            
            detection_lat = location_data['lat'] + (lat_offset * 0.001)
            detection_lon = location_data['lon'] + (lon_offset * 0.001)
            
            # Weight by confidence and area
            weight = region['confidence'] * np.log(region['area'] + 1)
            
            heat_data.append([detection_lat, detection_lon, weight])
        
        # Add heatmap layer
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name="Waste Detection Heatmap",
                radius=20,
                max_zoom=18,
                gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
            ).add_to(map_obj)
    
    def calculate_area_coverage(self, detection_results, location_bounds):
        """
        Calculate the area coverage of detected waste
        
        Args:
            detection_results: Results from waste detection
            location_bounds: Geographic bounds of the analyzed area
            
        Returns:
            dict: Area coverage statistics
        """
        if not detection_results or not detection_results.get('waste_regions'):
            return {
                'total_detections': 0,
                'total_waste_area': 0,
                'coverage_percentage': 0.0,
                'density_per_km2': 0.0
            }
        
        waste_regions = detection_results['waste_regions']
        
        # Calculate total waste area in pixels
        total_waste_pixels = sum(region['area'] for region in waste_regions)
        
        # Estimate pixel-to-area conversion (simplified)
        # In reality, this would use proper geospatial calculations
        approx_area_km2 = 1.0  # Placeholder for actual area calculation
        pixels_per_km2 = 1000000  # Placeholder conversion factor
        
        total_waste_area_km2 = total_waste_pixels / pixels_per_km2
        coverage_percentage = (total_waste_area_km2 / approx_area_km2) * 100
        density_per_km2 = len(waste_regions) / approx_area_km2
        
        return {
            'total_detections': len(waste_regions),
            'total_waste_area': total_waste_area_km2,
            'coverage_percentage': coverage_percentage,
            'density_per_km2': density_per_km2,
            'analyzed_area_km2': approx_area_km2
        }
    
    def generate_kml_export(self, location_data, detection_results):
        """
        Generate KML file content for export to Google Earth
        
        Args:
            location_data: Dictionary with location information
            detection_results: Results from waste detection
            
        Returns:
            str: KML file content
        """
        kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Water Body Waste Detection Results</name>
    <description>Waste detection results for {}</description>
    
    <Style id="wasteHigh">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href>
        </Icon>
      </IconStyle>
    </Style>
    
    <Style id="wasteMedium">
      <IconStyle>
        <color>ff00ffff</color>
        <scale>1.0</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href>
        </Icon>
      </IconStyle>
    </Style>
    
    <Style id="wasteLow">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>0.8</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href>
        </Icon>
      </IconStyle>
    </Style>
'''.format(location_data['name'])
        
        kml_content = kml_header
        
        # Add detection placemarks
        if detection_results and 'waste_regions' in detection_results:
            for i, region in enumerate(detection_results['waste_regions']):
                # Determine style based on confidence
                if region['confidence'] >= 0.8:
                    style = "wasteHigh"
                elif region['confidence'] >= 0.6:
                    style = "wasteMedium"
                else:
                    style = "wasteLow"
                
                # Simplified coordinate conversion
                lat = location_data['lat'] + (region['center'][1] * 0.0001)
                lon = location_data['lon'] + (region['center'][0] * 0.0001)
                
                placemark = f'''
    <Placemark>
      <name>Waste Detection #{i+1}</name>
      <description>
        Type: {region['waste_type']}
        Confidence: {region['confidence']:.1%}
        Area: {region['area']} pixels
      </description>
      <styleUrl>#{style}</styleUrl>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>'''
                
                kml_content += placemark
        
        kml_footer = '''
  </Document>
</kml>'''
        
        kml_content += kml_footer
        
        return kml_content
