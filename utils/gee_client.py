import os
from io import BytesIO
from datetime import datetime, timedelta

import requests
from PIL import Image


class GoogleEarthEngineClient:
    """Optional Google Earth Engine client for fetching live satellite thumbnails."""

    def __init__(self):
        self._ee = None
        self._initialized = False
        self._init_error = None
        self.project = (
            os.getenv("GOOGLE_EARTH_ENGINE_PROJECT")
            or os.getenv("GEE_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        self.service_account = os.getenv("GEE_SERVICE_ACCOUNT")
        self.private_key_file = os.getenv("GEE_PRIVATE_KEY_FILE") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        self.private_key_json = os.getenv("GEE_PRIVATE_KEY_JSON")
        self.collection_id = os.getenv(
            "GEE_IMAGE_COLLECTION", "COPERNICUS/S2_SR_HARMONIZED"
        )

    def _load_ee(self):
        if self._ee is None:
            try:
                import ee  # type: ignore

                self._ee = ee
            except Exception as exc:
                self._init_error = f"earthengine-api not available: {exc}"
                raise
        return self._ee

    def is_configured(self):
        return bool(
            self.project
            and (
                self.private_key_file
                or self.private_key_json
                or self.service_account
                or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            )
        )

    def get_status(self):
        if self._initialized:
            return {"ready": True, "message": "Google Earth Engine configured"}

        if self._init_error:
            return {"ready": False, "message": self._init_error}

        if not self.project:
            return {
                "ready": False,
                "message": "Missing GEE project. Set GOOGLE_EARTH_ENGINE_PROJECT or GEE_PROJECT.",
            }

        if not (self.private_key_file or self.private_key_json or self.service_account):
            return {
                "ready": False,
                "message": (
                    "Missing credentials. Set GOOGLE_APPLICATION_CREDENTIALS, "
                    "GEE_PRIVATE_KEY_FILE, or GEE_PRIVATE_KEY_JSON with GEE_SERVICE_ACCOUNT."
                ),
            }

        return {"ready": False, "message": "Google Earth Engine not initialized yet"}

    def initialize(self):
        if self._initialized:
            return True

        ee = self._load_ee()

        try:
            if self.service_account and (self.private_key_file or self.private_key_json):
                kwargs = {}
                if self.private_key_file:
                    kwargs["key_file"] = self.private_key_file
                if self.private_key_json:
                    kwargs["key_data"] = self.private_key_json
                credentials = ee.ServiceAccountCredentials(self.service_account, **kwargs)
                ee.Initialize(credentials=credentials, project=self.project)
            else:
                ee.Initialize(project=self.project)

            self._initialized = True
            self._init_error = None
            return True
        except Exception as exc:
            self._init_error = f"GEE initialization failed: {exc}"
            raise RuntimeError(self._init_error) from exc

    def _buffer_meters_for_resolution(self, resolution_label):
        mapping = {
            "High (10m)": 1500,
            "Medium (30m)": 2500,
            "Low (100m)": 5000,
        }
        return mapping.get(resolution_label, 2500)

    def _dimensions_for_resolution(self, resolution_label):
        mapping = {
            "High (10m)": 1024,
            "Medium (30m)": 768,
            "Low (100m)": 512,
        }
        return mapping.get(resolution_label, 768)

    def fetch_location_image(self, location, resolution_label="Medium (30m)", days_back=180):
        ee = self._load_ee()
        self.initialize()

        lat = location["lat"]
        lon = location["lon"]
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(self._buffer_meters_for_resolution(resolution_label)).bounds()

        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        collection = (
            ee.ImageCollection(self.collection_id)
            .filterBounds(point)
            .filterDate(start_date.isoformat(), end_date.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        try:
            image_count = collection.size().getInfo()
        except Exception as exc:
            raise RuntimeError(f"Unable to query Earth Engine collection: {exc}") from exc

        if not image_count:
            raise RuntimeError(
                f"No images found in {self.collection_id} for {location['name']} in the last {days_back} days."
            )

        image = collection.first()
        visualized = image.visualize(bands=["B4", "B3", "B2"], min=0, max=3000, gamma=1.2)
        thumb_url = visualized.getThumbURL(
            {
                "region": region,
                "dimensions": self._dimensions_for_resolution(resolution_label),
                "format": "png",
            }
        )

        response = requests.get(thumb_url, timeout=60)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")

        metadata = {
            "source": "Google Earth Engine",
            "collection_id": self.collection_id,
            "image_count": image_count,
            "project": self.project,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "cloud_cover": self._safe_get_info(image, "CLOUDY_PIXEL_PERCENTAGE"),
            "capture_time": self._safe_get_info(image, "system:time_start", as_timestamp=True),
            "thumb_url": thumb_url,
        }

        return pil_image, metadata

    def _safe_get_info(self, image, property_name, as_timestamp=False):
        try:
            value = image.get(property_name).getInfo()
            if as_timestamp and value:
                return datetime.utcfromtimestamp(value / 1000).isoformat() + "Z"
            return value
        except Exception:
            return None
