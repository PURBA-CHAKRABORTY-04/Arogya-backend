"""
distance_service.py — Batched Distance Calculation & Session Caching Service.

Calculates real travel distances between patient origin location and candidate doctor destinations.
Batches requests into a single API call per candidate pool to optimize network roundtrips
and caches results per search session in memory to avoid repeat API overhead.

Sample Batched Request Structure (Google Distance Matrix API):
GET https://maps.googleapis.com/maps/api/distancematrix/json
    ?origins=12.9716,77.5946
    &destinations=12.9352,77.6245|12.9250,77.5898|12.9784,77.6408
    &mode=driving
    &key=YOUR_API_KEY

Sample Batched Request Structure (OSRM Table API - Open Source Routing Machine):
GET http://router.project-osrm.org/table/v1/driving/77.5946,12.9716;77.6245,12.9352;77.5898,12.9250
    ?sources=0
    &destinations=1,2
"""

import math
import os
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any
import requests


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes straight-line distance in kilometers between two lat/long points using the Haversine formula.
    Used as an immediate fallback or for local distance approximation when APIs are unreachable.
    Real driving distance is typically 1.2x to 1.4x of straight-line distance.
    """
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c * 1.3, 2)  # Scale by 1.3x for realistic road driving factor


class BatchedDistanceService:
    """
    Service for resolving real travel distances for doctor search candidates.
    Supports in-memory session caching and batched API execution.
    """

    def __init__(self):
        # Session cache structure: { session_id: { (origin_str, dest_id): distance_km } }
        self._session_cache: Dict[str, Dict[str, float]] = {}

    def _get_cache_key(self, origin: str, dest_identifier: str) -> str:
        return f"{origin.strip().lower()}||{dest_identifier.strip().lower()}"

    def get_cached_distance(
        self, session_id: str, origin: str, dest_identifier: str
    ) -> Optional[float]:
        """Retrieves distance from session cache if available."""
        if not session_id or session_id not in self._session_cache:
            return None
        key = self._get_cache_key(origin, dest_identifier)
        return self._session_cache[session_id].get(key)

    def set_cached_distance(
        self, session_id: str, origin: str, dest_identifier: str, distance_km: float
    ) -> None:
        """Stores distance in session cache."""
        if not session_id:
            return
        if session_id not in self._session_cache:
            self._session_cache[session_id] = {}
        key = self._get_cache_key(origin, dest_identifier)
        self._session_cache[session_id][key] = round(distance_km, 2)

    def compute_batch_distances(
        self,
        session_id: str,
        patient_origin: Dict[str, Any],  # {"lat": float, "lng": float, "address": str}
        doctors: List[Dict[str, Any]],     # List of doctor dicts with id, address, lat, lng
    ) -> Dict[str, float]:
        """
        Computes travel distance in km for a batch of candidate doctors from patient_origin.
        Checks session cache first, then makes a single batched API call for missing entries.
        
        Returns:
            Dict mapping doctor_id (str) -> distance_km (float)
        """
        results: Dict[str, float] = {}
        missing_doctors: List[Dict[str, Any]] = []

        origin_lat = patient_origin.get("lat")
        origin_lng = patient_origin.get("lng")
        origin_address = patient_origin.get("address", "")
        origin_str = f"{origin_lat},{origin_lng}" if (origin_lat and origin_lng) else origin_address

        if not origin_str:
            # Default neutral distances if no patient location supplied
            return {str(d["id"]): 5.0 for d in doctors}

        # 1. Check session cache first
        for doc in doctors:
            doc_id = str(doc["id"])
            cached = self.get_cached_distance(session_id, origin_str, doc_id)
            if cached is not None:
                results[doc_id] = cached
            else:
                missing_doctors.append(doc)

        if not missing_doctors:
            return results

        # 2. Execute single batched calculation for uncached doctors
        batched_results = self._fetch_batched_matrix(origin_lat, origin_lng, origin_address, missing_doctors)

        # 3. Store in session cache & build final results dictionary
        for doc in missing_doctors:
            doc_id = str(doc["id"])
            dist = batched_results.get(doc_id, 8.0)  # Default fallback 8 km if missing
            self.set_cached_distance(session_id, origin_str, doc_id, dist)
            results[doc_id] = dist

        return results

    def _fetch_batched_matrix(
        self,
        origin_lat: Optional[float],
        origin_lng: Optional[float],
        origin_address: str,
        doctors: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Executes batched distance resolution (Google Distance Matrix -> OSRM -> Haversine fallback)."""
        distances: Dict[str, float] = {}

        # Option A: Google Distance Matrix API (if GOOGLE_MAPS_API_KEY environment variable is set)
        google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if google_api_key:
            try:
                origin_param = f"{origin_lat},{origin_lng}" if (origin_lat and origin_lng) else origin_address
                destinations = []
                for d in doctors:
                    if d.get("latitude") and d.get("longitude"):
                        destinations.append(f"{d['latitude']},{d['longitude']}")
                    else:
                        destinations.append(d.get("address") or d.get("city", "Bangalore"))

                url = "https://maps.googleapis.com/maps/api/distancematrix/json"
                params = {
                    "origins": origin_param,
                    "destinations": "|".join(destinations),
                    "mode": "driving",
                    "key": google_api_key,
                }
                resp = requests.get(url, params=params, timeout=4.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "OK" and data.get("rows"):
                        elements = data["rows"][0].get("elements", [])
                        for idx, el in enumerate(elements):
                            if idx < len(doctors):
                                doc_id = str(doctors[idx]["id"])
                                if el.get("status") == "OK" and "distance" in el:
                                    meters = el["distance"]["value"]
                                    distances[doc_id] = round(meters / 1000.0, 2)
                if distances:
                    return distances
            except Exception as exc:
                print(f"[DISTANCE SERVICE API ERROR] Google Distance Matrix failed: {exc}")

        # Option B: OSRM Public Table API (Open Source Routing Machine) if lat/lng available
        if origin_lat is not None and origin_lng is not None:
            valid_doc_coords = []
            valid_doc_ids = []
            for d in doctors:
                d_lat = d.get("latitude")
                d_lng = d.get("longitude")
                if d_lat is not None and d_lng is not None:
                    valid_doc_coords.append(f"{d_lng},{d_lat}")
                    valid_doc_ids.append(str(d["id"]))

            if valid_doc_coords:
                try:
                    coords_str = f"{origin_lng},{origin_lat};" + ";".join(valid_doc_coords)
                    osrm_url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?sources=0"
                    resp = requests.get(osrm_url, timeout=3.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("code") == "Ok" and "durations" in data:
                            # OSRM provides duration matrix; estimate distance (assuming avg 35 km/h driving speed)
                            durations = data["durations"][0][1:]  # skip source to source
                            for idx, dur_sec in enumerate(durations):
                                if idx < len(valid_doc_ids):
                                    dist_km = round((dur_sec / 3600.0) * 35.0, 2)
                                    distances[valid_doc_ids[idx]] = max(dist_km, 0.5)
                    if distances:
                        return distances
                except Exception as exc:
                    print(f"[DISTANCE SERVICE OSRM ERROR] OSRM API failed: {exc}")

        # Option C: Haversine distance with road driving factor for remaining candidates
        for doc in doctors:
            doc_id = str(doc["id"])
            if doc_id not in distances:
                d_lat = doc.get("latitude")
                d_lng = doc.get("longitude")
                if origin_lat is not None and origin_lng is not None and d_lat is not None and d_lng is not None:
                    distances[doc_id] = haversine_distance_km(origin_lat, origin_lng, d_lat, d_lng)
                else:
                    # Synthetic city-distance fallback estimation based on city/address strings
                    distances[doc_id] = 7.5  # Reasonable local urban commute default (7.5 km)

        return distances


# Global singleton distance service instance
distance_service = BatchedDistanceService()
