"""Date-interval-aware NLA matching, local imagery QA and resumable audit outputs."""
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import Point, shape

import aquadoc_core.imagery_quality as _imagery_quality
from aquadoc_core.imagery_quality import (FEATURE_VERSION, LakePolygons, QualityConfig, assess,
                              read_aligned, spatial_masks)
from .nla_preprocessing import digest

VERSION = 'nla-date-interval-matching-v1'
STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'


def date_interval(value):
    """Envelope for a calendar date with unknown time AND unknown UTC offset.

    UTC offsets from -12 through +14 hours give a conservative 50-hour envelope.
    These are uncertainty bounds, not claimed sampling timestamps.
    """
    day = datetime.combine(date.fromisoformat(value), time.min, tzinfo=timezone.utc)
    return day - timedelta(hours=14), day + timedelta(hours=36)


def temporal_check(captured_at, start, end, max_days):
    captured = datetime.fromisoformat(captured_at.replace('Z', '+00:00'))
    if captured.tzinfo is None or captured.utcoffset() is None:
        raise ValueError('Satellite timestamp must have a timezone')
    captured = captured.astimezone(timezone.utc)
    minimum = max((start - captured).total_seconds(), (captured - end).total_seconds(), 0) / 3600
    maximum = max(abs((captured - start).total_seconds()), abs((captured - end).total_seconds())) / 3600
    return dict(time_delta_min_hours=minimum, time_delta_max_hours=maximum,
                temporal_status='within_tolerance' if maximum <= max_days * 24 else
                ('ambiguous_time_gap' if minimum <= max_days * 24 else 'outside_time_window'))


class Catalogue:
    def search(self, longitude, latitude, start, end):
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET', 'POST'])
        session.mount('https://', HTTPAdapter(max_retries=retry))
        payload = dict(collections=['sentinel-2-l2a'], intersects=dict(type='Point', coordinates=[longitude, latitude]),
                       datetime=f'{start.isoformat()}/{end.isoformat()}', limit=100)
        url, method, body, features, seen = STAC_URL, 'POST', payload, {}, set()
        try:
            while True:
                request_key = json.dumps([url, method, body], sort_keys=True)
                if request_key in seen or len(seen) >= 100:
                    raise ValueError('STAC pagination loop or page safety limit reached')
                seen.add(request_key)
                response = session.request(method, url, json=body if method == 'POST' else None, timeout=(10, 45))
                response.raise_for_status()
                data = response.json()
                if data.get('type') != 'FeatureCollection' or not isinstance(data.get('features'), list):
                    raise ValueError('Invalid STAC response; not a no-scene result')
                for feature in data['features']:
                    features[feature['id']] = feature
                links = [link for link in data.get('links', []) if link.get('rel') == 'next']
                if not links:
                    break
                link = links[0]
                url, method = link['href'], link.get('method', 'GET').upper()
                if urlsplit(url).hostname != urlsplit(STAC_URL).hostname or method not in ('POST', 'GET'):
                    raise ValueError('Unsupported STAC pagination link')
                body = {**payload, **link.get('body', {})} if link.get('merge') else link.get('body')
        finally:
            session.close()
        return list(features.values())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def match_one(row, lake, lake_crs, config, max_days, catalogue, reader):
    latitude, longitude = float(row['latitude']), float(row['longitude'])
    start, end = date_interval(row['collected_date'])
    result = dict(observation_id=row['observation_id'], lake_id=row['lake_id'], site_id=row['site_id'],
                  location_group=row['location_group'], value=float(row['value']), unit=row['unit'],
                  collected_date=row['collected_date'], collection_time_precision='date', collection_timezone=None,
                  possible_collection_start_utc=start.isoformat(), possible_collection_end_utc=end.isoformat(),
                  status='no_scene', attempts=[], features={}, review_status='pending')
    _, _, lake_mask, neighborhood = spatial_masks(lake, lake_crs, latitude, longitude, config)
    result['lake_interior_pixels'] = int(lake_mask.sum())
    if lake_mask.sum() < config.min_pixels or lake_mask.sum() / neighborhood.sum() < config.min_valid_fraction:
        result.update(status='spatial_rejected', reason='insufficient_lake_interior_in_sampling_neighborhood', review_status='rejected')
        return result
    try:
        items = catalogue.search(longitude, latitude, start - timedelta(days=max_days), end + timedelta(days=max_days))
    except (requests.RequestException, ValueError, KeyError) as exc:
        # Store class only: exceptions may contain temporary signed URLs or credentials.
        result.update(status='search_error', error_type=type(exc).__name__)
        return result
    result['catalogue_scene_count'] = len(items)
    ranked = []
    for item in items:
        attempt = dict(scene_id=item.get('id'), captured_at=item.get('properties', {}).get('datetime'))
        try:
            check = temporal_check(attempt['captured_at'], start, end, max_days)
            attempt.update(check)
            if check['temporal_status'] != 'within_tolerance':
                attempt['status'] = check['temporal_status']
            elif not item.get('geometry') or not shape(item['geometry']).covers(Point(longitude, latitude)):
                attempt['status'] = 'outside_scene_footprint'
            else:
                ranked.append((check['time_delta_max_hours'], item['id'], item, attempt))
                continue
        except (ValueError, TypeError, KeyError, AttributeError):
            attempt['status'] = 'invalid_scene_metadata'
        result['attempts'].append(attempt)
    had_error = False
    for _, _, item, attempt in sorted(ranked, key=lambda v: (v[0], v[1])):
        try:
            arrays, provenance = reader(item, latitude, longitude, config)
            qa = assess(arrays, lake_mask, neighborhood, config)
            attempt.update(status='qa_passed' if qa['passed'] else 'qa_rejected', qa=qa, raster=provenance,
                           scene_cloud_cover=item.get('properties', {}).get('eo:cloud_cover'),
                           processing_baseline=item.get('properties', {}).get('s2:processing_baseline'))
            result['attempts'].append(attempt)
            if qa['passed']:
                result.update(status='passed_pending_review', scene_id=item['id'], captured_at=attempt['captured_at'],
                              features=qa['features'], qa=qa, feature_version=FEATURE_VERSION,
                              time_delta_min_hours=attempt['time_delta_min_hours'], time_delta_max_hours=attempt['time_delta_max_hours'])
                return result
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            attempt.update(status='imagery_error', error_type=type(exc).__name__)
            result['attempts'].append(attempt)
            had_error = True
    if items:
        result['status'] = 'imagery_error' if had_error else 'no_acceptable_scene'
    return result


def run_matching(candidates='data/processed/nla2017/chlorophyll_candidates.csv',
                 preprocessing_report='data/processed/nla2017/preprocessing_report.json',
                 polygons='data/raw/nla2017/NLA17_polgons.zip', raw_manifest='data/manifests/nla2017_raw.json',
                 output_dir='data/processed/nla2017/matching', max_days=3, workers=4, limit=None,
                 config=None, catalogue=None, reader=read_aligned, progress=None):
    if not 0 < max_days <= 7 or not 1 <= workers <= 8 or (limit is not None and limit < 1):
        raise ValueError('Require 0 < max-days <= 7, 1–8 workers, and positive limit')
    config = config or QualityConfig()
    output = Path(output_dir)
    prep = json.loads(Path(preprocessing_report).read_text())
    expected = [entry['sha256'] for entry in prep['outputs'] if entry['filename'] == Path(candidates).name]
    if expected != [digest(candidates)]:
        raise ValueError('Candidate file is not verified by preprocessing report')
    raw = json.loads(Path(raw_manifest).read_text())
    if [f['sha256'] for f in raw['files'] if f['filename'] == Path(polygons).name] != [digest(polygons)]:
        raise ValueError('Lake polygon checksum mismatch')
    with Path(candidates).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row['status'] != 'candidate' or row['unit'] != 'ug/L' or row['collection_time_precision'] != 'date' for row in rows):
        raise ValueError('Expected nonempty, date-only candidate observations in ug/L')
    if len({r['observation_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate observation IDs')
    settings = dict(version=VERSION, feature_version=FEATURE_VERSION, max_days=max_days, quality=asdict(config),
                    candidates_sha256=digest(candidates), polygons_sha256=digest(polygons),
                    implementation_sha256=digest(__file__), quality_code_sha256=digest(_imagery_quality.__file__),
                    date_policy='unknown-time-and-timezone envelope: UTC offsets -12 to +14; worst-case gap <= tolerance')
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    cache = output / 'checkpoints' / fingerprint
    lakes = LakePolygons(polygons)
    results, jobs = {}, []
    for row in rows:
        key = row['observation_id']
        cache_path = cache / (hashlib.sha256(key.encode()).hexdigest() + '.json')
        lake, reason = lakes.get(row['site_id'], float(row['latitude']), float(row['longitude']))
        if reason:
            results[key] = dict(observation_id=key, lake_id=row['lake_id'], site_id=row['site_id'], status='spatial_rejected', reason=reason, features={}, review_status='rejected')
        elif cache_path.exists():
            previous = json.loads(cache_path.read_text())
            if previous['status'] not in ('search_error', 'imagery_error'):
                results[key] = previous
            else:
                jobs.append((row, lake, cache_path))
        else:
            jobs.append((row, lake, cache_path))
    selected = jobs if limit is None else jobs[:limit]
    for row, _, _ in jobs[len(selected):]:
        results[row['observation_id']] = dict(observation_id=row['observation_id'], status='not_processed', reason='run_limit', features={})
    catalogue = catalogue or Catalogue()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(match_one, row, lake, lakes.crs, config, max_days, catalogue, reader): (row, path)
                   for row, lake, path in selected}
        for future in as_completed(futures):
            row, path = futures[future]
            result = future.result()
            write_json(path, result)
            results[row['observation_id']] = result
            if progress:
                progress(len(results), len(rows), result['status'])
    ordered = [results[row['observation_id']] for row in rows]
    write_json(output / 'matches.json', ordered)
    passed = [r for r in ordered if r['status'] == 'passed_pending_review']
    counts = dict(sorted(Counter(r['status'] for r in ordered).items()))
    report = dict(settings=settings, run_fingerprint=fingerprint, candidates=len(rows), counts=counts,
                  distinct_passed_lakes=len({r['lake_id'] for r in passed}), approved_training_pairs=0,
                  spatial_reasons=dict(Counter(r['reason'] for r in ordered if r['status'] == 'spatial_rejected')),
                  scene_attempt_statuses=dict(Counter(a['status'] for r in ordered for a in r.get('attempts', []))),
                  qa_reasons=dict(Counter(reason for r in ordered for a in r.get('attempts', []) for reason in a.get('qa', {}).get('reasons', []))),
                  complete=not any(r['status'] in ('not_processed', 'search_error', 'imagery_error') for r in ordered),
                  outputs=[dict(filename='matches.json', sha256=digest(output / 'matches.json'))],
                  limitations=['Automated passes require review; source observations remain provisional.',
                               'SCL water-only screening can exclude high-algae water and small/narrow lakes.',
                               'Cloud masks and lake polygons are imperfect; glint and adjacency effects remain.',
                               'A three-day temporal tolerance is a screening choice, not proof of stable chlorophyll.'])
    write_json(output / 'matching_report.json', report)
    return report
