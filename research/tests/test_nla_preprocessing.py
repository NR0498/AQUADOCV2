"""Source-contract tests independent of the locally downloaded EPA files."""
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

from research.lab_matching.cli import main
from research.lab_matching.nla_preprocessing import REQUIRED, STEMS, digest, preprocess


class NLAPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.out = self.root / 'processed'
        self.manifest = self.root / 'inventory.json'
        common = dict(UID='123', SITE_ID='LAKE-1', VISIT_NO='1', DATE_COL='30-May-17')
        self.tables = {
            'chemistry': [dict(common, ANALYTE='CHLA', RESULT='12.5', RESULT_UNITS='UG/L',
                               MDL='0.39', RL='', NARS_FLAG='', LAB='LAB', DATE_ANALYZED='',
                               MATRIX='WATER', SAMPLE_TYPE='CHLX', SAM_CODE='REGULAR', LAB_SAMPLE_ID='L1')],
            'sites': [dict(common, DATE_COL='5/30/2017', UNIQUE_ID='stable-lake', SITESAMP='Y',
                           LAT_DD83='40.5', LON_DD83='-94.1')],
            'profiles': [dict(common, INDEX_LAT_DD='40.31573', INDEX_LON_DD='w94.03035',
                              INDEX_GPS_FIX='3D', FLAG='')],
        }

    def run_pipeline(self):
        entries = []
        for role, stem in STEMS.items():
            path = self.raw / (stem + '-data.csv')
            with path.open('w', encoding='cp1252' if role == 'profiles' else 'utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED[role]))
                writer.writeheader()
                writer.writerows(self.tables[role])
            metadata = self.raw / (stem + '-metadata.txt')
            metadata.write_text('Fixture metadata\n')
            entries.extend(dict(filename=p.name, sha256=digest(p)) for p in (path, metadata))
        self.manifest.write_text(json.dumps({'files': entries}))
        return preprocess(self.raw, self.out, self.manifest)

    def rows(self, name='chlorophyll_observations.csv'):
        with (self.out / name).open(newline='') as f:
            return list(csv.DictReader(f))

    def test_exact_analyte_many_profile_rows_and_date_precision(self):
        self.tables['chemistry'].append(dict(self.tables['chemistry'][0], ANALYTE='CHL'))
        self.tables['profiles'].append(dict(self.tables['profiles'][0], INDEX_LAT_DD='', INDEX_LON_DD='', FLAG='depth-only'))
        report = self.run_pipeline()
        self.assertEqual(report['counts']['candidate_rows'], 1)
        self.assertEqual(report['counts']['non_chla_rows_filtered'], 1)
        row = self.rows()[0]
        self.assertEqual(row['profile_rows'], '2')
        self.assertEqual(row['collected_date'], '2017-05-30')
        self.assertEqual(row['collection_time_precision'], 'date')
        self.assertEqual(row['collection_timezone'], '')
        self.assertEqual(row['analyzed_date'], '')
        self.assertNotIn('collected_at', row)
        self.assertEqual(row['location_group'], 'nla2017:stable-lake')
        self.assertAlmostEqual(float(row['longitude']), -94.03035, places=4)
        self.assertEqual(row['quality_status'], 'provisional')
        self.assertEqual(row['satellite_match_status'], 'unpaired')

    def test_flags_and_censoring_preserve_original_values(self):
        self.tables['chemistry'][0].update(RESULT='0.1', NARS_FLAG='S')
        report = self.run_pipeline()
        self.assertEqual(report['counts']['review_rows'], 1)
        self.assertEqual(self.rows('chlorophyll_candidates.csv'), [])
        row = self.rows('chlorophyll_review.csv')[0]
        self.assertEqual(row['value'], '0.1')
        self.assertEqual(set(row['review_reasons'].split(';')), {'below_mdl', 'nars_flag_requires_review'})
        self.assertEqual(json.loads(row['source_record_json'])['RESULT'], '0.1')

    def test_unit_conversion_applies_to_both_limits(self):
        self.tables['chemistry'][0].update(RESULT='0.01', RESULT_UNITS='MG/L', MDL='0.0001', RL='0.0002')
        self.run_pipeline()
        row = self.rows()[0]
        self.assertEqual((row['value'], row['mdl'], row['rl'], row['unit']), ('10.0', '0.1', '0.2', 'ug/L'))

    def test_bad_and_missing_coordinates_do_not_use_centroid(self):
        for lat, lon, reason in [('43..14', '-94.03', 'invalid_index_coordinates'),
                                 ('', '', 'missing_index_coordinates')]:
            with self.subTest(reason=reason):
                self.tables['profiles'][0].update(INDEX_LAT_DD=lat, INDEX_LON_DD=lon)
                self.run_pipeline()
                row = self.rows()[0]
                self.assertIn(reason, row['review_reasons'])
                self.assertEqual(row['latitude'], '')
                self.assertEqual(row['site_latitude_nad83'], '40.5')

    def test_duplicate_site_and_chlorophyll_never_silently_overwrite(self):
        self.tables['sites'].append(dict(self.tables['sites'][0]))
        self.tables['chemistry'].append(dict(self.tables['chemistry'][0], RESULT='88'))
        self.run_pipeline()
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r['observation_id'] for r in rows}), 2)
        for row in rows:
            self.assertIn('site_join_not_one_to_one', row['review_reasons'])
            self.assertIn('duplicate_chlorophyll_visit', row['review_reasons'])
            self.assertEqual(row['status'], 'review')

    def test_conflicting_profile_and_join_dates(self):
        self.tables['profiles'].append(dict(self.tables['profiles'][0], INDEX_LAT_DD='41.1'))
        self.tables['sites'][0]['DATE_COL'] = '5/31/2017'
        self.run_pipeline()
        row = self.rows()[0]
        self.assertIn('conflicting_index_coordinates', row['review_reasons'])
        self.assertIn('site_identity_or_date_mismatch', row['review_reasons'])
        self.assertEqual(row['latitude'], '')

    def test_nonfinite_and_negative_concentrations_and_unknown_units(self):
        for value, unit, reason in [('nan', 'UG/L', 'invalid_value'), ('inf', 'UG/L', 'invalid_value'),
                                     ('-1', 'UG/L', 'invalid_value'), ('1', 'PPM', 'unsupported_unit')]:
            with self.subTest(value=value, unit=unit):
                self.tables['chemistry'][0].update(RESULT=value, RESULT_UNITS=unit)
                self.run_pipeline()
                self.assertIn(reason, self.rows()[0]['review_reasons'])
                self.assertEqual(self.rows()[0]['value'], '')

    def test_checksum_failure_leaves_existing_output_untouched(self):
        self.run_pipeline()
        previous = digest(self.out / 'chlorophyll_observations.csv')
        with (self.raw / (STEMS['chemistry'] + '-data.csv')).open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            preprocess(self.raw, self.out, self.manifest)
        self.assertEqual(previous, digest(self.out / 'chlorophyll_observations.csv'))

    def test_repeat_run_is_identical_and_raw_directory_protected(self):
        first = self.run_pipeline()
        report_hash = digest(self.out / 'preprocessing_report.json')
        second = preprocess(self.raw, self.out, self.manifest)
        self.assertEqual(first, second)
        self.assertEqual(report_hash, digest(self.out / 'preprocessing_report.json'))
        with self.assertRaisesRegex(ValueError, 'outside the raw'):
            preprocess(self.raw, self.raw, self.manifest)

    def test_cli_writes_report(self):
        self.run_pipeline()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(['preprocess-nla2017', '--raw-dir', str(self.raw),
                         '--output-dir', str(self.out), '--manifest', str(self.manifest)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())['counts']['candidate_rows'], 1)


if __name__ == '__main__':
    unittest.main()
