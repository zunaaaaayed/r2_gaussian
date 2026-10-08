import copy
import json
from pathlib import Path
import unittest
import audit_dicom as a

class CohortTest(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent
        self.m=json.loads((self.root/'cohort_v1.json').read_text())

    def rehash(self):
        self.m['manifest_sha256']=a.digest({k:v for k,v in self.m.items() if k!='manifest_sha256'})

    def test_real_metadata_and_split(self):
        a.validate_cohort(self.m,self.root)

    def test_duplicate_subject_rejected_even_after_rehash(self):
        self.m['subjects'][1]=copy.deepcopy(self.m['subjects'][0]);self.rehash()
        with self.assertRaisesRegex(AssertionError,'subject overlap'):a.validate_cohort(self.m,self.root)

    def test_candidate_pilot_excluded(self):
        self.m['subjects'][0]['subject_id']='LIDC-IDRI-0001';self.rehash()
        with self.assertRaisesRegex(AssertionError,'subject overlap'):a.validate_cohort(self.m,self.root)

    def test_series_tamper_rejected_even_after_rehash(self):
        self.m['subjects'][0]['series_uid']='1.2.3';self.rehash()
        with self.assertRaisesRegex(AssertionError,'identity mismatch'):a.validate_cohort(self.m,self.root)

    def test_split_tamper_rejected(self):
        self.m['subjects'][0]['role']='final_test';self.rehash()
        with self.assertRaisesRegex(AssertionError,'split counts'):a.validate_cohort(self.m,self.root)

if __name__=='__main__':unittest.main()
