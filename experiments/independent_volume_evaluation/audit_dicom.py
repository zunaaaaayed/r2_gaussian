"""Read-only geometry/identity audit of a public CT series ZIP; no pixel decoding."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_cohort(m, directory):
    assert digest({k:v for k,v in m.items() if k!='manifest_sha256'}) == m['manifest_sha256'], 'cohort hash mismatch'
    subjects, series = set(), set()
    counts = dict(development=0, validation=0, final_test=0)
    for r in m['subjects']:
        assert r['subject_id'] not in subjects and r['subject_id'] not in m['excluded_subjects'], 'subject overlap'
        assert r['series_uid'] not in series, 'duplicate series'
        subjects.add(r['subject_id']); series.add(r['series_uid']); counts[r['role']] += 1
        p = (directory / r['metadata_file']).resolve()
        assert directory.resolve() in p.parents, 'metadata path escape'
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['metadata_sha256'], 'metadata changed'
        rows = json.loads(p.read_text())
        assert len(rows)==1, 'ambiguous series'
        row = rows[0]
        assert (row['PatientID'],row['StudyInstanceUID'],row['SeriesInstanceUID'],row['ImageCount']) == (r['subject_id'],r['study_uid'],r['series_uid'],r['expected_image_count']), 'series identity mismatch'
    assert counts == dict(development=2, validation=2, final_test=2), 'incorrect split counts'


def audit(path, record):
    import numpy as np
    import pydicom
    headers=[]; other=[]
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        for name in archive.namelist():
            if name.endswith('/'):
                continue
            with archive.open(name) as stream:
                try:
                    ds=pydicom.dcmread(stream,stop_before_pixels=True)
                except pydicom.errors.InvalidDicomError:
                    other.append(name); continue
            if getattr(ds,'Modality',None)!='CT':
                other.append(name); continue
            assert str(ds.PatientID)==record['subject_id'], 'subject mismatch'
            assert str(ds.StudyInstanceUID)==record['study_uid'], 'study mismatch'
            assert str(ds.SeriesInstanceUID)==record['series_uid'], 'series mismatch'
            assert int(getattr(ds,'NumberOfFrames',1))==1, 'multiframe unsupported'
            headers.append({'sop_uid':str(ds.SOPInstanceUID),'position':list(map(float,ds.ImagePositionPatient)),
                            'orientation':list(map(float,ds.ImageOrientationPatient)),
                            'spacing':list(map(float,ds.PixelSpacing)), 'shape':[int(ds.Rows),int(ds.Columns)],
                            'slope':float(ds.RescaleSlope),'intercept':float(ds.RescaleIntercept)})
    assert len(headers)==record['expected_image_count'] and len(headers)>1, 'image count mismatch'
    assert len({h['sop_uid'] for h in headers})==len(headers), 'duplicate SOP'
    first=headers[0];orientation=np.array(first['orientation']);u,v=orientation[:3],orientation[3:]
    assert np.isfinite(orientation).all() and np.allclose([np.linalg.norm(u),np.linalg.norm(v),np.dot(u,v)],[1,1,0],atol=1e-5), 'invalid direction cosines'
    normal=np.cross(u,v)
    for h in headers:
        assert h['shape']==first['shape'], 'varying image shape'
        assert np.allclose(h['orientation'],orientation,atol=1e-5), 'varying orientation'
        assert np.allclose(h['spacing'],first['spacing'],atol=1e-6), 'varying pixel spacing'
        assert np.isfinite(h['position']).all() and np.isfinite([h['slope'],h['intercept']]).all(), 'nonfinite header'
    assert min(first['spacing'])>0, 'invalid spacing'
    positions=np.array([h['position'] for h in headers]); projected=positions@normal
    order=np.argsort(projected); increments=np.diff(positions[order],axis=0); dz=np.diff(projected[order])
    assert np.all(dz>1e-5), 'duplicate slice positions'
    assert np.allclose(dz,np.median(dz),rtol=1e-3,atol=1e-3), 'irregular slice spacing'
    assert np.allclose(increments,dz[:,None]*normal,atol=1e-3), 'in-plane shift/gantry tilt unsupported'
    return {'subject_id':record['subject_id'],'series_uid':record['series_uid'],'image_count':len(headers),
            'shape_rows_columns_slices':first['shape']+[len(headers)],'pixel_spacing_mm':first['spacing'],
            'slice_spacing_mm':float(np.median(dz)),'orientation':first['orientation'],
            'sort_rule':'ascending ImagePositionPatient dot cross(ImageOrientationPatient vectors)',
            'rescale_pairs':sorted({(h['slope'],h['intercept']) for h in headers}),
            'first_sorted_position':positions[order][0].tolist(),'last_sorted_position':positions[order][-1].tolist(),
            'other_zip_members':other,'geometry_header_audit':'passed',
            'pixel_values_checked':False,'preprocessing_complete':False,'pilot_independence_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--subject',required=True)
    parser.add_argument('--zip',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    assert not args.report.exists(), 'report already exists'
    m=json.loads(args.cohort.read_text());validate_cohort(m,args.cohort.parent)
    record=next(r for r in m['subjects'] if r['subject_id']==args.subject)
    result=audit(args.zip,record);result['cohort_sha256']=m['manifest_sha256']
    h=hashlib.sha256()
    with args.zip.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    result['raw_zip_sha256']=h.hexdigest()
    with args.report.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
