import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from sc4001_c.common import dump
from sc4001_c.m5_package import main
from sc4001_c.m5_import import verified_extract,MANIFEST


class ReturnPackageTests(unittest.TestCase):
    def test_disabled_accounting_preserves_evidence_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);previous=Path.cwd()
            try:
                os.chdir(root)
                dump(Path('member-c/results/M5-diagnostics/audit.json'),{'status':'PASS','fixture':True})
                run=Path('member-c/runs/M5-v1/synthetic-run')
                dump(run/'run_config.json',{'fixture':True})
                (run/'best').mkdir();(run/'best/model.safetensors').write_bytes(b'SYNTHETIC_TEST_ONLY')
                (run/'checkpoints').mkdir();(run/'checkpoints/optimizer.pt').write_bytes(b'SYNTHETIC_TEST_ONLY')
                def unavailable(command,**kwargs):
                    return subprocess.CompletedProcess(command,1,'','Slurm accounting storage is disabled' if command[0]=='sacct' else 'Invalid job id specified')
                with patch('subprocess.run',side_effect=unavailable):main(['--out',str(root/'return')])
                with tarfile.open(root/'return/M5-return-v1.tar.gz') as archive:
                    names=archive.getnames()
                    self.assertFalse(any('/best/' in n or '/checkpoints/' in n for n in names))
                    manifest=json.load(archive.extractfile('member-c/results/M5-diagnostics/RETURN_MANIFEST.json'))
                    for name,sha in manifest['files'].items():
                        self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(),sha)
                    evidence=json.load(archive.extractfile('member-c/results/M5-diagnostics/slurm-at-package.json'))
                    self.assertEqual(evidence['sacct_returncode'],1)
                    self.assertIn('disabled',evidence['sacct_stderr'])
                self.assertTrue((run/'best/model.safetensors').exists())
                self.assertTrue((run/'checkpoints/optimizer.pt').exists())
                imported=root/'imported'
                verified_extract(root/'return/M5-return-v1.tar.gz',imported)
                self.assertTrue((imported/'member-c/runs/M5-v1/synthetic-run/run_config.json').exists())
                with self.assertRaisesRegex(ValueError,'existing import'):
                    verified_extract(root/'return/M5-return-v1.tar.gz',imported)
            finally:os.chdir(previous)

    def test_return_import_rejects_path_traversal_and_tampering(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            def bundle(path,entries):
                with tarfile.open(path,'w:gz') as archive:
                    for name,data in entries:
                        info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
            bundle(root/'unsafe.tar.gz',[('../outside',b'bad')])
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                verified_extract(root/'unsafe.tar.gz',root/'unsafe-out')
            self.assertFalse((root/'unsafe-out').exists())
            manifest=json.dumps({'files':{'member-c/data.txt':hashlib.sha256(b'original').hexdigest()}}).encode()
            bundle(root/'changed.tar.gz',[(MANIFEST,manifest),('member-c/data.txt',b'changed')])
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                verified_extract(root/'changed.tar.gz',root/'changed-out')
            self.assertFalse((root/'changed-out').exists())


if __name__=='__main__':unittest.main()
