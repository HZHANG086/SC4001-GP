import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sc4001_c.common import dump,digest
from sc4001_c.m5_import import MANIFEST,main,verified_existing


class RawReturnReproductionTests(unittest.TestCase):
    def fixture(self,root):
        raw=root/'raw';name='member-c/results/example.csv'
        path=raw/name;path.parent.mkdir(parents=True);path.write_text('synthetic fixture only\n')
        dump(raw/MANIFEST,{'files':{name:digest(path)}})
        return raw,path

    def test_raw_cli_reaudits_without_archive_or_mutating_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,path=self.fixture(root);before=digest(raw/MANIFEST)
            with patch('sc4001_c.m5_import.audit_local') as audit:
                main(['--raw',str(raw),'--repo',str(root),'--out',str(root/'audit')])
                audit.assert_called_once_with(raw,root,root/'audit')
            record=json.loads((root/'audit/import.json').read_text())
            self.assertEqual(record['mode'],'existing_raw')
            self.assertNotIn('archive_sha256',record)
            self.assertEqual(digest(raw/MANIFEST),before)
            self.assertEqual(path.read_text(),'synthetic fixture only\n')

    def test_corrupt_missing_and_unmanifested_inputs_rejected_before_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,path=self.fixture(root)
            verified_existing(raw)
            path.write_text('changed prediction\n')
            with patch('sc4001_c.m5_import.audit_local') as audit:
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                    main(['--raw',str(raw),'--out',str(root/'audit')])
                audit.assert_not_called()
            path.unlink()
            with self.assertRaisesRegex(ValueError,'missing files'):verified_existing(raw)
            path.write_text('synthetic fixture only\n');(raw/'extra.csv').write_text('unexpected\n')
            with self.assertRaisesRegex(ValueError,'Unmanifested'):verified_existing(raw)

    def test_links_traversal_and_output_inside_raw_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw,path=self.fixture(root)
            (raw/'link').symlink_to(path)
            with self.assertRaisesRegex(ValueError,'Symlinks'):verified_existing(raw)
            (raw/'link').unlink()
            with self.assertRaises(SystemExit):
                main(['--raw',str(raw),'--out',str(raw/'audit')])
            dump(raw/MANIFEST,{'files':{'../outside':digest(path)}})
            with self.assertRaisesRegex(ValueError,'Unsafe'):verified_existing(raw)


if __name__=='__main__':unittest.main()
