"""CPU mathematical checks; these fixtures are not plotted as experiments."""
from pathlib import Path
import tempfile
import unittest

from diagnose_margin import detection_statistics, first_margin, selected_references


class MarginDiagnosticChecks(unittest.TestCase):
    def test_low_margin_risk_auc_and_fixed_threshold_direction(self):
        rows=[dict(image_id=image,margin=margin,error=label) for image,margin,label in
              (("a",.1,1),("a",2.,0),("b",.2,1),("b",4.,0))]
        result=detection_statistics(rows,"error",thresholds=(.1,1.),repeats=200)
        self.assertEqual(result["summary"]["auroc"],1)
        self.assertEqual(result["thresholds"][0]["triggered"],0)
        self.assertIsNone(result["thresholds"][0]["precision"])
        self.assertEqual(result["thresholds"][1]["precision"],1)
        self.assertEqual(result["thresholds"][1]["recall"],1)
        self.assertEqual(result["thresholds"][1]["trigger_rate"],.5)

    def test_one_class_auroc_is_na(self):
        result=detection_statistics([dict(image_id="a",margin=1,error=0),
                                     dict(image_id="b",margin=2,error=0)],"error",repeats=100)
        self.assertIsNone(result["summary"]["auroc"])
        self.assertEqual(result["summary"]["auroc_status"],"N/A_one_class")

    def test_empty_answer_error_is_retained_in_operational_recall(self):
        rows=[dict(image_id="a",margin=None,error=1),dict(image_id="b",margin=.2,error=1),
              dict(image_id="b",margin=4.,error=0)]
        result=detection_statistics(rows,"error",thresholds=(1.,),repeats=100)
        row=result["thresholds"][0]
        self.assertEqual(row["recall"],1)
        self.assertEqual(row["operational_recall"],.5)
        self.assertEqual(row["missing_margin_positives"],1)

    def test_only_first_nontermination_margin_is_used(self):
        self.assertEqual(first_margin(dict(token_ids=[17,99,151645],margin=[2.,.01,8.]),{151645}),
                         (0,17,2.))
        with self.assertRaises(ValueError):
            first_margin(dict(token_ids=[17],margin=[]),{151645})

    def test_reference_selection_excludes_nonselected_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"refs.jsonl"
            path.write_text('{"id":"dev","split":"development","references":[{"turn":1,"answer":"red"}]}\n'
                            '{"id":"heldout","split":"heldout","references":[{"turn":1,"answer":"blue"}]}\n')
            result=selected_references(path,{"dev"})
            self.assertEqual(result,{("dev",1):"red"})


if __name__=="__main__":
    unittest.main()
