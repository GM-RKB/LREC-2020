import unittest
import sys
sys.path.insert(0, 'tools/')
from fixer_evaluation import get_diff_log, get_diff_distance,get_eval_metrics, compare_diff_logs,get_diff_context


class TestTools(unittest.TestCase):
    original_text=  "A[[Data Science Task]] is a[[science task]] that optimizes[[data-driven decision-making task]]"
    noisy_text =    "A[[Dta Science aTsk]] is a[[szience task]] that optzimizes[[data-driven decision-making task]]"
    fixed_text =    "A[[Data Science aTsk]] is a[[science task]] thzt optimizes[[data-driven decision-making task]]"

    log_ON = [{'type': 2, 'pos': 4, 'chars': ['a']}, {'type': 1, 'pos': 15, 'chars': ['T', 'a']}, {
        'type': 4, 'pos': 29, 'chars': ['c', 'z']}, {'type': 3, 'pos': 51, 'chars': ['z']}]

    log_FN = [{'type': 2, 'pos': 4, 'chars': ['a']}, {'type': 4, 'pos': 29, 'chars': ['c', 'z']}, {
        'type': 4, 'pos': 45, 'chars': ['z', 'a']}, {'type': 3, 'pos': 51, 'chars': ['z']}]

    log_diff = ([{'type': 2, 'pos': 4, 'chars': ['a']}, {'type': 4, 'pos': 29, 'chars': ['c', 'z']}, {'type': 3, 'pos': 51, 'chars': ['z']}], 
                [{'type': 1, 'pos': 15, 'chars': ['T', 'a']}], 
                [{'type': 4, 'pos': 45, 'chars': ['z', 'a']}], 
                [])
    
    context_diff = [{'o_word': 'A[[Dta', 'f_word': 'A[[Data', 'f_pos': [4, 5]}, {'o_word': 'a[[szience', 'f_word': 'a[[science', 'f_pos': [29, 30]}, {
        'o_word': 'that', 'f_word': 'thzt', 'f_pos': [45, 46]}, {'o_word': 'optzimizes[[data-driven', 'f_word': 'optimizes[[data-driven', 'f_pos': [51, 51]}]



    def test_get_diff_log(self):
        self.assertEqual(get_diff_log(self.noisy_text, self.original_text), self.log_ON)
        self.assertEqual(get_diff_log(self.noisy_text, self.fixed_text), self.log_FN)

    def test_get_diff_distance(self):
        self.assertEqual(get_diff_distance(self.noisy_text,self.original_text), 5)
        self.assertEqual(get_diff_distance(self.noisy_text,self.fixed_text), 4)
        self.assertEqual(get_diff_distance(self.original_text,self.fixed_text), 3)
          
    def test_get_eval_metrics(self):
        self.assertEqual(get_eval_metrics(self.original_text,self.noisy_text,self.fixed_text),(3,1,1,0))
    
    def test_compare_diff_logs(self):
        self.assertEqual(compare_diff_logs(self.log_ON, self.log_FN), self.log_diff)

    def test_get_diff_context(self):
        self.assertEqual(get_diff_context(self.noisy_text, self.fixed_text),self.context_diff)

        t1="A[[Dzta Science Task]] is a[[science task]]"
        t2 = "A[[Data Science Task]] is a[[science task]]"
        self.assertEqual(get_diff_context(t1, t2), [{'o_word': 'A[[Dzta', 'f_word': 'A[[Data', 'f_pos': [4, 4]}])

        t1="A[[Data Science Task]] is a[[science tsak]]"
        self.assertEqual(get_diff_context(t1, t2), [{'o_word': 'tsak]]', 'f_word': 'task]]', 'f_pos': [38, 38]}])

