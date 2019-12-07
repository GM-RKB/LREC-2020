import unittest
from WikiFixerMLE import WikiFixer



class TestWikiFixerMLE(unittest.TestCase):
    test_text = "'''See:''' [[User]], [[Behavior]], [[Click-Through Data]], [[User Interface]].\n----\n\n__NOTOC__\n[[Category:Stub]]"
    train_data ="Welcome to my knowledge base: A semantic wiki with approximately 31, 729 pages each of which refers to either a concept, publication or person related to ...\nResearch themes such as: \nMachine learning algorithms, especially semi-supervised ones applied to sequence segmentation, graph-edge prediction, and dimensionality reduction.\nNatural language processing tasks, such as: semantic annotation, information extraction, text summarization, and knowledge engineering."*40
    

    def test_check_text(self):
        fixer = WikiFixer()
        self.assertTrue(fixer.check_text("https://www.gabormelli.com/RKB/HomePage"))
        self.assertFalse(fixer.check_text(self.test_text))
    
    def test_train_model(self):
        fixer = WikiFixer()
        fixer.train_model(text=self.train_data)
        history="sema"
        self.assertEqual(fixer.models[0][0].print_probs(history),{'n':1.0})

    def test_load_models(self):
        fixer = WikiFixer()
        fixer.load_models(models=["/mnt/efs/models/model6-0.json"])
        
    def test_fix_text(self):
        fixer = WikiFixer()
        fixer.load_models(models=["/mnt/efs/models/model6-0.json"])
        self.assertEqual(fixer.fix_text("== References =="), "== References ==")
        self.assertEqual(fixer.fix_text("== Referznces =="), "== References ==")
        


