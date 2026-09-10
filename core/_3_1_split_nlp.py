from core.spacy_utils import *
from core.utils.models import _3_1_SPLIT_BY_NLP
from core.utils import check_file_exists
from core.utils.step_diagnostics import diagnose_stage, diagnostic_step

@diagnose_stage("nlp")
@check_file_exists(_3_1_SPLIT_BY_NLP)
def split_by_spacy():
    with diagnostic_step("nlp.model_init"):
        nlp = init_nlp()
    with diagnostic_step("nlp.split_by_mark"):
        split_by_mark(nlp)
    with diagnostic_step("nlp.split_by_comma"):
        split_by_comma_main(nlp)
    with diagnostic_step("nlp.split_by_connector"):
        split_sentences_main(nlp)
    with diagnostic_step("nlp.split_by_root"):
        split_long_by_root_main(nlp)
    return

if __name__ == '__main__':
    split_by_spacy()
