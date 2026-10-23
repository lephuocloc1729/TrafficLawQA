from types import SimpleNamespace
from unittest.mock import Mock

from src.core import freeform_qa as qa
from src.core import sub_task_2


def test_preparation_reused_across_questions_without_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(qa.settings, 'image_query_detection_model', 'Yolo26s')
    monkeypatch.setattr(qa.settings, 'enable_symbolic_reasoning', True)
    monkeypatch.setattr(qa, '_prepare_demo_image', lambda path, directory, image_id: path)
    extract = Mock(side_effect=lambda **kw: {**kw['sample'], 'detected_signs': [
        {'image_name': 'sign.jpg', 'sign_name': 'P.102'}]})
    context = ([], [], [], {'signs': [{'id': 'P.102'}]})
    collect = Mock(return_value=context)
    llm = Mock(side_effect=AssertionError('LLM must not be called'))
    resources = SimpleNamespace(sign_extractor=object(), embedder=None,
                                vector_store=None, article_lookup=object(), llm_client=llm)
    answer = Mock(side_effect=lambda **kw: {
        'freeform_answer': kw['sample']['question'],
        'detected_signs': kw['sample']['detected_signs'],
    })
    monkeypatch.setattr(qa, 'extract_signs', extract)
    monkeypatch.setattr(sub_task_2, 'collect_retrieved_context', collect)
    monkeypatch.setattr(qa, 'get_freeform_answer', answer)
    prepared = qa.answer_image_question(tmp_path / 'image.jpg', '', resources=resources,
                                        use_vlm_filter=False, prepare_only=True)
    answer.assert_not_called()
    assert prepared['context'] == context
    for question in ['Biển gì?', 'Có được đi vào không?']:
        result = qa.answer_image_question(tmp_path / 'image.jpg', question,
                                         resources=resources, prepared=prepared)
        assert result['freeform_answer'] == question
        assert answer.call_args.kwargs['prepared_context'] == context
        assert answer.call_args.kwargs['sample']['question'] == question
    extract.assert_called_once()
    collect.assert_called_once()
    assert prepared['sample']['question'] == ''
    llm.assert_not_called()
